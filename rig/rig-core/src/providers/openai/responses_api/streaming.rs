//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{CompletionError, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::message::ReasoningContent;
use crate::providers::openai::responses_api::{
    ReasoningSummary, ResponsesCompletionModel, ResponsesUsage,
};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use crate::wasm_compat::WasmCompatSend;
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, enabled, info_span};
use tracing_futures::Instrument as _;

use super::{CompletionResponse, Output};

// ================================================================
// OpenAI Responses Streaming API
// ================================================================

/// A streaming completion chunk.
/// Streaming chunks can come in one of two forms:
/// - A response chunk (where the completed response will have the total token usage)
/// - An item chunk commonly referred to as a delta. In the completions API this would be referred to as the message delta.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamingCompletionChunk {
    Response(Box<ResponseChunk>),
    Delta(ItemChunk),
}

/// The final streaming response from the OpenAI Responses API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    /// Final provider usage, absent when no terminal event reported it.
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
}

pub(crate) fn reasoning_choices_from_done_item(
    id: &str,
    summary: &[ReasoningSummary],
    encrypted_content: Option<&str>,
) -> Vec<RawStreamingChoice<StreamingCompletionResponse>> {
    let mut choices = summary
        .iter()
        .map(|reasoning_summary| match reasoning_summary {
            ReasoningSummary::SummaryText { text } => RawStreamingChoice::Reasoning {
                id: Some(id.to_owned()),
                content: ReasoningContent::Summary(text.to_owned()),
            },
        })
        .collect::<Vec<_>>();

    if let Some(encrypted_content) = encrypted_content {
        choices.push(RawStreamingChoice::Reasoning {
            id: Some(id.to_owned()),
            content: ReasoningContent::Encrypted(encrypted_content.to_owned()),
        });
    }

    choices
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let reported = self.usage.as_ref()?;
        Some(crate::completion::Usage {
            input_tokens: reported.input_tokens,
            output_tokens: reported.output_tokens,
            total_tokens: reported.total_tokens,
            cached_input_tokens: reported
                .input_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
        })
    }
}

/// A response chunk from OpenAI's response API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseChunk {
    /// The response chunk type
    #[serde(rename = "type")]
    pub kind: ResponseChunkKind,
    /// The response itself
    #[serde(deserialize_with = "deserialize_streaming_response")]
    pub response: CompletionResponse,
    /// The item sequence
    pub sequence_number: u64,
}

/// Response chunk type.
/// Renames are used to ensure that this type gets (de)serialized properly.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ResponseChunkKind {
    #[serde(rename = "response.created")]
    ResponseCreated,
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,
    #[serde(rename = "response.completed")]
    ResponseCompleted,
    #[serde(rename = "response.failed")]
    ResponseFailed,
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete,
}

/// Streamed function arguments may stop inside JSON. Preserve the raw string
/// until the tool consumer validates it, without losing the terminal envelope.
fn retain_unparsed_arguments(item: &mut serde_json::Value) -> serde_json::Result<()> {
    if item.get("type").and_then(serde_json::Value::as_str) == Some("function_call") {
        if let Some(serde_json::Value::String(arguments)) = item.get_mut("arguments") {
            if !arguments.trim().is_empty()
                && serde_json::from_str::<serde_json::Value>(arguments).is_err()
            {
                *arguments = serde_json::to_string(arguments)?;
            }
        }
    }
    Ok(())
}

fn deserialize_streaming_output<'de, D>(deserializer: D) -> Result<Output, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = serde_json::Value::deserialize(deserializer)?;
    retain_unparsed_arguments(&mut value).map_err(serde::de::Error::custom)?;
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

fn deserialize_streaming_response<'de, D>(deserializer: D) -> Result<CompletionResponse, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut value = serde_json::Value::deserialize(deserializer)?;
    if let Some(output) = value
        .get_mut("output")
        .and_then(serde_json::Value::as_array_mut)
    {
        for item in output {
            retain_unparsed_arguments(item).map_err(serde::de::Error::custom)?;
        }
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

fn streamed_function_call(
    function: &super::OutputFunctionCall,
    identities: &mut std::collections::HashMap<String, String>,
) -> streaming::RawStreamingToolCall {
    let internal_id = identities
        .entry(function.id.clone())
        .or_insert_with(|| nanoid::nanoid!())
        .clone();
    streaming::RawStreamingToolCall::new(
        function.id.clone(),
        function.name.clone(),
        function.arguments.clone(),
    )
    .with_internal_call_id(internal_id)
    .with_call_id(function.call_id.clone())
}

fn response_error_message(error: Option<&super::ResponseError>, fallback: &str) -> String {
    if let Some(error) = error {
        if error.code.is_empty() {
            error.message.clone()
        } else {
            format!("{}: {}", error.code, error.message)
        }
    } else {
        format!("OpenAI response stream returned a {fallback}")
    }
}

fn response_chunk_error_message(
    kind: &ResponseChunkKind,
    response: &CompletionResponse,
) -> Option<String> {
    match kind {
        ResponseChunkKind::ResponseFailed => Some(response_error_message(
            response.error.as_ref(),
            "failed response",
        )),
        ResponseChunkKind::ResponseIncomplete => {
            let reason = response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str())
                .unwrap_or("unknown reason");

            Some(format!("OpenAI response stream was incomplete: {reason}"))
        }
        _ => None,
    }
}

/// An item message chunk from OpenAI's Responses API.
/// See
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ItemChunk {
    /// Item ID. Optional.
    pub item_id: Option<String>,
    /// The output index of the item from a given streamed response.
    pub output_index: u64,
    /// The item type chunk, as well as the inner data.
    #[serde(flatten)]
    pub data: ItemChunkKind,
}

/// The item chunk type from OpenAI's Responses API.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ItemChunkKind {
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(StreamingItemDoneOutput),
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(StreamingItemDoneOutput),
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(ContentPartChunk),
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(ContentPartChunk),
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(DeltaTextChunk),
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(OutputTextChunk),
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta(DeltaTextChunk),
    #[serde(rename = "response.refusal.done")]
    RefusalDone(RefusalTextChunk),
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta(DeltaTextChunkWithItemId),
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone(ArgsTextChunk),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta(SummaryTextChunk),
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone(SummaryTextChunk),
    // vLLM (and other open-weights Responses servers) stream the model's RAW
    // chain-of-thought as `response.reasoning_text.delta`, distinct from OpenAI's
    // hosted models which only expose a `reasoning_summary_text.delta`. Upstream
    // rig only decodes the summary events, so raw reasoning is silently dropped.
    // The deltas accumulate the full reasoning text; the matching `.done` event
    // is redundant for capture and is skipped by the loop's unknown-event path.
    // Uses DeltaTextChunk (no `item_id`): ItemChunk's `#[serde(flatten)]`
    // already consumes the top-level `item_id`, so an inner struct that also
    // required it would fail to deserialize and the event would be dropped.
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta(DeltaTextChunk),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingItemDoneOutput {
    pub sequence_number: u64,
    #[serde(deserialize_with = "deserialize_streaming_output")]
    pub item: Output,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentPartChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub part: ContentPartChunkPart,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPartChunkPart {
    OutputText { text: String },
    SummaryText { text: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunkWithItemId {
    pub item_id: String,
    pub content_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefusalTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub refusal: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArgsTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryPartChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    pub part: SummaryPartChunkPart,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryTextChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SummaryPartChunkPart {
    SummaryText { text: String },
}

impl<T> ResponsesCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + WasmCompatSend + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;
        request.stream = Some(true);

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Responses streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/responses")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        // let request_builder = self.client.post_reqwest("/responses").json(&request);

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        span.record("gen_ai.provider.name", "openai");
        span.record("gen_ai.request.model", &self.model);
        // Build the request with proper headers for SSE
        let client = self.client.clone();

        let mut event_source = GenericEventSource::new(client, req);

        let stream = stream! {
            let mut final_usage = None;

            let mut tool_calls: Vec<streaming::RawStreamingToolCall> = Vec::new();
            let mut tool_call_internal_ids: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            let span = tracing::Span::current();

            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::trace!("SSE connection opened");
                        tracing::info!("OpenAI stream started");
                        continue;
                    }
                    Ok(Event::Message(evt)) => {
                        // Skip heartbeat messages or empty data
                        if evt.data.trim().is_empty() {
                            continue;
                        }

                        let data = serde_json::from_str::<StreamingCompletionChunk>(&evt.data);

                        let Ok(data) = data else {
                            let err = data.unwrap_err();
                            debug!("Couldn't serialize data as StreamingCompletionResponse: {:?}", err);
                            continue;
                        };

                        if let StreamingCompletionChunk::Delta(chunk) = &data {
                            match &chunk.data {
                                ItemChunkKind::OutputItemAdded(message) => {
                                    if let StreamingItemDoneOutput { item: Output::FunctionCall(func), .. } = message {
                                        let internal_call_id = tool_call_internal_ids
                                            .entry(func.id.clone())
                                            .or_insert_with(|| nanoid::nanoid!())
                                            .clone();
                                        yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                            id: func.id.clone(),
                                            internal_call_id,
                                            content: streaming::ToolCallDeltaContent::Name(func.name.clone()),
                                        });
                                    }
                                }
                                ItemChunkKind::OutputItemDone(message) => {
                                    match message {
                                        StreamingItemDoneOutput {  item: Output::FunctionCall(func), .. } => {
                                            tool_calls.push(streamed_function_call(func, &mut tool_call_internal_ids));
                                        }

                                        StreamingItemDoneOutput {  item: Output::Reasoning {  summary, id, encrypted_content, .. }, .. } => {
                                            for reasoning_choice in reasoning_choices_from_done_item(
                                                id,
                                                summary,
                                                encrypted_content.as_deref(),
                                            ) {
                                                yield Ok(reasoning_choice);
                                            }
                                        }
                                        StreamingItemDoneOutput { item: Output::Message(msg), .. } => {
                                            yield Ok(streaming::RawStreamingChoice::MessageId(msg.id.clone()));
                                        }
                                    }
                                }
                                ItemChunkKind::OutputTextDelta(delta) => {
                                    yield Ok(streaming::RawStreamingChoice::Message(delta.delta.clone()))
                                }
                                ItemChunkKind::ReasoningTextDelta(delta) => {
                                    // Carry the reasoning item's id (from the outer ItemChunk —
                                    // the delta's own item_id is consumed by ItemChunk's flatten).
                                    // Without an id, threading this reasoning back into a later
                                    // request fails `openai_reasoning_from_core` ("an OpenAI-generated
                                    // ID is required"), which breaks multi-turn tool loops.
                                    yield Ok(streaming::RawStreamingChoice::ReasoningDelta { id: chunk.item_id.clone(), reasoning: delta.delta.clone() })
                                }
                                ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                                    yield Ok(streaming::RawStreamingChoice::ReasoningDelta { id: None, reasoning: delta.delta.clone() })
                                }
                                ItemChunkKind::RefusalDelta(delta) => {
                                    yield Ok(streaming::RawStreamingChoice::Message(delta.delta.clone()))
                                }
                                ItemChunkKind::FunctionCallArgsDelta(delta) => {
                                    let internal_call_id = tool_call_internal_ids
                                        .entry(delta.item_id.clone())
                                        .or_insert_with(|| nanoid::nanoid!())
                                        .clone();
                                    yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                        id: delta.item_id.clone(),
                                        internal_call_id,
                                        content: streaming::ToolCallDeltaContent::Delta(delta.delta.clone())
                                    })
                                }

                                _ => { continue }
                            }
                        }

                        if let StreamingCompletionChunk::Response(chunk) = data {
                            let ResponseChunk { kind, response, .. } = *chunk;

                            match kind {
                                ResponseChunkKind::ResponseCompleted => {
                                    span.record("gen_ai.response.id", response.id.as_str());
                                    span.record("gen_ai.response.model", response.model.as_str());
                                    if let Some(usage) = response.usage {
                                        final_usage = Some(usage);
                                    }
                                }
                                ResponseChunkKind::ResponseIncomplete
                                    if response.incomplete_details.as_ref()
                                        .is_some_and(|details| details.reason == "max_output_tokens") => {
                                    span.record("gen_ai.response.id", response.id.as_str());
                                    span.record("gen_ai.response.model", response.model.as_str());
                                    if let Some(usage) = response.usage {
                                        final_usage = Some(usage);
                                    }
                                    for output in &response.output {
                                        if let Output::FunctionCall(function) = output {
                                            if !tool_calls.iter().any(|call| call.id == function.id) {
                                                tool_calls.push(streamed_function_call(function, &mut tool_call_internal_ids));
                                            }
                                        }
                                    }
                                    break;
                                }
                                ResponseChunkKind::ResponseFailed | ResponseChunkKind::ResponseIncomplete => {
                                    let error_message = response_chunk_error_message(&kind, &response)
                                        .expect("terminal response should have an error message");
                                    yield Err(CompletionError::ProviderError(error_message));
                                    break;
                                }
                                _ => continue,
                            }
                        }
                    }
                    Err(crate::http_client::Error::StreamEnded) => {
                        event_source.close();
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ProviderError(error.to_string()));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            event_source.close();

            for tool_call in tool_calls {
                yield Ok(RawStreamingChoice::ToolCall(tool_call))
            }

            if let Some(usage) = &final_usage {
                span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                if let Some(cached) = usage.input_tokens_details.as_ref().and_then(|d| d.cached_tokens) {
                    span.record("gen_ai.usage.cached_tokens", cached);
                }
            }
            tracing::info!("OpenAI stream finished");

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: final_usage
            }));
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ItemChunkKind, ResponseChunk, StreamingCompletionChunk, reasoning_choices_from_done_item,
    };
    use crate::completion::CompletionModel;
    use crate::http_client::mock::MockStreamingClient;
    use crate::message::ReasoningContent;
    use crate::providers::openai::responses_api::{
        AdditionalParameters, CompletionResponse, IncompleteDetailsReason, Output,
        OutputTokensDetails, ReasoningSummary, ResponseError, ResponseObject, ResponseStatus,
        ResponsesUsage,
    };
    use crate::streaming::{RawStreamingChoice, StreamedAssistantContent};
    use bytes::Bytes;
    use futures::StreamExt;
    use serde_json::{self, json};

    use crate::{
        client::CompletionClient,
        completion::{Message, ToolDefinition},
        providers::openai,
        streaming::StreamingChat,
        tool::{Tool, ToolError},
    };

    struct ExampleTool;

    impl Default for MockStreamingClient {
        fn default() -> Self {
            Self {
                sse_bytes: Bytes::new(),
            }
        }
    }

    impl std::fmt::Debug for MockStreamingClient {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockStreamingClient")
                .finish_non_exhaustive()
        }
    }

    fn sample_response(status: ResponseStatus) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            created_at: 0,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-5.4".to_string(),
            usage: None,
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: AdditionalParameters::default(),
        }
    }

    fn sse_event_bytes(event: serde_json::Value) -> Bytes {
        Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&event).expect("event should serialize")
        ))
    }

    async fn first_error_from_event(
        event: serde_json::Value,
    ) -> crate::completion::CompletionError {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_event_bytes(event),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect_err("stream should surface a provider error")
    }

    async fn final_usage_from_event(event: serde_json::Value) -> Option<ResponsesUsage> {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_event_bytes(event),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        while let Some(item) = stream.next().await {
            match item.expect("completed stream should not error") {
                StreamedAssistantContent::Final(res) => return res.usage,
                _ => continue,
            }
        }

        panic!("stream should yield a final response");
    }

    impl Tool for ExampleTool {
        type Args = ();
        type Error = ToolError;
        type Output = String;
        const NAME: &'static str = "example_tool";

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: self.name(),
                description: "A tool that returns some example text.".to_string(),
                parameters: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                }),
            }
        }

        async fn call(&self, _input: Self::Args) -> Result<Self::Output, Self::Error> {
            let result = "Example answer".to_string();
            Ok(result)
        }
    }

    #[test]
    fn reasoning_done_item_emits_summary_then_encrypted() {
        let summary = vec![
            ReasoningSummary::SummaryText {
                text: "step 1".to_string(),
            },
            ReasoningSummary::SummaryText {
                text: "step 2".to_string(),
            },
        ];
        let choices = reasoning_choices_from_done_item("rs_1", &summary, Some("enc_blob"));

        assert_eq!(choices.len(), 3);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "rs_1" && text == "step 1"
        ));
        assert!(matches!(
            choices.get(1),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "rs_1" && text == "step 2"
        ));
        assert!(matches!(
            choices.get(2),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Encrypted(data),
            }) if id == "rs_1" && data == "enc_blob"
        ));
    }

    #[test]
    fn reasoning_done_item_without_encrypted_emits_summary_only() {
        let summary = vec![ReasoningSummary::SummaryText {
            text: "only summary".to_string(),
        }];
        let choices = reasoning_choices_from_done_item("rs_2", &summary, None);

        assert_eq!(choices.len(), 1);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "rs_2" && text == "only summary"
        ));
    }

    #[test]
    fn content_part_added_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "part": {
                "type": "output_text",
                "text": "hello"
            }
        }))
        .expect("content part event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ContentPartAdded(_)
                )
        ));
    }

    #[test]
    fn content_part_done_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 4,
            "part": {
                "type": "summary_text",
                "text": "done"
            }
        }))
        .expect("content part done event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ContentPartDone(_)
                )
        ));
    }

    #[test]
    fn reasoning_text_delta_deserializes_vllm_raw_reasoning() {
        // vLLM/d4f streams the model's RAW chain-of-thought as this event
        // (OpenAI hosted models only emit reasoning_summary_text.delta).
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "content_index": 0,
            "delta": "We",
            "item_id": "940fd02b9cc0c165",
            "output_index": 0,
            "sequence_number": 4,
            "type": "response.reasoning_text.delta"
        }))
        .expect("reasoning_text.delta event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(chunk.data, ItemChunkKind::ReasoningTextDelta(ref d) if d.delta == "We")
        ));
    }

    #[test]
    fn reasoning_summary_part_added_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 5,
            "part": {
                "type": "summary_text",
                "text": "step 1"
            }
        }))
        .expect("reasoning summary part event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ReasoningSummaryPartAdded(_)
                )
        ));
    }

    #[test]
    fn reasoning_summary_part_done_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 6,
            "part": {
                "type": "summary_text",
                "text": "step 2"
            }
        }))
        .expect("reasoning summary part done event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ReasoningSummaryPartDone(_)
                )
        ));
    }

    #[tokio::test]
    async fn response_failed_chunk_surfaces_provider_error_without_empty_code_prefix() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: String::new(),
            message: "maximum context length exceeded".to_string(),
        });

        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert_eq!(
            err.to_string(),
            "ProviderError: maximum context length exceeded"
        );
    }

    #[tokio::test]
    async fn response_failed_chunk_surfaces_provider_error_with_code_prefix() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "context_length_exceeded".to_string(),
            message: "maximum context length exceeded".to_string(),
        });

        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert_eq!(
            err.to_string(),
            "ProviderError: context_length_exceeded: maximum context length exceeded"
        );
    }

    #[tokio::test]
    async fn response_incomplete_without_token_limit_remains_an_error() {
        for reason in [
            None,
            Some("content_filter"),
            Some("max_messages"),
            Some("steered"),
            Some("unknown"),
        ] {
            let mut response = sample_response(ResponseStatus::Incomplete);
            response.incomplete_details = reason.map(|reason| IncompleteDetailsReason {
                reason: reason.into(),
            });
            let event =
                json!({"type":"response.incomplete", "sequence_number":1, "response":response});
            let err = first_error_from_event(event).await;
            assert_eq!(
                err.to_string(),
                format!(
                    "ProviderError: OpenAI response stream was incomplete: {}",
                    reason.unwrap_or("unknown reason")
                )
            );
        }
    }

    #[tokio::test]
    async fn response_incomplete_token_limit_preserves_tool_arguments_and_usage() {
        for (arguments, expected) in [
            (r#"{"text":"ok"}"#, json!({"text":"ok"})),
            (r#"{"text":"unfinished"#, json!(r#"{"text":"unfinished"#)),
        ] {
            for item_done in [false, true] {
                let call = json!({"type":"function_call", "id":"fc_1", "call_id":"call_1",
                    "name":"echo", "arguments":arguments, "status":"incomplete"});
                let mut response =
                    serde_json::to_value(sample_response(ResponseStatus::Incomplete)).unwrap();
                response["incomplete_details"] = json!({"reason":"max_output_tokens"});
                response["output"] = json!([call]);
                response["usage"] = json!({"input_tokens":10,"output_tokens":5,"total_tokens":15,
                    "input_tokens_details":{"cached_tokens":2}, "output_tokens_details":{"reasoning_tokens":1}});
                let mut payload = String::new();
                if item_done {
                    payload.push_str(&format!(
                        "data: {}\n\n",
                        json!({"type":"response.output_item.done",
                        "output_index":0,"sequence_number":1,"item":call})
                    ));
                }
                payload.push_str(&format!(
                    "data: {}\n\n",
                    json!({"type":"response.incomplete",
                    "sequence_number":2,"response":response})
                ));
                let client = openai::Client::builder()
                    .http_client(MockStreamingClient {
                        sse_bytes: Bytes::from(payload),
                    })
                    .api_key("test-key")
                    .build()
                    .unwrap();
                let model = client.completion_model("gpt-test");
                let mut stream = model
                    .stream(model.completion_request("hello").build())
                    .await
                    .unwrap();
                let mut calls = Vec::new();
                let mut usage = None;
                while let Some(item) = stream.next().await {
                    match item.expect("a token-limit stop is an explicit terminal response") {
                        StreamedAssistantContent::ToolCall { tool_call, .. } => {
                            calls.push(tool_call)
                        }
                        StreamedAssistantContent::Final(response) => usage = response.usage,
                        _ => {}
                    }
                }
                assert_eq!(
                    calls.len(),
                    1,
                    "item_done={item_done}, arguments={arguments}"
                );
                assert_eq!(calls[0].call_id.as_deref(), Some("call_1"));
                assert_eq!(calls[0].function.name, "echo");
                assert_eq!(calls[0].function.arguments, expected);
                let usage = usage.expect("terminal usage must survive a token-limit stop");
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
                assert_eq!(usage.total_tokens, 15);
                assert_eq!(usage.input_tokens_details.unwrap().cached_tokens, Some(2));
                assert_eq!(
                    usage.output_tokens_details.unwrap().reasoning_tokens,
                    Some(1)
                );
            }
        }
    }

    #[test]
    fn partial_arguments_are_preserved_only_by_streaming_decoders() {
        let arguments = r#"{"text":"unfinished"#;
        let mut response =
            serde_json::to_value(sample_response(ResponseStatus::Incomplete)).unwrap();
        response["incomplete_details"] = json!({"reason":"max_output_tokens"});
        response["output"] = json!([{"type":"function_call", "id":"fc_1",
            "call_id":"call_1", "name":"echo", "arguments":arguments,
            "status":"incomplete"}]);
        assert!(serde_json::from_value::<CompletionResponse>(response.clone()).is_err());
        let chunk: ResponseChunk = serde_json::from_value(json!({
            "type":"response.incomplete", "sequence_number":1, "response":response
        }))
        .unwrap();
        let Output::FunctionCall(function) = &chunk.response.output[0] else {
            panic!("expected a function call");
        };
        assert_eq!(function.arguments, json!(arguments));
    }

    #[tokio::test]
    async fn missing_usage_stays_absent_on_completed_and_unfinished_streams() {
        for kind in ["response.completed", "response.created"] {
            let event = json!({
                "type": kind,
                "sequence_number": 1,
                "response": sample_response(ResponseStatus::Completed),
            });
            assert!(final_usage_from_event(event).await.is_none(), "{kind}");
        }
    }

    #[test]
    fn missing_usage_breakdowns_remain_distinct_from_zero() {
        use super::StreamingCompletionResponse;
        use crate::completion::GetTokenUsage;
        for details in [
            json!({}),
            json!({"input_tokens_details":{}, "output_tokens_details":{}}),
        ] {
            let mut raw = json!({"input_tokens":10, "output_tokens":5, "total_tokens":15});
            raw.as_object_mut()
                .unwrap()
                .extend(details.as_object().unwrap().clone());
            let response: StreamingCompletionResponse =
                serde_json::from_value(json!({"usage":raw})).unwrap();
            let usage = response.usage.as_ref().unwrap();
            assert_eq!(
                usage
                    .input_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens),
                None
            );
            assert_eq!(
                usage
                    .output_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                None
            );
            assert_eq!(response.token_usage().unwrap().total_tokens, 15);
        }
        let absent: StreamingCompletionResponse = serde_json::from_value(json!({})).unwrap();
        assert!(absent.token_usage().is_none());
    }

    #[tokio::test]
    async fn response_completed_chunk_populates_final_usage() {
        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: Some(0),
            }),
            total_tokens: 15,
        });

        let event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });

        let usage = final_usage_from_event(event).await.expect("reported usage");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    // requires `derive` rig-core feature due to using tool macro
    #[tokio::test]
    #[ignore = "requires API key"]
    async fn test_openai_streaming_tools_reasoning() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var should exist");
        let client = openai::Client::new(&api_key).expect("Failed to build client");
        let agent = client
            .agent("gpt-5.2")
            .max_tokens(8192)
            .tool(ExampleTool)
            .additional_params(serde_json::json!({
                "reasoning": {"effort": "high"}
            }))
            .build();

        let chat_history: Vec<Message> = Vec::new();
        let mut stream = agent
            .stream_chat("Call my example tool", &chat_history)
            .multi_turn(5)
            .await;

        while let Some(item) = stream.next().await {
            println!("Got item: {item:?}");
        }
    }
}
