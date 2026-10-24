use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    completion::GetTokenUsage,
    http_client,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ================================================================
// Main openrouter Client
// ================================================================
const OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExtBuilder;

type OpenRouterApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<OpenRouterExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<OpenRouterExtBuilder, OpenRouterApiKey, H>;

impl Provider for OpenRouterExt {
    type Builder = OpenRouterExtBuilder;

    const VERIFY_PATH: &'static str = "/key";
}

impl<H> Capabilities<H> for OpenRouterExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for OpenRouterExt {}

impl ProviderBuilder for OpenRouterExtBuilder {
    type Extension<H>
        = OpenRouterExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = OpenRouterApiKey;

    const BASE_URL: &'static str = OPENROUTER_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(OpenRouterExt)
    }
}

impl ProviderClient for Client {
    type Input = OpenRouterApiKey;

    /// Create a new openrouter client from the `OPENROUTER_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");

        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    /// Provider-reported completion tokens, including reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    /// Provider-reported prompt cache usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<crate::providers::openai::completion::PromptTokensDetails>,
    /// Provider-reported completion subdivisions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details:
        Option<crate::providers::openai::completion::CompletionTokensDetails>,
    pub total_tokens: usize,
    #[serde(default)]
    pub cost: f64,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = self
            .completion_tokens
            .unwrap_or_else(|| self.total_tokens.saturating_sub(self.prompt_tokens))
            as u64;
        usage.cached_input_tokens = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .unwrap_or_default() as u64;
        usage.total_tokens = self.total_tokens as u64;

        Some(usage)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::openrouter::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}

#[cfg(test)]
mod usage_observation_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_retains_cache_and_reasoning_details() {
        let usage: Usage = serde_json::from_value(json!({
            "prompt_tokens":100, "completion_tokens":20, "total_tokens":120,
            "prompt_tokens_details":{"cached_tokens":80},
            "completion_tokens_details":{"reasoning_tokens":7}
        }))
        .unwrap();
        let wire = serde_json::to_value(&usage).unwrap();
        assert_eq!(wire["completion_tokens_details"]["reasoning_tokens"], 7);
        assert_eq!(usage.token_usage().unwrap().cached_input_tokens, 80);
    }

    #[test]
    fn missing_output_and_stream_usage_remain_unreported() {
        let usage: Usage =
            serde_json::from_value(json!({"prompt_tokens":100,"total_tokens":100})).unwrap();
        assert!(
            serde_json::to_value(&usage)
                .unwrap()
                .get("completion_tokens")
                .is_none()
        );
        let response: crate::providers::openrouter::streaming::StreamingCompletionResponse =
            serde_json::from_value(json!({"usage":null})).unwrap();
        assert!(response.token_usage().is_none());
    }
}
