"""Judge model management using LiteLLM."""

import logging
import os
from time import sleep
from typing import Optional

import litellm

from .constants import MAX_RETRY_ATTEMPTS, TIME_TO_BREATH
from .exceptions import JudgeModelError

logger = logging.getLogger(__name__)


# TODO: Move to llama-stack
class JudgeModelManager:
    """Manages judge model loading and evaluation using llm."""

    def __init__(self, judge_provider: str, judge_model: str):
        """Initialize judge model manager."""
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self._setup_litellm()

    def _setup_litellm(self) -> None:
        """Initialize LiteLLM with provider-specific configuration."""
        try:
            logger.info(
                f"Setting up LiteLLM for {self.judge_provider}/{self.judge_model}"
            )

            provider = self.judge_provider.lower()

            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise JudgeModelError(
                        "OPENAI_API_KEY environment variable is required for OpenAI provider"
                    )
                self.model_name = self.judge_model

            elif provider == "azure":
                # Azure OpenAI - LiteLLM format: azure/{deployment_name}
                # Keep the deployment_name same as model name for consistency
                api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
                deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

                if not all([api_key, api_base]):
                    raise JudgeModelError(
                        "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required for Azure provider"
                    )

                # Use deployment name if provided, otherwise use model name
                deployment = deployment_name or self.judge_model
                self.model_name = f"azure/{deployment}"

            elif provider == "watsonx":
                # Watsonx - LiteLLM format: watsonx/{model}
                api_key = os.environ.get("WATSONX_API_KEY")
                api_base = os.environ.get("WATSONX_API_BASE")
                project_id = os.environ.get("WATSONX_PROJECT_ID")

                if not all([api_key, api_base, project_id]):
                    raise JudgeModelError(
                        "WATSONX_API_KEY, WATSONX_API_BASE, and WATSONX_PROJECT_ID environment variables are required for Watsonx provider"
                    )

                self.model_name = f"watsonx/{self.judge_model}"

            else:
                # Generic provider - try as-is
                logger.warning(f"Using generic provider format for {provider}")
                self.model_name = f"{provider}/{self.judge_model}"

            # Set LiteLLM configuration
            litellm.set_verbose = False  # Disable verbose logging

        except Exception as e:
            raise JudgeModelError(f"Failed to setup JudgeLLM using LiteLLM: {e}")

    def evaluate_response(self, prompt: str) -> Optional[str]:
        """Evaluate response using judge model via LiteLLM."""
        judge_response = None
        for retry_counter in range(MAX_RETRY_ATTEMPTS):
            try:
                # Use LiteLLM completion
                response = litellm.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                    timeout=300,
                )

                # Extract content from response
                judge_response = response.choices[0].message.content.strip()
                break

            except Exception as e:
                if retry_counter == MAX_RETRY_ATTEMPTS - 1:
                    raise JudgeModelError(
                        f"Judge model evaluation failed after {MAX_RETRY_ATTEMPTS} attempts: {e}"
                    )

                logger.warning(f"Judge model attempt {retry_counter + 1} failed: {e}")
                sleep(TIME_TO_BREATH)

        return judge_response
