"""HTTP client for agent API communication."""

import httpx
from typing import Optional
import logging

from .exceptions import AgentAPIError

logger = logging.getLogger(__name__)


class AgentHttpClient:
    """HTTP client for agent API communication."""

    def __init__(self, endpoint: str, token_file: Optional[str] = None):
        """Initialize HTTP client."""
        self.endpoint = endpoint
        self._setup_client(token_file)

    def _setup_client(self, token_file: Optional[str]) -> None:
        """Initialize HTTP client with authentication."""
        try:
            self.client = httpx.Client(base_url=self.endpoint, verify=False)

            if token_file:
                token = self._read_token_file(token_file)
                self.client.headers.update({"Authorization": f"Bearer {token}"})

        except Exception as e:
            raise AgentAPIError(f"Failed to setup HTTP client: {e}")

    def _read_token_file(self, token_file: str) -> str:
        """Read authentication token from file."""
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise AgentAPIError(f"Token file not found: {token_file}")
        except Exception as e:
            raise AgentAPIError(f"Error reading token file: {e}")

    def query_agent(
        self, query: str, provider: str, model: str, timeout: int = 300
    ) -> str:
        """Query the agent and return response."""
        try:
            response = self.client.post(
                "/v1/query",
                json={
                    "query": query,
                    "provider": provider,
                    "model": model,
                },
                timeout=timeout,
            )
            response.raise_for_status()

            response_data = response.json()
            if "response" not in response_data:
                raise AgentAPIError("Agent response missing 'response' field")

            logger.info(f"Agent response >\n{response_data['response'].strip()}")
            return response_data["response"].strip()

        except httpx.TimeoutException:
            raise AgentAPIError(f"Agent query timeout after {timeout} seconds")
        except httpx.HTTPStatusError as e:
            raise AgentAPIError(
                f"Agent API error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise AgentAPIError(f"Unexpected error querying agent: {e}")

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
