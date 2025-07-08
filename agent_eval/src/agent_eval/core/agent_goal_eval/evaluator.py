"""Evaluation runner that orchestrates different evaluation types."""

import re
import logging
from typing import Optional

from ..utils.exceptions import ScriptExecutionError
from ..utils.api_client import AgentHttpClient
from ..utils.judge import JudgeModelManager
from ..utils.prompt import ANSWER_CORRECTNESS_PROMPT
from .models import EvaluationDataConfig, EvaluationResult
from .script_runner import ScriptRunner

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates different types of evaluations."""

    def __init__(
        self,
        agent_client: AgentHttpClient,
        judge_manager: Optional[JudgeModelManager] = None,
        kubeconfig: Optional[str] = None,
    ):
        """Initialize evaluation runner."""
        self.agent_client = agent_client
        self.judge_manager = judge_manager
        self.kubeconfig = kubeconfig

    def run_evaluation(
        self, data_config: EvaluationDataConfig, agent_provider: str, agent_model: str
    ) -> EvaluationResult:
        """Run a single evaluation based on configuration."""
        try:
            # Execute setup script if provided
            if data_config.eval_setup_script:
                try:
                    ScriptRunner.run_script(
                        data_config.eval_setup_script, kubeconfig_file=self.kubeconfig
                    )
                    logger.info(
                        f"Setup script executed successfully for {data_config.eval_id}"
                    )
                except ScriptExecutionError as e:
                    logger.error(f"Setup script failed for {data_config.eval_id}: {e}")
                    return EvaluationResult(
                        eval_id=data_config.eval_id,
                        query=data_config.eval_query,
                        response="",
                        eval_type=data_config.eval_type,
                        result="FAILED",
                        error=f"Setup script failed: {e}",
                    )

            response = self.agent_client.query_agent(
                data_config.eval_query, agent_provider, agent_model
            )

            # Evaluate response based on type
            success = self._evaluate_response(data_config, response)

            # Execute cleanup script if provided
            if data_config.eval_cleanup_script:
                try:
                    ScriptRunner.run_script(
                        data_config.eval_cleanup_script, kubeconfig_file=self.kubeconfig
                    )
                    logger.info(
                        f"Cleanup script executed successfully for {data_config.eval_id}"
                    )
                except ScriptExecutionError as e:
                    logger.warning(
                        f"Cleanup script failed for {data_config.eval_id}: {e}"
                    )

            return EvaluationResult(
                eval_id=data_config.eval_id,
                query=data_config.eval_query,
                response=response,
                eval_type=data_config.eval_type,
                result="PASS" if success else "FAIL",
            )

        except Exception as e:
            logger.error(f"Evaluation failed for {data_config.eval_id}: {e}")
            return EvaluationResult(
                eval_id=data_config.eval_id,
                query=data_config.eval_query,
                response="",
                eval_type=data_config.eval_type,
                result="FAILED",
                error=str(e),
            )

    def _evaluate_response(
        self, data_config: EvaluationDataConfig, response: str
    ) -> bool:
        """Evaluate response based on configuration type."""
        if data_config.eval_type == "script":
            return self._evaluate_script(data_config)
        elif data_config.eval_type == "sub-string":
            return self._evaluate_substring(data_config, response)
        elif data_config.eval_type == "judge-llm":
            return self._evaluate_judge_llm(data_config, response)
        else:
            logger.error(f"Unknown evaluation type: {data_config.eval_type}")
            return False

    def _evaluate_script(self, data_config: EvaluationDataConfig) -> bool:
        """Evaluate using script execution."""
        try:
            result = ScriptRunner.run_script(
                data_config.eval_verify_script,
                check_return_code=False,
                kubeconfig_file=self.kubeconfig,
            )
            return result.returncode == 0
        except ScriptExecutionError as e:
            logger.error(f"Script evaluation failed: {e}")
            return False

    def _evaluate_substring(
        self, data_config: EvaluationDataConfig, response: str
    ) -> bool:
        """Evaluate using substring matching."""
        if not data_config.expected_key_words:
            return False

        response_lower = response.lower()
        for keyword in data_config.expected_key_words:
            if keyword.lower() in response_lower:
                return True
        return False

    def _extract_numeric_result(self, response: str) -> int:
        """Extract numeric result from judge response."""
        # Look for 1 or 0 in the response
        response = response.strip()

        # Direct match
        if response == "1":
            return 1
        elif response == "0":
            return 0

        # Look for digits in the response
        numbers = re.findall(r"\b[01]\b", response)
        if numbers:
            return int(numbers[0])

        # If no clear numeric result, default to 0 (fail)
        logger.warning(
            f"Could not extract numeric result from judge response: {response}"
        )
        return 0

    def _evaluate_judge_llm(
        self, data_config: EvaluationDataConfig, response: str
    ) -> bool:
        """Evaluate using judge LLM."""
        if not self.judge_manager:
            logger.error("Judge model manager not available for judge-llm evaluation")
            return False

        if not data_config.expected_response:
            logger.error("Expected response not provided for judge-llm evaluation")
            return False

        # Format prompt with parameters
        prompt = ANSWER_CORRECTNESS_PROMPT.format(
            question=data_config.eval_query,
            answer=data_config.expected_response,
            response=response,
        )
        try:
            judge_resp = self.judge_manager.evaluate_response(prompt)
            if judge_resp is None:
                return False
            # Extract numeric result (looking for 1 or 0)
            result = self._extract_numeric_result(judge_resp)
            return result == 1
        except Exception as e:
            logger.error(f"Judge LLM evaluation failed: {e}")
            return False
