"""Core orchestrator for agent evaluation."""

import logging
from typing import List

from tqdm import tqdm

from .eval_data import AgentGoalEvalDataManager
from .evaluator import EvaluationRunner
from ..utils.api_client import AgentHttpClient
from ..utils.judge import JudgeModelManager
from .models import EvaluationResult
from .results import ResultsManager

logger = logging.getLogger(__name__)


class AgentGoalEval:
    """Orchestrator for agent goal evaluation."""

    def __init__(self, eval_args):
        """Initialize agent goal evaluation."""
        self.eval_args = eval_args
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all evaluation components."""
        # Configuration manager
        self.config_manager = AgentGoalEvalDataManager(self.eval_args.eval_data_yaml)

        # Agent HTTP client
        self.agent_client = AgentHttpClient(
            self.eval_args.agent_endpoint, self.eval_args.agent_auth_token_file
        )

        # Judge model manager (optional)
        self.judge_manager = None
        if self.eval_args.judge_provider and self.eval_args.judge_model:
            self.judge_manager = JudgeModelManager(
                self.eval_args.judge_provider, self.eval_args.judge_model
            )

        # Evaluation runner
        self.evaluation_runner = EvaluationRunner(
            self.agent_client,
            self.judge_manager,
            kubeconfig=getattr(self.eval_args, "kubeconfig", None),
        )

        # Results manager
        self.results_manager = ResultsManager(self.eval_args.result_dir)

    def get_eval_result(self) -> None:
        """Run all evaluations and save results."""
        try:
            configs = self.config_manager.get_eval_data()
            logger.info(f"Running {len(configs)} evaluations")

            results = []
            for config in tqdm(configs, desc="Running evaluations"):
                result = self.evaluation_runner.run_evaluation(
                    config, self.eval_args.agent_provider, self.eval_args.agent_model
                )
                results.append(result)

                # Log individual result
                logger.info(f"Evaluation {config.eval_id}: {result.result}")
                if result.error:
                    logger.error(f"Error in {config.eval_id}: {result.error}")

            # Save results
            self.results_manager.save_results(results)

            # Print summary
            self._print_summary(results)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            # Clean up resources
            self._cleanup()

    def _print_summary(self, results: List[EvaluationResult]) -> None:
        """Print evaluation summary."""
        total = len(results)
        passed = sum(1 for r in results if r.result == "PASS")
        failed = sum(1 for r in results if r.result == "FAIL")
        errored = sum(1 for r in results if r.result == "FAILED")

        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Evaluations: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errored: {errored}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print(f"{'='*50}\n")

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "agent_client"):
                self.agent_client.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
