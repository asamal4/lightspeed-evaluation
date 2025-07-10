"""Command line interface for agent evaluation."""

import argparse
import logging
import sys

from .core.utils.utils import add_common_arguments
from .core.utils.constants import DEFAULT_RESULT_DIR

from .core.agent_goal_eval.agent_goal_eval import AgentGoalEval

logger = logging.getLogger(__name__)


def _args_parser(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate agent goal responses using various evaluation types"
    )

    # Add common arguments
    add_common_arguments(parser)

    # Agent evaluation specific arguments
    parser.add_argument(
        "--eval_data_yaml",
        type=str,
        required=True,
        help="Path to YAML eval data file with evaluation scenarios",
    )

    parser.add_argument(
        "--agent_endpoint",
        type=str,
        default="http://localhost:8080",
        help="Agent API endpoint URL",
    )

    parser.add_argument(
        "--agent_provider", type=str, required=True, help="Agent provider name"
    )

    parser.add_argument(
        "--agent_model", type=str, required=True, help="Agent model name"
    )

    parser.add_argument(
        "--agent_auth_token_file",
        type=str,
        help="Path to .txt file containing agent authentication token",
    )

    parser.add_argument(
        "--judge_provider",
        type=str,
        help="Judge model provider (e.g., openai, azure, watsonx)",
    )

    parser.add_argument(
        "--judge_model", type=str, help="Judge model name for LLM-based evaluations"
    )

    parser.add_argument("--kubeconfig", type=str, help="Path to the kubeconfig file")

    parser.add_argument(
        "--result_dir",
        type=str,
        default=DEFAULT_RESULT_DIR,
        help="Directory to save evaluation results",
    )

    return parser.parse_args(args)


def main() -> None:
    """Entry point for agent evaluation."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Parse arguments
        args = _args_parser(sys.argv[1:])

        # Create and run evaluation
        evaluator = AgentGoalEval(args)
        evaluator.get_eval_result()

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
