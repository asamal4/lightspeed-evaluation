"""Results management for agent evaluation."""

import logging
from pathlib import Path
from typing import List

from pandas import DataFrame

from .models import EvaluationResult

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages evaluation results and output."""

    def __init__(self, output_dir: str):
        """Initialize results manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        results: List[EvaluationResult],
        filename: str = "agent_goal_eval_results.csv",
    ) -> None:
        """Save evaluation results to CSV file."""
        if not results:
            logger.warning("No results to save")
            return

        try:
            # Convert results to DataFrame
            data = []
            for result in results:
                data.append(
                    {
                        "eval_id": result.eval_id,
                        "query": result.query,
                        "response": result.response,
                        "eval_type": result.eval_type,
                        "result": result.result,
                        "error": result.error or "",
                    }
                )

            df = DataFrame(data)

            # Save to CSV
            output_path = self.output_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
