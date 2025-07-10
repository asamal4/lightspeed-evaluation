"""Data models for agent evaluation."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvaluationResult:
    """Evaluation result data structure."""

    eval_id: str
    query: str
    response: str
    eval_type: str
    result: str
    error: Optional[str] = None


@dataclass
class EvaluationDataConfig:
    """Single evaluation data configuration."""

    eval_id: str
    eval_query: str
    eval_type: str = "judge-llm"
    expected_response: Optional[str] = None
    expected_key_words: Optional[List[str]] = None
    eval_setup_script: Optional[str] = None
    eval_verify_script: Optional[str] = None
    eval_cleanup_script: Optional[str] = None
