"""Custom exceptions for agent evaluation."""


class AgentEvaluationError(Exception):
    """Base exception for agent evaluation errors."""

    pass


class ConfigurationError(AgentEvaluationError):
    """Configuration-related errors."""

    pass


class AgentAPIError(AgentEvaluationError):
    """Agent API communication errors."""

    pass


class ScriptExecutionError(AgentEvaluationError):
    """Script execution errors."""

    pass


class JudgeModelError(AgentEvaluationError):
    """Judge model errors."""

    pass
