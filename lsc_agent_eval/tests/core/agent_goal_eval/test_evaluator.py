"""Tests for evaluation runner."""

import os
import tempfile
from unittest.mock import Mock

import pytest

from lsc_agent_eval.core.agent_goal_eval.evaluator import EvaluationRunner
from lsc_agent_eval.core.agent_goal_eval.models import (
    EvaluationDataConfig,
    EvaluationResult,
)
from lsc_agent_eval.core.agent_goal_eval.script_runner import ScriptRunner
from lsc_agent_eval.core.utils.api_client import AgentHttpClient
from lsc_agent_eval.core.utils.exceptions import AgentAPIError, ScriptExecutionError
from lsc_agent_eval.core.utils.judge import JudgeModelManager


class TestEvaluationRunner:
    """Test EvaluationRunner."""

    @pytest.fixture
    def mock_agent_client(self):
        """Mock agent client."""
        mock_client = Mock(spec=AgentHttpClient)
        mock_client.query_agent.return_value = "Test agent response"
        return mock_client

    @pytest.fixture
    def mock_script_runner(self):
        """Mock script runner."""
        mock_runner = Mock(spec=ScriptRunner)
        mock_runner.run_script.return_value = True
        return mock_runner

    @pytest.fixture
    def mock_judge_manager(self):
        """Mock judge manager."""
        mock_judge = Mock(spec=JudgeModelManager)
        mock_judge.evaluate_response.return_value = "1"
        return mock_judge

    @pytest.fixture
    def sample_config_judge_llm(self):
        """Sample judge-llm evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_001",
            eval_query="What is Openshift Virtualization?",
            eval_type="judge-llm",
            expected_response="OpenShift Virtualization is an extension of the OpenShift Container Platform",
        )

    @pytest.fixture
    def get_test_script_path(self):
        """Create a temporary test script file and cleanup."""
        # Setup
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write('#!/bin/bash\necho "test script"\nexit 0')
            script_path = f.name
        os.chmod(script_path, 0o755)

        yield script_path

        # Cleanup
        os.unlink(script_path)

    @pytest.fixture
    def sample_config_script(self, get_test_script_path):
        """Sample script evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_002",
            eval_query="Deploy nginx",
            eval_type="script",
            eval_verify_script=get_test_script_path,
        )

    @pytest.fixture
    def sample_config_substring(self):
        """Sample sub-string evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_003",
            eval_query="What is Podman?",
            eval_type="sub-string",
            expected_keywords=["container", "podman"],
        )

    def test_init(self, mock_agent_client, mock_script_runner, mock_judge_manager):
        """Test EvaluationRunner initialization."""
        runner = EvaluationRunner(
            mock_agent_client,
            mock_script_runner,
            mock_judge_manager,
        )

        assert runner.agent_client == mock_agent_client
        assert runner.script_runner == mock_script_runner
        assert runner.judge_manager == mock_judge_manager

    def test_init_without_judge_manager(self, mock_agent_client, mock_script_runner):
        """Test EvaluationRunner initialization without judge manager."""
        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        assert runner.agent_client == mock_agent_client
        assert runner.script_runner == mock_script_runner
        assert runner.judge_manager is None

    def test_run_evaluation_judge_llm_success(
        self,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test successful judge-llm evaluation."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        result = runner.run_evaluation(
            sample_config_judge_llm,
            "watsonx",
            "ibm/granite-3-3-8b-instruct",
            "conversation-uuid-123",
        )

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_001"
        assert result.query == "What is Openshift Virtualization?"
        assert result.eval_type == "judge-llm"
        assert result.result == "PASS"
        assert result.conversation_uuid == "conversation-uuid-123"
        assert result.error is None

        # Verify agent was called
        mock_agent_client.query_agent.assert_called_once_with(
            {
                "query": "What is Openshift Virtualization?",
                "provider": "watsonx",
                "model": "ibm/granite-3-3-8b-instruct",
            },
            "conversation-uuid-123",
        )

        # Verify judge was called
        mock_judge_manager.evaluate_response.assert_called_once()

    def test_run_evaluation_judge_llm_failure(
        self,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test failed judge-llm evaluation."""
        # Mock judge to return 0 (failure)
        mock_judge_manager.evaluate_response.return_value = "0"

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        result = runner.run_evaluation(
            sample_config_judge_llm,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert result.result == "FAIL"
        assert result.error is None

    def test_run_evaluation_script_success(
        self, mock_agent_client, mock_script_runner, sample_config_script
    ):
        """Test successful script evaluation."""
        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_script,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_002"
        assert result.eval_type == "script"
        assert result.result == "PASS"
        assert result.error is None

        # Verify agent was called
        mock_agent_client.query_agent.assert_called_once()

        # Verify script was run
        mock_script_runner.run_script.assert_called_once_with(
            sample_config_script.eval_verify_script
        )

    def test_run_evaluation_script_failure(
        self, mock_agent_client, mock_script_runner, sample_config_script
    ):
        """Test failed script evaluation."""
        # Mock script to return False (failure)
        mock_script_runner.run_script.return_value = False

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_script,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert result.result == "FAIL"
        assert result.error is None

    def test_run_evaluation_script_execution_error(
        self, mock_agent_client, mock_script_runner, sample_config_script
    ):
        """Test script evaluation with execution error."""
        # Mock script to raise exception
        mock_script_runner.run_script.side_effect = ScriptExecutionError(
            "Script failed"
        )

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_script,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert result.result == "ERROR"
        assert "Script failed" in result.error

    def test_run_evaluation_substring_success(
        self, mock_agent_client, mock_script_runner, sample_config_substring
    ):
        """Test successful sub-string evaluation."""
        # Mock agent response containing expected keywords
        mock_agent_client.query_agent.return_value = (
            "Podman is an open-source container engine developed by Red Hat"
        )

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_substring,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert result.eval_id == "test_003"
        assert result.result == "PASS"
        assert result.eval_type == "sub-string"
        assert result.error is None

    def test_run_evaluation_substring_failure(
        self, mock_agent_client, mock_script_runner, sample_config_substring
    ):
        """Test sub-string evaluation failure."""
        # Mock agent response not containing expected keywords
        mock_agent_client.query_agent.return_value = "No information available"

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_substring,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert result.eval_id == "test_003"
        assert result.result == "FAIL"
        assert result.eval_type == "sub-string"
        assert result.error is None

    def test_run_evaluation_agent_api_error(
        self, mock_agent_client, mock_script_runner, sample_config_judge_llm
    ):
        """Test evaluation with agent API error."""
        # Mock agent client to raise API error
        mock_agent_client.query_agent.side_effect = AgentAPIError(
            "API connection failed"
        )

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_judge_llm,
            "openai",
            "gpt-4",
            "conversation-uuid-123",
        )

        assert result.eval_id == "test_001"
        assert result.result == "ERROR"
        assert result.eval_type == "judge-llm"
        assert "API connection failed" in result.error

    def test_run_evaluation_no_verify_script_for_script_type(
        self, mock_agent_client, mock_script_runner
    ):
        """Test script evaluation without verify script should not occur due to Pydantic validation."""
        # This test verifies that Pydantic prevents invalid configurations
        with pytest.raises(Exception):  # Pydantic ValidationError
            EvaluationDataConfig(
                eval_id="test_invalid",
                eval_query="Test query",
                eval_type="script",
                # Missing eval_verify_script
            )

    def test_substring_evaluation_logic(self, mock_agent_client, mock_script_runner):
        """Test sub-string evaluation logic with various scenarios."""
        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        config = EvaluationDataConfig(
            eval_id="substring_test",
            eval_query="Test query",
            eval_type="sub-string",
            expected_keywords=["keyword1", "keyword2"],
        )

        # Test all keywords present - should PASS
        mock_agent_client.query_agent.return_value = (
            "Response with keyword1 and keyword2"
        )
        result = runner.run_evaluation(config, "openai", "gpt-4", "uuid-123")
        assert result.result == "PASS"

        # Test some keywords missing (only one present) - should FAIL
        mock_agent_client.query_agent.return_value = "Response with only keyword1"
        result = runner.run_evaluation(config, "openai", "gpt-4", "uuid-123")
        assert result.result == "FAIL"

        # Test no keywords present - should FAIL
        mock_agent_client.query_agent.return_value = "Response with no matching terms"
        result = runner.run_evaluation(config, "openai", "gpt-4", "uuid-123")
        assert result.result == "FAIL"

        # Test case insensitive matching
        mock_agent_client.query_agent.return_value = (
            "Response with KEYWORD1 and Keyword2"
        )
        result = runner.run_evaluation(config, "openai", "gpt-4", "uuid-123")
        assert result.result == "PASS"

    def test_conversation_uuid_propagation(
        self, mock_agent_client, mock_script_runner, mock_judge_manager
    ):
        """Test that conversation UUID is properly propagated to results."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        config = EvaluationDataConfig(
            eval_id="uuid_test",
            eval_query="Test query",
            eval_type="judge-llm",
            expected_response="Test response",
        )

        test_uuid = "test-conversation-uuid-456"
        result = runner.run_evaluation(config, "openai", "gpt-4", test_uuid)

        assert result.conversation_uuid == test_uuid

        # Verify UUID was passed to agent client
        mock_agent_client.query_agent.assert_called_once_with(
            {
                "query": "Test query",
                "provider": "openai",
                "model": "gpt-4",
            },
            test_uuid,
        )
