"""Tests for script runner."""

import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lsc_agent_eval.core.agent_goal_eval.script_runner import ScriptRunner
from lsc_agent_eval.core.utils.exceptions import ScriptExecutionError


class TestScriptRunner:
    """Test ScriptRunner."""

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_success(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test successful script execution."""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        # Run test using instance method
        runner = ScriptRunner()
        result = runner.run_script("test_script.sh")

        # Assertions
        assert result  # Instance method returns boolean
        mock_exists.assert_called_once()
        mock_chmod.assert_called_once_with(0o755)
        mock_subprocess_run.assert_called_once_with(
            [str(Path("test_script.sh").resolve())],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=300,
            check=False,
        )

    @patch("pathlib.Path.exists")
    def test_run_script_file_not_found(self, mock_exists):
        """Test script execution with non-existent file."""
        mock_exists.return_value = False

        runner = ScriptRunner()
        with pytest.raises(ScriptExecutionError, match="Script not found"):
            runner.run_script("missing_script.sh")

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_with_kubeconfig(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution with kubeconfig."""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        # Test with kubeconfig set during initialization
        runner = ScriptRunner(kubeconfig="./kubeconfig")
        result = runner.run_script("test_script.sh")

        assert result
        # Verify environment includes KUBECONFIG
        expected_env = os.environ.copy()
        expected_env["KUBECONFIG"] = "./kubeconfig"
        mock_subprocess_run.assert_called_once_with(
            [str(Path("test_script.sh").resolve())],
            text=True,
            capture_output=True,
            env=expected_env,
            timeout=300,
            check=False,
        )

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_failure(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution failure."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Script failed"
        mock_subprocess_run.return_value = mock_result

        runner = ScriptRunner()
        result = runner.run_script("test_script.sh")

        # Instance method returns False for non-zero return codes
        assert not result

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_subprocess_error(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution with subprocess error."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_subprocess_run.side_effect = subprocess.SubprocessError(
            "Subprocess failed"
        )

        runner = ScriptRunner()
        with pytest.raises(ScriptExecutionError, match="Error running script"):
            runner.run_script("test_script.sh")

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_unexpected_error(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution with unexpected error."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_subprocess_run.side_effect = Exception("Unexpected error")

        runner = ScriptRunner()
        with pytest.raises(
            ScriptExecutionError, match="Unexpected error running script"
        ):
            runner.run_script("test_script.sh")

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_chmod_error(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution with chmod error."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_chmod.side_effect = OSError("Permission denied")

        runner = ScriptRunner()
        with pytest.raises(
            ScriptExecutionError, match="Unexpected error running script"
        ):
            runner.run_script("test_script.sh")

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_absolute_path(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution with absolute path."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        absolute_path = "/test/test_script.sh"
        runner = ScriptRunner()
        result = runner.run_script(absolute_path)

        assert result
        mock_subprocess_run.assert_called_once_with(
            [absolute_path],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=300,
            check=False,
        )

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_relative_path(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution with relative path."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        relative_path = "./scripts/test.sh"
        runner = ScriptRunner()
        result = runner.run_script(relative_path)

        assert result

        expected_path = str(Path("scripts/test.sh").resolve())
        mock_subprocess_run.assert_called_once_with(
            [expected_path],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=300,
            check=False,
        )

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_environment_preservation(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test that original environment is preserved."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        # Set test environment variable
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            runner = ScriptRunner()
            runner.run_script("test_script.sh")

            # Verify environment includes test variable
            expected_env = os.environ.copy()
            mock_subprocess_run.assert_called_once_with(
                [str(Path("test_script.sh").resolve())],
                text=True,
                capture_output=True,
                env=expected_env,
                timeout=300,
                check=False,
            )

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_kubeconfig_absolute_path(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test kubeconfig with absolute path."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        kubeconfig_path = "/home/user/.kube/config"
        runner = ScriptRunner(kubeconfig=kubeconfig_path)
        runner.run_script("test_script.sh")

        expected_env = os.environ.copy()
        expected_env["KUBECONFIG"] = kubeconfig_path
        mock_subprocess_run.assert_called_once_with(
            [str(Path("test_script.sh").resolve())],
            text=True,
            capture_output=True,
            env=expected_env,
            timeout=300,
            check=False,
        )

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_no_kubeconfig(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test script execution without kubeconfig."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        runner = ScriptRunner()  # No kubeconfig
        result = runner.run_script("test_script.sh")

        assert result
        mock_subprocess_run.assert_called_once_with(
            [str(Path("test_script.sh").resolve())],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=300,
            check=False,
        )

    @patch("subprocess.run")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.chmod")
    def test_run_script_capture_output(
        self, mock_chmod, mock_exists, mock_is_file, mock_subprocess_run
    ):
        """Test that script execution captures output."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Script output"
        mock_result.stderr = "Script error"
        mock_subprocess_run.return_value = mock_result

        runner = ScriptRunner()
        result = runner.run_script("test_script.sh")

        assert result
        # Note: Instance method returns boolean, not the result object

    def test_script_runner_init(self):
        """Test ScriptRunner initialization."""
        runner = ScriptRunner()
        assert runner.kubeconfig is None

        runner_with_config = ScriptRunner(kubeconfig="~/kubeconfig")
        assert runner_with_config.kubeconfig == "~/kubeconfig"

    def test_get_environment_without_kubeconfig(self):
        """Test get_environment without kubeconfig."""
        runner = ScriptRunner()
        env = runner.get_environment()

        # Should return copy of os.environ
        assert env == os.environ.copy()
        assert env is not os.environ  # Should be a copy

    def test_get_environment_with_kubeconfig(self):
        """Test get_environment with kubeconfig."""
        runner = ScriptRunner(kubeconfig="~/kubeconfig")
        env = runner.get_environment()

        expected_env = os.environ.copy()
        expected_env["KUBECONFIG"] = "~/kubeconfig"
        assert env == expected_env
