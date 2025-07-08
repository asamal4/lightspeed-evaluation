"""Script execution handler for agent evaluation."""

import subprocess
import os
from pathlib import Path
from typing import Optional

from ..utils.exceptions import ScriptExecutionError


class ScriptRunner:
    """Handles script execution with proper error handling."""

    @staticmethod
    def run_script(
        script_file: str,
        check_return_code: bool = True,
        kubeconfig_file: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run a script and return the result.

        Args:
            script_path: Path to the script to execute
            check_return_code: Whether to raise exception on non-zero return code
            kubeconfig: Optional path to kubeconfig file for kubernetes operations

        Returns:
            subprocess.CompletedProcess: Result of script execution
        """
        script_path: Path = Path(script_file)

        if not script_path.exists():
            raise ScriptExecutionError(f"Script not found: {script_path}")

        try:
            # Make script executable
            script_path.chmod(0o755)

            # Prepare environment variables
            env = os.environ.copy()
            if kubeconfig_file:
                kubeconfig_path: Path = Path(kubeconfig_file)
                if kubeconfig_path.exists():
                    env["KUBECONFIG"] = str(kubeconfig_path.absolute())
                else:
                    raise ScriptExecutionError(
                        f"Kubeconfig file not found: {kubeconfig_path}"
                    )

            result = subprocess.run(
                [str(script_path)], capture_output=True, text=True, check=False, env=env
            )

            if check_return_code and result.returncode != 0:
                raise ScriptExecutionError(
                    f"Script failed with return code {result.returncode}: {result.stderr}"
                )

            return result

        except subprocess.SubprocessError as e:
            raise ScriptExecutionError(f"Error running script {script_path}: {e}")
        except Exception as e:
            raise ScriptExecutionError(
                f"Unexpected error running script {script_path}: {e}"
            )
