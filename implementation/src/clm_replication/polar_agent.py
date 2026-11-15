"""Polar custom harness invoking the published CLM agent."""

import json
import shlex

from polar.agent.base import BaseHarness
from polar.runtime.models import ExecInput


class ClmHarness(BaseHarness):
    def run_steps(self, instruction):
        # A heredoc delivers the instruction without shell interpolation.
        encoded = json.dumps(instruction, ensure_ascii=False)
        command = (
            "mkdir -p /polar/session/workspace /polar/session/artifacts\n"
            "cd /polar/session/workspace\n"
            "/usr/local/bin/python3 - <<'CLM_INSTRUCTION_JSON'\n"
            "from pathlib import Path\n"
            "import json\n"
            f"text = json.loads({encoded!r})\n"
            "text += '\\n\\nWrite your final answer to /polar/session/artifacts/answer.txt before submitting.'\n"
            "text += '\\nPublic web retrieval is available through Bash. Use: /mount-data/CLM-Replication/env/agent/bin/python -m clm_replication.web search \\\"your query\\\" or fetch \\\"https://example.org/page\\\". Search returns up to ten results, fetch returns up to 32768 characters of page text.'\n"
            "Path('/polar/session/instruction.txt').write_text(text)\n"
            "CLM_INSTRUCTION_JSON\n"
            "export LITELLM_LOCAL_MODEL_COST_MAP=true\n"
            "export PYTHONPATH=/mount-data/CLM-Replication/src:/mount-data/CLM-Replication/src/context-language-models/clm\n"
            "exec /mount-data/CLM-Replication/env/agent/bin/python -m clm_replication.agent_runner "
            "--instruction /polar/session/instruction.txt --model " + shlex.quote(self.model_name)
        )
        return [ExecInput(command=command)]
