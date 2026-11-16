"""Summary harness retaining the existing task delivery and web instructions."""
from polar.runtime.models import ExecInput
from clm_replication.polar_agent import ClmHarness


class SummaryHarness(ClmHarness):
    def run_steps(self, instruction):
        original = super().run_steps(instruction)
        result = []
        source = 'export PYTHONPATH=/mount-data/CLM-Replication/src:/mount-data/CLM-Replication/src/context-language-models/clm'
        target = 'export PYTHONPATH=/mount-data/CLM-Replication/env/summary-agent-deps:/mount-data/CLM-Replication/src:/mount-data/CLM-Replication/src/context-language-models/clm'
        entry = '-m clm_replication.agent_runner '
        for step in original:
            if step.command.count(source) != 1 or step.command.count(entry) != 1:
                raise ValueError('The pinned task delivery entrypoint differs')
            command = step.command.replace(source, target).replace(entry, '-m clm_replication.summary_runner ')
            command += ' --tokenizer /mount-data/CLM-Replication/summary-assets/tokenizer'
            result.append(ExecInput(command=command))
        return result
