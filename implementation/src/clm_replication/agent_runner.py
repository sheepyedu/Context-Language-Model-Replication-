"""Execute the official CLM agent inside one isolated Polar task container."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import inspect
import time
from types import SimpleNamespace

from clm_harness.clm_agent import harness


class ContainerEnvironment:
    """Harbor's file and exec interface for the current task container."""

    def __init__(self, logs_dir):
        self.logs_dir = logs_dir
        self.command_index = 0

    async def exec(self, command, timeout_sec=120, **kwargs):
        directory = self.logs_dir / 'commands' / f'{self.command_index:04d}'
        self.command_index += 1
        directory.mkdir(parents=True)
        (directory / 'INTENT.json').write_text(json.dumps({
            'command': command, 'timeout_seconds': timeout_sec, 'started': time.time(),
        }))
        mirror = Path(harness._CTX_FILE)
        if mirror.exists():
            shutil.copyfile(mirror, directory / 'mirror.before.txt')
        process = await asyncio.create_subprocess_exec(
            "/bin/bash", "-lc", command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout_sec)
        except TimeoutError:
            import signal
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = await communication
        (directory / 'stdout.raw').write_bytes(stdout)
        (directory / 'stderr.raw').write_bytes(stderr)
        if mirror.exists():
            shutil.copyfile(mirror, directory / 'mirror.after.txt')
        (directory / 'RESULT.json').write_text(json.dumps({
            'pid': process.pid, 'return_code': process.returncode,
            'timed_out': timed_out, 'finished': time.time(),
        }))
        if timed_out:
            raise TimeoutError(f'Command exceeded {timeout_sec} seconds; partial output was archived')
        return SimpleNamespace(stdout=stdout.decode(errors="replace"),
                               stderr=stderr.decode(errors="replace"), return_code=process.returncode)

    async def upload_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    async def download_file(self, source_path, target_path):
        shutil.copyfile(source_path, target_path)


class RecordedClmAgent(harness.ClmAgent):
    """Add event evidence without changing the prompt or the context editor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events = []
        original_step = self._ctx.step

        async def recorded_step(command, messages, **step_kwargs):
            self.events[-1]['command'] = command
            self.save_events()
            try:
                outcome = await original_step(command, messages, **step_kwargs)
            except Exception as error:
                self.events[-1]['tool_error'] = type(error).__name__
                self.save_events()
                raise
            empty = not (outcome.result.stdout or "").strip() and not (outcome.result.stderr or "").strip()
            pure_edit = bool(outcome.ctx_changed) and outcome.result.return_code == 0 and empty
            self.events[-1].update({"command": command, "return_code": outcome.result.return_code,
                                   "ctx_changed": bool(outcome.ctx_changed), "pure_edit": pure_edit,
                                   "submitted": self._is_successful_submit(
                                       command, outcome.result, outcome.render())})
            self.save_events()
            return outcome

        self._ctx.step = recorded_step

    def save_events(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.logs_dir / "edit_events.tmp"
        temporary.write_text(json.dumps(self.events, ensure_ascii=False, indent=2))
        temporary.replace(self.logs_dir / "edit_events.json")

    async def _query_with_retry(self, model, messages):
        # The gateway keeps the authoritative raw request and response. Unknown
        # completions terminate this session instead of resubmitting the call.
        response = await super()._query_with_retry(model, messages)
        tools = response.choices[0].message.model_dump().get('tool_calls') or []
        self.events.append({"ordinal": len(self.events), "response_id": response.id,
                            "pure_edit": False, "command": None,
                            "raw_arguments": tools[0]['function'].get('arguments') if tools else None})
        self.save_events()
        return response


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    harness._MAX_RETRIES = 1
    log_dir = Path(os.environ["AGENT_LOG_DIR"])
    configuration = dict(
        logs_dir=log_dir, model_name="openai/" + args.model,
        api_base=os.environ["OPENAI_BASE_URL"], max_steps=80, max_tokens=4096,
        temperature=0.7, top_p=0.95, enable_thinking=True,
        context_budget_tokens=28672, context_budget_reserve_tokens=2048,
        flops_model_key="9b", cost_metric="auto",
    )
    agent = RecordedClmAgent(**configuration)
    defaults = {name: item.default for name, item in inspect.signature(harness.ClmAgent.__init__).parameters.items()
                if item.default is not inspect.Parameter.empty}
    defaults.update(configuration)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'AGENT_EFFECTIVE_CONFIG.json').write_text(json.dumps({
        'configuration': defaults, 'lm_call_cap': agent.lm_call_cap,
        'strict_target': agent._budget.strict_target, 'allow_edit_growth': agent._ctx.allow_edit_growth,
    }, indent=2, default=str))
    environment = ContainerEnvironment(log_dir)
    await agent.setup(environment)
    await agent.run(Path(args.instruction).read_text(), environment, SimpleNamespace())


if __name__ == "__main__":
    asyncio.run(main())
