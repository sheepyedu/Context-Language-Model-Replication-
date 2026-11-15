"""Polar evaluator preserving task outcomes, edit events and original costs."""

import asyncio
import json
from pathlib import Path

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult

from .judge import grade_answer
from .rewards import Outcome, prefix_reuse_flops, validate_masks


class DeepResearchEvaluator(BaseTrajectoryEvaluator):
    def __init__(self, question, expected_answer, prompt_type="Single Answer", **kwargs):
        self.question = question
        self.expected_answer = expected_answer
        self.prompt_type = prompt_type

    async def evaluate(self, trajectory, **runtime):
        session = Path(runtime["session_dir"])
        events = json.loads((session / "logs/agent/edit_events.json").read_text())
        if len(events) != len(trajectory.traces):
            raise ValueError("Edit event and authoritative completion counts differ")
        pure_turns = []
        calls = []
        malformed_tool_calls = 0
        for trace, event in zip(trajectory.traces, events):
            messages = trace.response_messages
            tools = messages[0].get("tool_calls", []) if messages else []
            actual_command = None
            if tools:
                try:
                    raw_arguments = tools[0]['function']['arguments']
                    if 'raw_arguments' in event and event['raw_arguments'] != raw_arguments:
                        raise ValueError('Recorded arguments differ from the original model response')
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if tools[0]['function']['name'] == 'bash' and isinstance(arguments, dict):
                        # This is the pinned official harness's exact parser rule.
                        # Original arguments remain in the trace and event.
                        actual_command = str(arguments.get('command') or '').strip()
                except (json.JSONDecodeError, AttributeError, TypeError):
                    malformed_tool_calls += 1
            if event["command"] != actual_command:
                raise ValueError("Edit event is not bound to the original tool arguments")
            pure = bool(event["pure_edit"])
            validate_masks(trace.response_ids, trace.response_logprobs or [], trace.loss_mask,
                           [int(pure)] * len(trace.response_ids))
            pure_turns.append(pure)
            calls.append({"prompt_ids": trace.prompt_ids, "response_ids": trace.response_ids})
        cost, call_costs = prefix_reuse_flops(calls)
        output_path = session / "artifacts/answer.txt"
        response = output_path.read_text() if output_path.exists() else ""
        if response.strip():
            result = await asyncio.to_thread(grade_answer, self.question, self.expected_answer,
                                            response, session / "logs/judge", self.prompt_type,
                                            {'session_id': runtime['session_id'], 'task_id': runtime['task_id'],
                                             'policy_version': trajectory.metadata.get('policy_version'),
                                             **trajectory.metadata.get('task_metadata', {})})
            success = bool(result["success"])
        else:
            result = {"success": False, "reason": "missing_or_empty_final_answer_artifact"}
            success = False
        agent_result = runtime["agent_result"]
        clean = (agent_result.status == "completed" and agent_result.return_code == 0
                 and bool(events) and bool(events[-1].get("submitted")))
        no_tool = not any(item["command"] for item in events)
        format_failure = not response.strip() or malformed_tool_calls > 0
        outcome = Outcome(str(runtime["session_id"]), success, clean, cost, no_tool, format_failure)
        metadata = {"success": success, "clean_finish": clean, "flops": cost,
                    "pure_edit_by_call": pure_turns, "no_tool": no_tool,
                    "format_failure": format_failure, "judge": result, "call_costs": call_costs,
                    "malformed_tool_calls": malformed_tool_calls}
        (session / "logs/clm_outcome.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        return EvalResult(outcome_reward=outcome.reward, metadata={"clm": metadata})
