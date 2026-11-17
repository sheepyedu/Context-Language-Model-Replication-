"""DeepSearchQA answer evaluation through the server's credential wrapper."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from .accounting import record_tokens

MODEL = "deepseek-v4-1-flash-261017"


def parse_grade(text):
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    result = json.loads(value)
    correctness = result["Answer Correctness"]
    details = correctness["Correctness Details"]
    excessive = correctness["Excessive Answers"]
    if not isinstance(details, dict) or not details:
        raise ValueError("Judge did not provide expected answer details")
    if any(type(value) is not bool for value in details.values()):
        raise ValueError("Correctness details must contain JSON booleans")
    if not isinstance(excessive, list) or any(not isinstance(item, str) for item in excessive):
        raise ValueError("Excessive Answers must be a list of strings")
    if not isinstance(correctness["Explanation"], str):
        raise ValueError("Judge explanation must be a string")
    return {
        "success": all(details.values()) and not excessive,
        "matched_count": sum(details.values()),
        "expected_count": len(details),
        "excessive_count": len(excessive),
        "grade": result,
    }


def grade_answer(question, answer, response, output, prompt_type="Single Answer", scope=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # An existing intent is never used to issue a second request.
    result_path = output / "RESULT.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    rubric_path = Path(__file__).with_name("judge_rubric.txt")
    rubric = rubric_path.read_text()
    prompt = rubric.format(prompt=question, answer=answer, response=response, prompt_type=prompt_type)
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 4096,
               "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": prompt}]}
    encoded = json.dumps(payload, ensure_ascii=False)
    intent = {"model": MODEL, 'call_id': uuid.uuid4().hex, "request_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
              "rubric_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
              "created_unix": time.time()}
    with (output / "INTENT.json").open("x") as handle:
        json.dump(intent, handle, indent=2)
    (output / "request.json").write_text(encoded)
    start = time.monotonic()
    process = subprocess.run(
        ["/mount-data/Tools/skill-api", "request", "--service", "deepseek", "--protocol", "chat", "--timeout", "180"],
        input=encoded, text=True, capture_output=True,
    )
    (output / "response.raw").write_text(process.stdout)
    (output / "stderr.txt").write_text(process.stderr)
    receipt = {"exit_code": process.returncode, "elapsed_seconds": time.monotonic()-start}
    (output / "RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    try:
        raw = json.loads(process.stdout)
    except json.JSONDecodeError:
        raw = None
    record_tokens(output, call_id=intent['call_id'], role='judge', model_requested=MODEL,
                  response=raw, scope={**(scope or {}), 'judge_revision': 'judge-no-thinking-v1'}, error={'exit_code': process.returncode} if process.returncode else None)
    if process.returncode:
        raise RuntimeError(f"Judge transport failure, exit={process.returncode}, evidence={output}")
    if raw is None:
        raise RuntimeError('Judge response is not a complete JSON object')
    choice = raw["choices"][0]
    if choice["finish_reason"] != "stop" or raw.get("usage") is None:
        raise RuntimeError("Judge generation is not complete")
    result = parse_grade(choice["message"]["content"])
    result.update({"reported_model": raw.get("model"), "response_id": raw.get("id"),
                   "usage": raw["usage"], "model_requested": MODEL, "judge_revision": "judge-no-thinking-v1"})
    temporary = output / "RESULT.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(result_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    case = json.loads(Path(args.case).read_text())
    print(json.dumps(grade_answer(case["question"], case["answer"], case["response"], args.output,
                                 case.get("prompt_type", "Single Answer")), ensure_ascii=False))
