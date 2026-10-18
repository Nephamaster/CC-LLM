"""Optional, explicit OpenAI-compatible judge; no API calls during ordinary scoring."""
import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..common import ensure_dir, iter_jsonl, read_journal, signature, write_line

# FuxiBench's sft_criteria / sft_prompt, without the langchain runtime dependency.
CRITERIA = "\n" + "\n".join([
    "包含要点: 理解问题和标准回答，之后从标准答案中提取出一个答案要点，预测答案必须包含这个要点。",
    "与标准答案一致: 如果预测答案是一段文字，那么预测答案必须与预标准答案传达的信息一致。",
    "符合问题和任务要求: 如果问题要求创作诗词, 那么回答中的诗词创作必须符合要求的主题和格式。基于对作品的理解进行评判。",
    "预测答案简洁明了: 预测答案需要简洁明了，如果出现大量重复内容则直接判断为错误。",
])
PROMPT = '''作为作为一名阅卷老师，请根据以下评分标准对以下回答进行评价，评价结果的形式是：如果回答正确，输出"Y" ，如果回答错误输出"N"。绝对不要输出额外的内容。
评分标准: {criteria}
<start>
***
[任务输入]: {input}
***
[标准答案]: {output}
***
[预测答案]: {prediction}
***
<end>

[评价结果]:
'''


def prediction_signature(row):
    return signature({"id": row["id"], "input": row["src"], "gold": row["gold"], "prediction": row["predict_clean"]})


def parse_judgment(text):
    lines = [line.strip().strip('。.!！"“” ') for line in text.splitlines() if line.strip()]
    if not lines or lines[-1] not in {"Y", "N"}:
        raise ValueError("judge must end in an unambiguous Y or N")
    return int(lines[-1] == "Y")


def judge_file(predictions_path, output_path, *, client=None):
    model, endpoint = os.environ.get("JUDGE_MODEL"), os.environ.get("JUDGE_BASE_URL")
    if not model or not endpoint:
        raise ValueError("set JUDGE_MODEL and JUDGE_BASE_URL to explicitly enable judging")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query:
        raise ValueError("judge URL must be an HTTP(S) base URL without embedded credentials/query parameters")
    contract = {"model": model, "base_url": endpoint.rstrip("/"), "prompt_sha256": signature([PROMPT, CRITERIA])}
    output = Path(output_path)
    existing = {}
    if output.exists():
        for entry in read_journal(output):
            if entry["judge"] != contract:
                raise ValueError("judge configuration changed; use a new judgment file")
            existing[entry["id"]] = entry

    def request(prompt):
        data = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": 1024}).encode()
        headers = {"Content-Type": "application/json"}
        if os.environ.get("JUDGE_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["JUDGE_API_KEY"]
        req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    client = client or request
    ensure_dir(output.parent)
    pending = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in iter_jsonl(predictions_path):
            if row.get("error"):
                raise ValueError("finish successful inference before requesting judgments")
            if row.get("metric") != "judge":
                continue
            fingerprint = prediction_signature(row)
            prior = existing.get(row["id"])
            if prior:
                if prior["prediction_sha256"] != fingerprint:
                    raise ValueError("prediction changed since judgment")
                if prior.get("score") in (0, 1):
                    continue
            entry = {"id": row["id"], "prediction_sha256": fingerprint, "judge": contract}
            try:
                raw = client(PROMPT.format(criteria=CRITERIA, input=row["src"], output=row["gold"], prediction=row["predict_clean"]))
                entry.update(raw=raw, score=parse_judgment(raw), error=None)
            except Exception as exc:
                # Never serialize an HTTP exception body which could echo credentials.
                entry.update(score=None, error=type(exc).__name__)
                pending += 1
            write_line(handle, entry)
            handle.flush()
    return {"pending": pending, "complete": pending == 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = judge_file(args.predictions, args.output)
    print(report)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
