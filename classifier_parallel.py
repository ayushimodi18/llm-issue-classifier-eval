from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import time
import sys
import logging
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MODELS, MAX_WORKERS

load_dotenv()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# OpenAI-compatible Inference Client


API_KEY  = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv(
    "BASE_URL",
    "https://api.openai.com/v1"
)

if not API_KEY:
    print("ERROR: OPENAI_API_KEY not set in environment.")
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# Model Selection


if len(sys.argv) < 2:
    print("Usage:")
    print("  python classifier_parallel.py gpt4o")
    print("  python classifier_parallel.py gpt41")
    print("  python classifier_parallel.py gpt51")
    sys.exit()

model_key = sys.argv[1]

if model_key not in MODELS:
    print("Invalid model. Available:", list(MODELS.keys()))
    sys.exit()

MODEL_NAME          = MODELS[model_key]["model"]
COST_PER_1K_INPUT   = MODELS[model_key]["cost_per_1k_input_tokens"]
COST_PER_1K_OUTPUT  = MODELS[model_key]["cost_per_1k_output_tokens"]
OUTPUT_FILE         = MODELS[model_key]["output_file"].replace(".csv", "_parallel.csv")

print(f"\nRunning Model  : {MODEL_NAME}")
print(f"Output File    : {OUTPUT_FILE}")
print(f"Concurrency    : {MAX_WORKERS}")
print(f"Base URL       : {BASE_URL}")


# Worker Function


VALID_LABELS = {"bug", "enhancement", "question", "documentation", "security", "other"}

def process_issue(index, row):
    title       = str(row["title"])
    description = str(row["description"])

    prompt = f"""Classify this GitHub issue into exactly one category:

bug
enhancement
question
documentation
security
other

Definitions:
bug = something is broken or not working correctly
enhancement = feature request or improvement
question = user asking for help, clarification, or usage guidance
documentation = README, docs, examples, guides, documentation updates
security = vulnerabilities, permissions, authentication, secrets
other = anything that does not fit the above categories

Return ONLY the label name. No explanation, no punctuation.

Issue Title:
{title}

Issue Description:
{description[:1000]}
"""

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0,
        )

        latency    = time.time() - start_time
        raw_output = response.choices[0].message.content.strip()
        prediction = raw_output.lower().strip()

        # Normalize — extract first word if model returns extra text
        if prediction not in VALID_LABELS:
            first_word = prediction.split()[0] if prediction else "other"
            prediction = first_word if first_word in VALID_LABELS else "other"

        prompt_tokens     = response.usage.prompt_tokens     if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens      = response.usage.total_tokens      if response.usage else 0

        # Cost traceable: (input_tokens/1k * input_rate) + (output_tokens/1k * output_rate)
        call_cost = (
            (prompt_tokens     / 1000) * COST_PER_1K_INPUT
          + (completion_tokens / 1000) * COST_PER_1K_OUTPUT
        )

        return {
            "index": index, "prediction": prediction, "raw_output": raw_output,
            "latency": latency, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "total_tokens": total_tokens,
            "call_cost": call_cost, "error_type": "none"
        }

    except Exception as e:
        msg = str(e).lower()
        logging.error(f"Issue {index} failed: {e}")

        error_type = "rate_limit" if ("429" in msg or "rate" in msg) else \
                     "timeout"    if ("timeout" in msg or "timed out" in msg) else "other"

        return {
            "index": index, "prediction": "error", "raw_output": str(e),
            "latency": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "call_cost": 0, "error_type": error_type
        }


# Load & Run


df      = pd.read_csv("Data/ground_truth_dataset.csv")
results = []

wall_clock_start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures   = [executor.submit(process_issue, i, row) for i, row in df.iterrows()]
    completed = 0
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        completed += 1
        print(f"{completed}/{len(df)} -> {result['prediction']}")

wall_clock_elapsed = time.time() - wall_clock_start

results.sort(key=lambda x: x["index"])

rate_limit_errors = sum(1 for r in results if r["error_type"] == "rate_limit")
timeout_errors    = sum(1 for r in results if r["error_type"] == "timeout")
other_errors      = sum(1 for r in results if r["error_type"] == "other")

df["predicted_label"]   = [r["prediction"]        for r in results]
df["raw_output"]        = [r["raw_output"]         for r in results]
df["model_name"]        = MODEL_NAME
df["latency_seconds"]   = [r["latency"]            for r in results]
df["prompt_tokens"]     = [r["prompt_tokens"]      for r in results]
df["completion_tokens"] = [r["completion_tokens"]  for r in results]
df["total_tokens"]      = [r["total_tokens"]       for r in results]
df["call_cost"]         = [r["call_cost"]          for r in results]
df["wall_clock_time"]   = wall_clock_elapsed
df["concurrency"]       = MAX_WORKERS
df["rate_limit_errors"] = rate_limit_errors
df["timeout_errors"]    = timeout_errors
df["other_errors"]      = other_errors

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

total_cost = df["call_cost"].sum()
throughput = len(df) / wall_clock_elapsed if wall_clock_elapsed > 0 else 0
latencies  = df["latency_seconds"]

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Wall Clock : {wall_clock_elapsed:.2f}s  (concurrency={MAX_WORKERS})")
print(f"Throughput : {throughput:.2f} req/s")
print(f"p50 latency: {latencies.quantile(0.50):.2f}s")
print(f"p95 latency: {latencies.quantile(0.95):.2f}s")
print(f"Total Cost : ${total_cost:.6f}  |  Per call: ${total_cost/len(df):.8f}")
print(f"Errors     : rate_limit={rate_limit_errors}  timeout={timeout_errors}  other={other_errors}")
