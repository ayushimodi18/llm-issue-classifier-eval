from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import time
import sys
import logging

from config import MODELS

# -----------------------------
# Environment Setup
# -----------------------------

load_dotenv()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------
# Error Counters
# -----------------------------

rate_limit_errors = 0
timeout_errors = 0
other_errors = 0

# -----------------------------
# Model Selection
# -----------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python classifier.py gpt4o")
    print("python classifier.py gpt41")
    print("python classifier.py gpt51")
    sys.exit()

model_key = sys.argv[1]

if model_key not in MODELS:
    print("Invalid model.")
    print("Available models:", list(MODELS.keys()))
    sys.exit()

MODEL_NAME = MODELS[model_key]["model"]
OUTPUT_FILE = MODELS[model_key]["output_file"]

print(f"\nRunning Model: {MODEL_NAME}")
print(f"Output File: {OUTPUT_FILE}")

# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv("Data/ground_truth_dataset.csv")

predictions = []
raw_outputs = []
latencies = []

prompt_tokens_list = []
completion_tokens_list = []
total_tokens_list = []

# -----------------------------
# Classification Loop
# -----------------------------

for index, row in df.iterrows():

    title = str(row["title"])
    description = str(row["description"])

    prompt = f"""
Classify this GitHub issue into exactly one category:

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

Return ONLY the label name.

Issue Title:
{title}

Issue Description:
{description[:1000]}
"""

    try:

        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        latency = time.time() - start_time

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        raw_output = response.choices[0].message.content.strip()

        prediction = raw_output.lower()

        predictions.append(prediction)
        raw_outputs.append(raw_output)

        latencies.append(latency)

        prompt_tokens_list.append(prompt_tokens)
        completion_tokens_list.append(completion_tokens)
        total_tokens_list.append(total_tokens)

        print(f"{index + 1}/{len(df)} -> {prediction}")

    except Exception as e:

        error_message = str(e).lower()

        if "429" in error_message:
            rate_limit_errors += 1

        elif "timeout" in error_message:
            timeout_errors += 1

        else:
            other_errors += 1

        logging.error(
            f"Issue {index + 1}: {str(e)}"
        )

        predictions.append("error")
        raw_outputs.append(str(e))

        latencies.append(0)

        prompt_tokens_list.append(0)
        completion_tokens_list.append(0)
        total_tokens_list.append(0)

        print(f"{index + 1}/{len(df)} -> ERROR")

    time.sleep(1)

# -----------------------------
# Save Results
# -----------------------------

df["predicted_label"] = predictions
df["raw_output"] = raw_outputs
df["model_name"] = MODEL_NAME
df["latency_seconds"] = latencies

df["prompt_tokens"] = prompt_tokens_list
df["completion_tokens"] = completion_tokens_list
df["total_tokens"] = total_tokens_list

df["rate_limit_errors"] = rate_limit_errors
df["timeout_errors"] = timeout_errors
df["other_errors"] = other_errors

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nPredictions saved successfully!")

# -----------------------------
# Latency Statistics
# -----------------------------

print("\nLatency Statistics")
print("--------------------------------")

print(
    "Average Latency:",
    round(sum(latencies) / len(latencies), 2),
    "seconds"
)

print(
    "Minimum Latency:",
    round(min(latencies), 2),
    "seconds"
)

print(
    "Maximum Latency:",
    round(max(latencies), 2),
    "seconds"
)

# -----------------------------
# Token Statistics
# -----------------------------

print("\nToken Statistics")
print("--------------------------------")

print(
    "Total Prompt Tokens:",
    sum(prompt_tokens_list)
)

print(
    "Total Completion Tokens:",
    sum(completion_tokens_list)
)

print(
    "Total Tokens:",
    sum(total_tokens_list)
)

# -----------------------------
# Error Statistics
# -----------------------------

print("\nError Statistics")
print("--------------------------------")

print("Rate Limit Errors:", rate_limit_errors)
print("Timeout Errors:", timeout_errors)
print("Other Errors:", other_errors)