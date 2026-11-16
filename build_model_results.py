import pandas as pd
import json
import os
import sys

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from config import MODELS, MAX_WORKERS

os.makedirs("Results", exist_ok=True)

is_parallel = len(sys.argv) > 1 and sys.argv[1].lower() == "parallel"

MODEL_KEYS = ["gpt4o", "gpt41", "gpt51"]
comparison_rows = []

for model_key in MODEL_KEYS:

    model_name = MODELS[model_key]["model"]
    csv_file   = MODELS[model_key]["output_file"]

    if is_parallel:
        csv_file = csv_file.replace(".csv", "_parallel.csv")

    safe_key    = model_key
    output_file = f"Results/{safe_key}_results_parallel.json" if is_parallel else f"Results/{safe_key}_results.json"

    if not os.path.exists(csv_file):
        print(f"Skipping {model_name} — {csv_file} not found")
        continue

    print(f"\nProcessing {model_name}")

    df = pd.read_csv(csv_file)

    # Classification Metrics
    accuracy  = accuracy_score(df["ground_truth"], df["predicted_label"])
    precision = precision_score(df["ground_truth"], df["predicted_label"], average="weighted", zero_division=0)
    recall    = recall_score(df["ground_truth"], df["predicted_label"], average="weighted", zero_division=0)
    f1        = f1_score(df["ground_truth"], df["predicted_label"], average="weighted", zero_division=0)
    per_class = classification_report(df["ground_truth"], df["predicted_label"], output_dict=True, zero_division=0)

    # Latency — per-request stats meaningful even for parallel runs
    p50_latency = float(df["latency_seconds"].quantile(0.50))
    p95_latency = float(df["latency_seconds"].quantile(0.95))
    avg_latency = float(df["latency_seconds"].mean())

    # Wall clock — real elapsed time stored by classifier (not sum of serial latencies)
    if "wall_clock_time" in df.columns:
        wall_clock_time = float(df["wall_clock_time"].iloc[0])
    else:
        wall_clock_time = float(df["latency_seconds"].max())

    concurrency = int(df["concurrency"].iloc[0]) if "concurrency" in df.columns else MAX_WORKERS
    throughput  = len(df) / wall_clock_time if wall_clock_time > 0 else 0

    # Errors
    error_count       = int((df["predicted_label"] == "error").sum())
    error_rate        = (error_count / len(df)) * 100
    rate_limit_errors = int(df["rate_limit_errors"].max()) if "rate_limit_errors" in df.columns else 0
    timeout_errors    = int(df["timeout_errors"].max())    if "timeout_errors"    in df.columns else 0
    other_errors      = int(df["other_errors"].max())      if "other_errors"      in df.columns else 0

    # Cost — traceable via call_cost column
    # call_cost = (prompt_tokens/1k * input_rate) + (completion_tokens/1k * output_rate)
    if "call_cost" in df.columns:
        total_cost    = float(df["call_cost"].sum())
        cost_per_call = float(df["call_cost"].mean())
    else:
        # Legacy fallback for old CSVs without call_cost column
        rate          = MODELS[model_key].get("cost_per_1k_input_tokens", 0.00015)
        total_cost    = (int(df["total_tokens"].sum()) / 1000) * rate
        cost_per_call = total_cost / len(df)

    total_tokens = int(df["total_tokens"].sum())

    labels = sorted(list(set(df["ground_truth"]) | set(df["predicted_label"])))
    cm     = confusion_matrix(df["ground_truth"], df["predicted_label"], labels=labels)

    results = {
        "model": model_name,
        "class_distribution": df["predicted_label"].value_counts().to_dict(),
        "accuracy":          round(float(accuracy),  4),
        "precision":         round(float(precision), 4),
        "recall":            round(float(recall),    4),
        "f1":                round(float(f1),        4),
        "per_class_metrics": per_class,
        "p50_latency":       round(p50_latency, 4),
        "p95_latency":       round(p95_latency, 4),
        "avg_latency":       round(avg_latency, 4),
        "wall_clock_time":   round(wall_clock_time, 2),
        "throughput":        round(float(throughput), 4),
        "concurrency":       concurrency,
        "error_count":       error_count,
        "error_rate":        round(float(error_rate), 2),
        "rate_limit_errors": rate_limit_errors,
        "timeout_errors":    timeout_errors,
        "other_errors":      other_errors,
        "total_tokens":      total_tokens,
        "cost_per_call":     round(cost_per_call, 8),
        "total_cost":        round(total_cost, 6),
        "labels":            labels,
        "confusion_matrix":  cm.tolist(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"  Saved: {output_file}")

    comparison_rows.append({k: v for k, v in results.items() if k not in ("per_class_metrics", "confusion_matrix", "labels", "class_distribution")})

comparison_df     = pd.DataFrame(comparison_rows)
comparison_output = "Results/model_comparison_parallel.csv" if is_parallel else "Results/model_comparison.csv"
comparison_df.to_csv(comparison_output, index=False)

print(f"\nDone! Saved: {comparison_output}")
