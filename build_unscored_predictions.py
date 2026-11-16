import pandas as pd
import os

os.makedirs("Results", exist_ok=True)

gpt4o = pd.read_csv(
    "Data/unscored_gpt4o_parallel.csv"
)

gpt41 = pd.read_csv(
    "Data/unscored_gpt41_parallel.csv"
)

gpt51 = pd.read_csv(
    "Data/unscored_gpt51_parallel.csv"
)

results = pd.DataFrame()

results["issue_id"] = gpt4o["issue_id"]
results["title"] = gpt4o["title"]

results["gpt4o_prediction"] = (
    gpt4o["predicted_label"]
)

results["gpt41_prediction"] = (
    gpt41["predicted_label"]
)

results["gpt51_prediction"] = (
    gpt51["predicted_label"]
)

results["gpt4o_raw_output"] = (
    gpt4o["raw_output"]
)

results["gpt41_raw_output"] = (
    gpt41["raw_output"]
)

results["gpt51_raw_output"] = (
    gpt51["raw_output"]
)

results.to_csv(
    "Results/unscored_predictions.csv",
    index=False
)

print("\nCreated:")
print("Results/unscored_predictions.csv")

print(
    f"Rows: {len(results)}"
)