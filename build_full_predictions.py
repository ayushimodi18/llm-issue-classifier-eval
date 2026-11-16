import pandas as pd
import os

os.makedirs("Results", exist_ok=True)

print("Loading prediction files...")

gpt4o = pd.read_csv(
    "Data/model_predictions_gpt4omini_parallel.csv"
)

gpt41 = pd.read_csv(
    "Data/model_predictions_gpt41mini_parallel.csv"
)

gpt51 = pd.read_csv(
    "Data/model_predictions_gpt51_parallel.csv"
)

full_predictions = pd.DataFrame()

# ----------------------------------
# Common Fields
# ----------------------------------

full_predictions["title"] = gpt4o["title"]

full_predictions["ground_truth"] = (
    gpt4o["ground_truth"]
)

# ----------------------------------
# Predictions
# ----------------------------------

full_predictions["gpt4o_prediction"] = (
    gpt4o["predicted_label"]
)

full_predictions["gpt41_prediction"] = (
    gpt41["predicted_label"]
)

full_predictions["gpt51_prediction"] = (
    gpt51["predicted_label"]
)

# ----------------------------------
# Raw Outputs
# ----------------------------------

full_predictions["gpt4o_raw_output"] = (
    gpt4o["raw_output"]
)

full_predictions["gpt41_raw_output"] = (
    gpt41["raw_output"]
)

full_predictions["gpt51_raw_output"] = (
    gpt51["raw_output"]
)

# ----------------------------------
# Save File
# ----------------------------------

full_predictions.to_csv(
    "Results/full_predictions.csv",
    index=False
)

print("\nCreated:")
print("Results/full_predictions.csv")

print(
    f"Rows: {len(full_predictions)}"
)

print("\nColumns:")
print(full_predictions.columns.tolist())