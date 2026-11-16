import pandas as pd
import json
import os

os.makedirs("Results", exist_ok=True)

df = pd.read_csv(
    "Results/unscored_predictions.csv"
)

results = {}

models = [
    "gpt4o_prediction",
    "gpt41_prediction",
    "gpt51_prediction"
]

for model in models:

    counts = (
        df[model]
        .value_counts()
        .to_dict()
    )

    results[model] = counts

with open(
    "Results/unscored_class_distribution.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nCreated:")
print(
    "Results/unscored_class_distribution.json"
)