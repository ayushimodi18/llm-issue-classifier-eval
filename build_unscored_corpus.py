import pandas as pd
import os

os.makedirs("Data", exist_ok=True)

issues = pd.read_csv(
    "Data/issues.csv"
)

ground_truth = pd.read_csv(
    "Data/ground_truth_dataset.csv"
)

unscored = issues[
    ~issues["issue_id"].isin(
        ground_truth["issue_id"]
    )
]

unscored.to_csv(
    "Data/unscored_corpus.csv",
    index=False
)

print("\nCreated:")
print("Data/unscored_corpus.csv")

print(
    f"Total Issues      : {len(issues)}"
)

print(
    f"Labeled Issues    : {len(ground_truth)}"
)

print(
    f"Unscored Issues   : {len(unscored)}"
)