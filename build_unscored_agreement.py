import pandas as pd
import json
import os

os.makedirs("Results", exist_ok=True)

df = pd.read_csv(
    "Results/unscored_predictions.csv"
)

# ----------------------------------
# Agreement Logic
# ----------------------------------

all_same = (
    (df["gpt4o_prediction"] == df["gpt41_prediction"])
    &
    (df["gpt41_prediction"] == df["gpt51_prediction"])
)

agreement_count = int(
    all_same.sum()
)

total_issues = len(df)

disagreement_count = (
    total_issues - agreement_count
)

agreement_rate = (
    agreement_count / total_issues
) * 100

# ----------------------------------
# Save Disagreements
# ----------------------------------

disagreements = df[
    ~all_same
]

disagreements.to_csv(
    "Results/unscored_disagreements.csv",
    index=False
)

# ----------------------------------
# Save Agreement Stats
# ----------------------------------

results = {
    "total_issues": total_issues,
    "agreement_count": agreement_count,
    "disagreement_count": disagreement_count,
    "agreement_rate": round(
        float(agreement_rate),
        2
    )
}

with open(
    "Results/unscored_agreement_stats.json",
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )

print("\nCreated:")
print(
    "Results/unscored_agreement_stats.json"
)
print(
    "Results/unscored_disagreements.csv"
)

print("\nAgreement Summary")
print("---------------------")
print(
    f"Total Issues: {total_issues}"
)
print(
    f"Agreement Count: {agreement_count}"
)
print(
    f"Disagreement Count: {disagreement_count}"
)
print(
    f"Agreement Rate: {agreement_rate:.2f}%"
)