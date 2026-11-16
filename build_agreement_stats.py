import pandas as pd
import json
import os

os.makedirs("Results", exist_ok=True)

df = pd.read_csv(
    "Results/full_predictions.csv"
)

disagreements = df[

    (df["gpt4o_prediction"] != df["gpt41_prediction"])

    |

    (df["gpt41_prediction"] != df["gpt51_prediction"])

    |

    (df["gpt4o_prediction"] != df["gpt51_prediction"])

]

total_issues = len(df)

disagreement_count = len(disagreements)

agreement_count = (
    total_issues
    - disagreement_count
)

agreement_rate = round(
    (agreement_count / total_issues) * 100,
    2
)

results = {

    "total_issues": total_issues,

    "agreement_count": agreement_count,

    "disagreement_count": disagreement_count,

    "agreement_rate": agreement_rate
}

with open(
    "Results/agreement_stats.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nCreated:")
print("Results/agreement_stats.json")

print(
    f"Agreement Rate: {agreement_rate}%"
)

print(
    f"Disagreements: {disagreement_count}"
)