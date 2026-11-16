import pandas as pd
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

disagreements.to_csv(
    "Results/disagreements.csv",
    index=False
)

print("\nCreated:")
print("Results/disagreements.csv")

print(
    f"Disagreements: {len(disagreements)}"
)