import pandas as pd

df = pd.read_csv("Data/labeled_issues.csv")

for index, row in df.head(20).iterrows():

    print("\n" + "=" * 70)
    print(f"Issue {index + 1}")
    print("=" * 70)

    print("Title:")
    print(row["title"])

    print("\nDescription Preview:")
    print(str(row["description"])[:300])

    print("\n")