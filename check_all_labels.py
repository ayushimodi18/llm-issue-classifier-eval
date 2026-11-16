import pandas as pd

df = pd.read_csv("Data/issues.csv")

print(df["github_labels"].value_counts().head(50))