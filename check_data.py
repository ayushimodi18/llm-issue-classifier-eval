import pandas as pd

df = pd.read_csv("Data/issues.csv")

print("\nColumns:")
print(df.columns)

print("\nTotal Issues:")
print(len(df))

print("\nFirst 10 Issues:")
print(df[["issue_id", "title", "status"]].head(10))