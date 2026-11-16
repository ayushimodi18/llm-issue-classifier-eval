import pandas as pd

print("Creating ground truth dataset...")

# Read downloaded issues
df = pd.read_csv("Data/issues.csv")

# Mapping GitHub labels to assignment labels
label_map = {
    "bug": "bug",
    "suggestion": "enhancement",
    "enhancement": "enhancement",
    "feature request": "enhancement",

    "docs": "documentation",
    "documentation": "documentation",

    "security vulnerability": "security",
    "security": "security",

    "question": "question",
    "support": "question",

    "duplicate": "other",
    "invalid": "other",
    "wontfix": "other"
}

ground_truth_rows = []

for _, row in df.iterrows():

    labels = str(row.get("github_labels", "")).lower()

    assigned_label = None

    for github_label, final_label in label_map.items():

        if github_label in labels:
            assigned_label = final_label
            break

    if assigned_label:

        ground_truth_rows.append({
            "issue_id": row["issue_id"],
            "title": row["title"],
            "description": row["description"],
            "status": row["status"],
            "ground_truth": assigned_label
        })

# Create dataframe
ground_truth_df = pd.DataFrame(ground_truth_rows)

# Save file
ground_truth_df.to_csv(
    "Data/ground_truth_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("\nGround Truth Dataset Created Successfully!")
print("Rows:", len(ground_truth_df))

print("\nLabel Distribution:")
print(ground_truth_df["ground_truth"].value_counts())
