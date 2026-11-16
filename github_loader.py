import requests
import pandas as pd

from config import GITHUB_REPO

print(f"Fetching issues from GitHub ({GITHUB_REPO})...")

all_issues = []
page = 1

while True:

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"

    params = {
        "state": "all",
        "per_page": 100,
        "page": page
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error:", response.status_code)
        break

    issues = response.json()

    if len(issues) == 0:
        break

    print(f"Page {page}: {len(issues)} issues downloaded")

    for issue in issues:

        # Collect GitHub labels
        label_names = []

        for label in issue.get("labels", []):
            label_names.append(label.get("name"))

        all_issues.append({
            "issue_id": issue.get("id"),
            "title": issue.get("title"),
            "description": issue.get("body", ""),
            "status": issue.get("state"),
            "github_labels": ", ".join(label_names)
        })

    page += 1

# Create dataframe
df = pd.DataFrame(all_issues)

# Save CSV
df.to_csv("Data/issues.csv", index=False)

print("\nDownload Complete!")
print(f"Total Issues: {len(df)}")
print("Saved to Data/issues.csv")