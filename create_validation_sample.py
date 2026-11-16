import os
import sys

import pandas as pd

OUTPUT_FILE = "Data/manual_validation.csv"

# Safety guard: don't overwrite manual labels that were already filled in.
if os.path.exists(OUTPUT_FILE) and "--force" not in sys.argv:
    print(f"{OUTPUT_FILE} already exists (it may contain your manual labels).")
    print("Re-run with --force to overwrite it:")
    print("  python create_validation_sample.py --force")
    sys.exit(0)

df = pd.read_csv(
    "Data/ground_truth_dataset.csv"
)

sample_df = df.sample(
    n=100,
    random_state=42
)

sample_df["manual_label"] = ""

sample_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Created {OUTPUT_FILE}"
)
