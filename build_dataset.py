"""Prepare the supplied public JSONL dataset for the project.

No internet download occurs here. Put the dataset file at:
data/phishing_and_benign_email_dataset.jsonl
Then run this script to copy it to the training filename:
data/phishing_raw.jsonl
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "phishing_and_benign_email_dataset.jsonl"
target = ROOT / "data" / "phishing_raw.jsonl"

if not source.exists():
    raise SystemExit(
        f"Missing {source.name}. Download the public dataset from the project README and place it in the data folder."
    )

shutil.copyfile(source, target)
print(f"Prepared: {target}")
