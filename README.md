# Dataset

The source dataset is the public **Phishing and Benign Email Dataset** by `darkknight25` on Hugging Face:

https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset

The public dataset contains 200 rows and is labelled `phishing` or `benign`.

This project uses a balanced **100-row subset** (50 phishing, 50 benign) copied directly from the public dataset records so the prototype remains quick to train and easy to reproduce.

The file used by `train.py` is:

`data/phishing_raw.jsonl`

No URLs are opened during training or prediction.
