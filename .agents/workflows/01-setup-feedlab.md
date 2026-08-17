---
description: Initializes the FeedLab project architecture, installs local ML dependencies, and sets up the SQLite database with an automated JSON account seeder.
---

## Steps

1. **Initialize Project Structure**
   - Create directories: `perception/`, `agent/`, `automation/`, `database/`, and `dashboard/`.
   - Create empty `__init__.py` files in each.
   - Create `config.py` in the root with:
     - `LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"`
     - `LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"`
     - `TARGET_BIAS = "Your Bias Here"`

2. **Setup Dependencies**
   - Create `requirements.txt` containing: `mlx-whisper`, `easyocr`, `opencv-python`, `Appium-Python-Client`, `bertopic`, `streamlit`, `pandas`, `requests`, `sqlite3`.
   - Execute `pip install -r requirements.txt` using the terminal tool.

3. **Build Database Schema**
   - Create `database/db.py`. Write a script using `sqlite3` to create two tables in `reels.db`:
     - `accounts`: `username` (PK), `password`, `status` (WARMING_UP, ACTIVE, COOLDOWN, BANNED), `daily_scroll_minutes`, `last_used_timestamp`.
     - `reels`: `id` (PK), `timestamp`, `account_username`, `transcript`, `ocr_text`, `alignment_score`, `action_taken`.

4. **Account Seeder Script**
   - Create `database/seed.py`.
   - Write a Python script that reads `accounts.json` from the root directory.
   - Iterate through the JSON and execute an `INSERT OR IGNORE` into the `accounts` table to populate the database without duplicating existing entries.
   - Write a CLI execution block `if __name__ == "__main__":` that prints a table of the imported accounts and their current statuses.