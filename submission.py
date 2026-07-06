import os
import sys
import requests
from pathlib import Path


BASE_URL = "http://34.63.153.158" #donot change

API_KEY  = "93684f2bf1d14acedf752c31dbecc931"  # REPLACE WITH YOUR API KEY

TASK_ID   = "22-forging-task" #donot change

FILE_PATH = Path("C:\\Users\\Varun\\OneDrive\\Desktop\\Saarland University\\Semester 1\\TML\\tml2026_task4\\submission.zip") 

SUBMIT = True # set to True to enable submission

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)
OUTPUT_FILE = FILE_PATH

if not os.path.isfile(FILE_PATH):
    die(f"File not found: {FILE_PATH}")

try:
    with open(FILE_PATH, "rb") as f:
        files = {
            "file": (OUTPUT_FILE.name, f, "zip"),
        }
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files=files,
            # timeout=(50, 300),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

    resp.raise_for_status()

    submission_id = body.get("submission_id")
    print("Successfully submitted.")
    print("Server response:", body)
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
    