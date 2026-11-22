import os, sys, json, requests
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def env():
    # credentials are read from the environment only (export SERVER_URL=... API_KEY=...); nothing is stored in the repository
    return {'SERVER_URL': os.environ['SERVER_URL'], 'API_KEY': os.environ['API_KEY']}
TASK_ID = "34-watermark-localisation-image"
def submit(path):
    e = env()
    with open(path, 'rb') as f:
        r = requests.post(f"{e['SERVER_URL']}/submit/{TASK_ID}", headers={"X-API-Key": e['API_KEY']},
                          files={"file": (os.path.basename(path), f, "application/octet-stream")}, timeout=(10, 600))
    try: body = r.json()
    except Exception: body = r.text[:500]
    print(r.status_code, body)
    return body
if __name__ == '__main__':
    submit(sys.argv[1])
