"""
SHIELD — download_train_data.py
================================
Monitors all pending GEE export tasks and downloads each completed CSV
from Google Drive directly into the local "Train Data" directory.

Run AFTER batch_export.py --no-wait:
    python download_train_data.py

Or specify a custom input file / output folder:
    python download_train_data.py --input my_regions.csv --out "My Data"

How it works
------------
1. Reads output_name list from batch_template.csv (or --input file)
2. Polls GEE task list every 30s until all matching tasks finish
3. For each completed task, searches Google Drive for the CSV by name
4. Downloads it to Train Data/ (skips if already downloaded)

Dependencies (all already installed with GEE):
    earthengine-api, google-api-python-client, google-auth
"""

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "batch_template.csv"
DEFAULT_OUT    = "Rain Data"
POLL_INTERVAL  = 30   # seconds between GEE task status checks

from shield.config import GEE_PROJECT


# ─────────────────────────────────────────────────────────────────────────────
# Google Drive API helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_drive_service():
    """
    Build a Google Drive API service using the same OAuth credentials
    that were saved by ee.Authenticate().
    Credentials are stored in ~/.config/earthengine/credentials.
    """
    import json
    from pathlib import Path as _P
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    cred_path = _P.home() / ".config" / "earthengine" / "credentials"
    if not cred_path.exists():
        raise FileNotFoundError(
            f"OAuth credentials not found at {cred_path}.\n"
            "Run  python -c \"import ee; ee.Authenticate()\"  first."
        )

    raw = json.loads(cred_path.read_text())

    # The EE credentials file stores a refresh token; rebuild a Credentials object
    creds = Credentials(
        token=raw.get("access_token"),
        refresh_token=raw.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=raw.get("client_id"),
        client_secret=raw.get("client_secret"),
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/earthengine",
        ],
    )

    # Refresh if expired
    if not creds.valid:
        creds.refresh(Request())

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_drive_file(service, name: str):
    """
    Search Google Drive for a file named exactly '<name>.csv'.
    Returns the file dict (id, name, mimeType) or None if not found.
    """
    # GEE exports add .csv extension automatically
    query = f"name='{name}.csv' and trashed=false"
    result = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, mimeType, size)",
        pageSize=5,
    ).execute()
    files = result.get("files", [])
    return files[0] if files else None


def _download_file(service, file_id: str, dest_path: Path):
    """Download a Drive file by ID to dest_path."""
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest_path.write_bytes(buf.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
# GEE task polling
# ─────────────────────────────────────────────────────────────────────────────

def _get_task_states(ee, target_names: set) -> dict:
    """
    Returns {output_name: state} for all GEE tasks whose description
    matches one of target_names.
    States: READY | RUNNING | COMPLETED | FAILED | CANCELLED
    """
    tasks  = ee.batch.Task.list()
    states = {}
    for t in tasks:
        desc = t.config.get("description", "")
        if desc in target_names:
            states[desc] = t.status()["state"]
    return states


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Poll GEE tasks and download completed CSVs to Rain Data/"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"CSV/Excel with output_name column (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Local output directory (default: '{DEFAULT_OUT}')"
    )
    parser.add_argument(
        "--poll", type=int, default=POLL_INTERVAL,
        help=f"Seconds between status checks (default: {POLL_INTERVAL})"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir.resolve())

    # ── Load output names ─────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    if input_path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(input_path, dtype=str)
    else:
        df = pd.read_csv(input_path, dtype=str)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    target_names = set(df["output_name"].str.strip().tolist())
    log.info("Watching %d export tasks…", len(target_names))

    # ── Init Earth Engine ────────────────────────────────────────────────────
    import ee
    try:
        ee.Initialize(project=GEE_PROJECT)
        log.info("Earth Engine ready ✓")
    except Exception as e:
        log.error("EE init failed: %s", e)
        sys.exit(1)

    # ── Init Drive ────────────────────────────────────────────────────────────
    log.info("Connecting to Google Drive…")
    try:
        drive = _build_drive_service()
        log.info("Google Drive connected ✓")
    except Exception as e:
        log.error("Drive connection failed: %s", e)
        log.error("Make sure you ran  python -c \"import ee; ee.Authenticate()\"")
        sys.exit(1)

    # ── Poll + download loop ──────────────────────────────────────────────────
    downloaded  = set()
    failed      = set()
    remaining   = set(target_names)

    # Skip names already downloaded
    for name in list(remaining):
        if (out_dir / f"{name}.csv").exists():
            log.info("  Already exists, skipping: %s.csv", name)
            downloaded.add(name)
            remaining.discard(name)

    log.info(
        "Tasks: %d to watch | %d already downloaded | polling every %ds",
        len(remaining), len(downloaded), args.poll
    )
    log.info("Track at: https://code.earthengine.google.com/tasks")

    while remaining:
        states = _get_task_states(ee, remaining)

        for name in list(remaining):
            state = states.get(name, "UNKNOWN")

            if state == "COMPLETED":
                log.info("  ✅ COMPLETED: %s — searching Drive…", name)
                drive_file = _find_drive_file(drive, name)
                if drive_file:
                    dest = out_dir / f"{name}.csv"
                    _download_file(drive, drive_file["id"], dest)
                    size_kb = dest.stat().st_size // 1024
                    log.info("     ⬇  Downloaded → %s  (%d KB)", dest.name, size_kb)
                    downloaded.add(name)
                    remaining.discard(name)
                else:
                    log.warning(
                        "  ⚠  Task COMPLETED but '%s.csv' not found on Drive yet — retrying next poll.",
                        name
                    )

            elif state in ("FAILED", "CANCELLED"):
                log.error("  ❌ %s: %s — skipping.", name, state)
                failed.add(name)
                remaining.discard(name)

            else:
                pass  # READY or RUNNING — keep waiting

        if remaining:
            done_so_far = len(downloaded) + len(failed)
            log.info(
                "Progress: %d/%d done (%d downloading, %d failed) — next check in %ds…",
                done_so_far, len(target_names), len(remaining), len(failed), args.poll
            )
            time.sleep(args.poll)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE")
    print(f"  ✅ Downloaded : {len(downloaded)} files → {out_dir.resolve()}")
    print(f"  ❌ Failed     : {len(failed)} files")
    if failed:
        print("  Failed exports:")
        for n in sorted(failed):
            print(f"    • {n}")
    print("=" * 60)
    print(f"\nNext step: open shield.app and train using files in '{out_dir}'")


if __name__ == "__main__":
    main()
