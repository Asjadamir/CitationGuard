"""Fetch the published SciFact release, or install a verified offline archive.

Only the four named JSONL files are copied; archive paths are never extracted.
"""

import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
ARCHIVE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
FILES = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl")
MAX_BYTES = 10 * 1024 * 1024


def install_archive(content: bytes, destination: Path) -> None:
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("SciFact archive checksum mismatch; no files installed.")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        files = {}
        for name in FILES:
            member = archive.getmember(f"data/{name}")
            if not member.isfile() or member.size > 20 * 1024 * 1024:
                raise ValueError(f"Invalid SciFact member: {name}")
            files[name] = archive.extractfile(member).read()
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = destination / name
        temporary = target.with_suffix(".jsonl.tmp")
        temporary.write_bytes(data)
        temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing release archive offline")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/raw/scifact/data")
    args = parser.parse_args()
    try:
        if args.archive:
            content = args.archive.read_bytes()
        else:
            print("Downloading public SciFact data (no credentials; may take several minutes)...")
            with urllib.request.urlopen(DATA_URL, timeout=300) as response:
                content = response.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise ValueError("Archive exceeds the 10 MiB download limit.")
        install_archive(content, args.destination)
    except (OSError, ValueError, KeyError, tarfile.TarError, urllib.error.URLError) as error:
        parser.exit(1, f"Data setup failed: {error}\nRetry later or use --archive with the official release.\n")
    print(f"Verified SHA-256 {ARCHIVE_SHA256}\nInstalled SciFact in {args.destination}")


if __name__ == "__main__":
    main()
