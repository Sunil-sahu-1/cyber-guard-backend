"""Download and prepare public Cyber Guard training datasets.

Large/licensed datasets are intentionally not committed to Git.
Run this from the backend root:

    python -m ml_training.download_datasets

For sources that require registration/terms acceptance, this script prints
the official download page instead of bypassing access controls.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"

SOURCES = {
    "ember2018": (
        "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2",
        DATA / "malware" / "ember" / "ember_dataset_2018_2.tar.bz2",
        "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812",
    ),
    "asvspoof2021_df": (
        "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval.tar.gz?download=1",
        DATA / "voice" / "asvspoof2021_df" / "ASVspoof2021_DF_eval.tar.gz",
        None,
    ),
    "asvspoof2021_pa": (
        "https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval.tar.gz?download=1",
        DATA / "voice" / "asvspoof2021_pa" / "ASVspoof2021_PA_eval.tar.gz",
        None,
    ),
}

MANUAL = {
    "FaceForensics++": "https://github.com/ondyari/FaceForensics",
    "DFDC": "https://ai.meta.com/datasets/dfdc/",
    "CIC-IDS2017": "https://www.unb.ca/cic/datasets/ids-2017.html",
    "PhishTank": "https://phishtank.org/developer_info.php",
    "Enron Email": "https://www.cs.cmu.edu/~enron/",
    "SpamAssassin corpus": "https://spamassassin.apache.org/publiccorpus/",
}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[skip] {target}")
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, target)
    print(f"[saved] {target}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:bz2"
        print(f"[extract] {path.name}")
        with tarfile.open(path, mode) as tf:
            tf.extractall(destination)
    else:
        print(f"[manual extraction required] {path}")


def main() -> None:
    for p in [
        DATA / "voice" / "asvspoof2021_df",
        DATA / "voice" / "asvspoof2021_pa",
        DATA / "deepfake" / "faceforensics",
        DATA / "deepfake" / "dfdc",
        DATA / "network" / "cic_ids2017",
        DATA / "malware" / "ember",
        DATA / "phishing" / "urls",
        DATA / "phishing" / "emails",
        DATA / "behavior",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    for name, (url, target, expected) in SOURCES.items():
        try:
            download(url, target)
            if expected:
                actual = sha256(target)
                if actual != expected:
                    raise RuntimeError(
                        f"{name}: SHA256 mismatch. expected={expected}, actual={actual}"
                    )
            print(f"[ok] {name}")
        except Exception as exc:
            print(f"[error] {name}: {exc}")

    print("\nManual/terms-based sources:")
    for name, url in MANUAL.items():
        print(f"  - {name}: {url}")

    print("\nDo not commit dataset files. The repository .gitignore excludes datasets/*.")


if __name__ == "__main__":
    main()
