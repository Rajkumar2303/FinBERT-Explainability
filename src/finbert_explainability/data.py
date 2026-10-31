"""Load the pinned original archive without executing remote dataset scripts."""
import hashlib
import unicodedata
from pathlib import Path
from zipfile import ZipFile

from datasets import ClassLabel, Dataset, Features, Value
from huggingface_hub import hf_hub_download

LABELS = ["negative", "neutral", "positive"]


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_subset(archive, filename):
    with ZipFile(archive) as z:
        matches = [n for n in z.namelist() if n.split("/")[-1] == filename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {filename} in archive")
        rows = []
        for line in z.read(matches[0]).decode("latin-1").splitlines():
            if not line.strip():
                continue
            sentence, label = line.rsplit("@", 1)
            if not sentence.strip() or label.strip() not in LABELS:
                raise ValueError(f"Malformed row in {filename}")
            rows.append({"sentence": sentence.strip(), "label": LABELS.index(label.strip())})
    return Dataset.from_list(rows, features=Features({"sentence": Value("string"), "label": ClassLabel(names=LABELS)}))


def annotate_membership(base, unanimous):
    keys = {(normalize(r["sentence"]), r["label"]) for r in unanimous}
    base_keys = {(normalize(r["sentence"]), r["label"]) for r in base}
    if keys - base_keys:
        raise ValueError("allagree contains sentence/label pairs absent from 50agree")
    rows = []
    for i, row in enumerate(base):
        rows.append({**row, "row_id": f"fpb50-{i:05d}", "source_index": i,
                     "is_allagree": (normalize(row["sentence"]), row["label"]) in keys})
    return rows


def load_source(config, archive=None):
    path = Path(archive) if archive else Path(hf_hub_download(
        repo_id=config.dataset_repo, repo_type="dataset", revision=config.revision,
        filename="data/FinancialPhraseBank-v1.0.zip", cache_dir=config.cache_dir))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config.archive_sha256:
        raise ValueError("Source archive SHA256 does not match configuration")
    base = read_subset(path, "Sentences_50Agree.txt")
    unanimous = read_subset(path, "Sentences_AllAgree.txt")
    return annotate_membership(base, unanimous), {"archive_sha256": digest,
        "subset": "sentences_50agree", "source_rows": len(base), "allagree_source_rows": len(unanimous)}
