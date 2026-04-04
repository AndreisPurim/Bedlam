import csv
import os
from typing import Iterable


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


class PerfWriter:
    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        _ensure_parent(self.path)
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
