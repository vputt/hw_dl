import csv
from pathlib import Path


def write_submission(keys: list[str], scores: list[float], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        for key, score in zip(keys, scores):
            writer.writerow([key, f"{score:.10f}"])
    return output_path
