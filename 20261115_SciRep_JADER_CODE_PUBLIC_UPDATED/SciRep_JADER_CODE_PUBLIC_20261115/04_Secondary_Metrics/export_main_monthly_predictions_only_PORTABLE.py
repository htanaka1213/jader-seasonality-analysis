# -*- coding: utf-8 -*-
"""JADER Scientific Reports: export all saved revised-monthly forecasts into one CSV ZIP.
This script NEVER fits models and never alters the main results.
This public portable version contains no hard-coded private computer path.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib

# Place the saved main-analysis ZIP next to this script, next to the
# repository directory, or at the repository root; or set SOURCE_ZIP below.
HERE = Path(__file__).resolve().parent
SOURCE_ZIP = None  # e.g. Path('/path/to/SciRep_GitHub_REVISED_20261111_ENVFIX.zip')
OUTPUT_DIR = HERE
OUTPUT_NAME = "MAIN_MONTHLY_PREDICTIONS_ALL_AEs.zip"


def discover_zip():
    if SOURCE_ZIP is not None:
        p = Path(SOURCE_ZIP).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Saved forecast ZIP not found: {p}")
        return p
    name = "SciRep_GitHub_REVISED_20261111_ENVFIX.zip"
    for folder in (HERE, HERE.parent, HERE.parent.parent, Path.cwd()):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {name}. Set SOURCE_ZIP to the saved forecast ZIP path."
    )


def main():
    source_zip = discover_zip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_zip = OUTPUT_DIR / OUTPUT_NAME
    temp_zip = OUTPUT_DIR / (OUTPUT_NAME + ".part")
    n_files = 0
    n_rows = 0
    header = None
    try:
        with ZipFile(source_zip, "r") as source, ZipFile(
            temp_zip, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as out:
            paths = sorted(x for x in source.namelist()
                           if x.endswith("/revised_cv_monthly_paired.csv"))
            if len(paths) != 2952:
                raise ValueError(f"Expected 2,952 source prediction CSVs, found {len(paths)}")
            with out.open("MAIN_MONTHLY_PREDICTIONS_ALL_AEs.csv", "w", force_zip64=True) as dest:
                for path in paths:
                    lines = source.read(path).splitlines(keepends=True)
                    if len(lines) != 133:
                        raise ValueError(f"Expected header + 132 monthly rows: {path}")
                    this_header = lines[0].removeprefix(b"\xef\xbb\xbf").rstrip(b"\r\n")
                    if header is None:
                        header = this_header
                        dest.write(b"\xef\xbb\xbf" + header + b"\r\n")
                    elif this_header != header:
                        raise ValueError(f"Inconsistent CSV column names: {path}")
                    for line in lines[1:]:
                        dest.write(line if line.endswith(b"\n") else line + b"\r\n")
                    n_files += 1
                    n_rows += 132
                    if n_files % 500 == 0:
                        print(f"[EXPORT] {n_files}/2952", flush=True)
        if n_rows != 389664:
            raise RuntimeError(f"Unexpected output rows: {n_rows}")
        temp_zip.replace(output_zip)
    except Exception:
        temp_zip.unlink(missing_ok=True)
        raise
    print(f"[COMPLETE] {n_files} files / {n_rows} monthly forecasts", flush=True)
    print(f"[RESULTS] {output_zip}", flush=True)
    print(f"[SHA256] {hashlib.sha256(output_zip.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
