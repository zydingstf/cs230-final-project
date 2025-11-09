from pathlib import Path
import pandas as pd
import emoji

RAW_DIR = Path("raw").resolve()
OUT_PATH = Path("combined.csv")
EXCLUDE = ["rabbit", "rabbit_face", "egg", "hatching_egg"]
from pathlib import Path
import pandas as pd
import csv, re
from io import StringIO

def _sniff_delimiter(path, enc):
    sample = open(path, "rb").read(256_000)  # small peek
    try:
        txt = sample.decode(enc, errors="ignore")
        return csv.Sniffer().sniff(txt, delimiters=[",", "\t", ";", "|"]).delimiter
    except Exception:
        return ","  # default

def _sanitize_text(text: str) -> str:
    # Remove NULLs and control chars (keep \n, \r, \t)
    text = text.replace("\x00", "")
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

def safe_read_csv(path, chunksize=None, expected_sep=None):
    """
    Robust CSV reader that:
    - tries multiple encodings
    - falls back to Python engine if C engine fails
    - sniffs delimiter
    - sanitizes control chars if needed
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    for enc in encodings:
        sep = expected_sep or _sniff_delimiter(path, enc)

        # 1) Try fast C engine with forgiving options
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                sep=sep,
                engine="c",
                on_bad_lines="skip",   # skip broken rows
                low_memory=False,
                dtype=str,             # avoid type inference surprises
                chunksize=chunksize,
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
            )
        except pd.errors.ParserError:
            pass  # fall through to python engine

        # 2) Try Python engine (more tolerant to bad quoting/long fields)
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
                chunksize=chunksize,
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
            )
        except pd.errors.ParserError:
            pass

        # 3) Last resort: sanitize text then parse
        try:
            with open(path, "r", encoding=enc, errors="ignore", newline="") as f:
                raw = f.read()
            cleaned = _sanitize_text(raw)
            return pd.read_csv(
                StringIO(cleaned),
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
                chunksize=chunksize,
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
            )
        except Exception:
            continue

    # If we’re here, everything failed
    raise pd.errors.ParserError(f"Could not parse CSV: {path}")

def strip_emojis_from_df(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) == 0:
        return df
    # replace emojis in every object column
    for c in obj_cols:
        # convert to str to be safe, then remove emojis
        df[c] = df[c].astype(str).map(lambda x: emoji.replace_emoji(x, replace=""))
    return df

frames = []
for csv_path in sorted(RAW_DIR.glob("*.csv")):
    print(f"Handling{csv_path}")
    label = csv_path.stem
    if label in EXCLUDE:
        print(f"Skip{label}")
        continue;
    reader = safe_read_csv(csv_path, chunksize=None)
    if isinstance(reader, pd.DataFrame):
        df = reader
    else:
        df = pd.concat(reader, ignore_index=True)
    df.insert(0, "label", label)  # put label as first column
    df = strip_emojis_from_df(df)
    frames.append(df)

combined = pd.concat(frames, axis=0, ignore_index=True)
combined.to_csv(OUT_PATH, index=False)
print(f"Done. Wrote {len(combined):,} rows to {OUT_PATH}")
