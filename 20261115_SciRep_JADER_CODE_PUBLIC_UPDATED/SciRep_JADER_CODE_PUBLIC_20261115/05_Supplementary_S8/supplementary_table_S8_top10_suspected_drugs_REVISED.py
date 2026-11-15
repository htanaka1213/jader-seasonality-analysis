# -*- coding: utf-8 -*-
"""
Supplementary Table S8 (revised manuscript): top 10 suspected drugs for the adverse events discussed in the text.

Based on supplementary_top10_suspected_drugs.py (code used for the submitted manuscript; former Supplementary Table S7).
Changes from the original code
  1. Target AEs updated to those discussed in the revised manuscript (edit TARGET_AES below if needed).
  2. Counting rule unchanged: number of unique cases (識別番号) in which the drug was recorded as a suspected drug,
     among cases with the AE (onset Jan 2005–Dec 2019) in each dataset.
  3. Ties at the 10th rank are reported (column "tie_at_rank10") instead of being cut silently by drug name.
  4. Dataset sizes are checked against Figure 1 (421,496 / 410,338 / 11,158 cases).
  5. The original DRUG file (cp932) can be read directly; drug202409_utf8.csv is used if it exists.
  6. All outputs are saved in the "output" folder next to this script.

How to run (Spyder): set JADER_DIR and VACCINE_LIST_FILE below, then press F5.
"""
from pathlib import Path
import datetime
import pandas as pd

# ============================== settings (edit here) ==============================
JADER_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_jader"          # folder with reac202409.csv and drug202409(_utf8).csv
VACCINE_LIST_FILE = Path(__file__).resolve().parents[1] / "01_Data_Preparation" / "vaccine_names_extracted.csv"  # same list as in the main analysis
TOP_N = 10

TARGET_AES = {   # dataset -> list of (English name used in the manuscript, JADER term)
    "Non-vaccine": [
        ("Hyponatremia", "低ナトリウム血症"),
        ("Pyrexia", "発熱"),
        ("Decreased appetite", "食欲減退"),
        ("Influenza", "インフルエンザ"),
        ("Hypothermia", "低体温"),
        ("Abnormal behavior", "異常行動"),
    ],
    "Vaccine": [
        ("Guillain–Barre syndrome", "ギラン・バレー症候群"),
        ("Acute disseminated encephalomyelitis", "急性散在性脳脊髄炎"),
    ],
}
EXPECTED_CASES = {"Overall": 421496, "Non-vaccine": 410338, "Vaccine": 11158}   # Figure 1 (before the >=100 filter)
# ===================================================================================

try:
    BASE = Path(__file__).resolve().parent
except NameError:                      # e.g. code pasted into a console
    BASE = Path.cwd()
OUT = BASE.parent / "outputs" / "supplementary_S8"
OUT.mkdir(parents=True, exist_ok=True)
today = datetime.datetime.now().strftime("%Y%m%d")


def read_csv_jp(path, **kw):
    for enc in ("shift-jis", "cp932", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"cannot decode {path}")


# ---------------------------------------------------------------- load
reac_file = JADER_DIR / "reac202409.csv"
drug_utf8 = JADER_DIR / "drug202409_utf8.csv"
drug_file = drug_utf8 if drug_utf8.is_file() else JADER_DIR / "drug202409.csv"
for f in (reac_file, drug_file, VACCINE_LIST_FILE):
    if not f.is_file():
        raise FileNotFoundError(f"File not found: {f}  (check the settings at the top of the script)")

df_reac = read_csv_jp(reac_file, dtype=str)
usecols = ["識別番号", "医薬品の関与", "医薬品（一般名）"]
df_drug = (pd.read_csv(drug_file, encoding="utf-8", dtype=str, usecols=usecols) if drug_file == drug_utf8
           else read_csv_jp(drug_file, dtype=str, usecols=usecols))
df_vaccine = pd.read_csv(VACCINE_LIST_FILE, encoding="utf-8")
print("REAC rows:", len(df_reac), "| DRUG rows:", len(df_drug), "| DRUG file:", drug_file.name)

# ---------------------------------------------------------------- onset date (same rule as the main analysis)
d = df_reac["有害事象の発現日"].fillna("").astype(str).str[:8]
m8, m6 = d.str.match(r"^\d{8}$"), d.str.match(r"^\d{6}$")
r8 = df_reac[m8].copy(); r8["ym"] = pd.to_datetime(d[m8], format="%Y%m%d", errors="coerce").dt.strftime("%Y%m")
r8 = r8[r8["ym"].notna()]
r6 = df_reac[m6].copy(); r6["ym"] = d[m6]
reac = pd.concat([r8, r6], ignore_index=True)
reac = reac[pd.to_numeric(reac["ym"], errors="coerce").between(200501, 201912)].copy()

# ---------------------------------------------------------------- suspected drugs and datasets
sus = df_drug[df_drug["医薬品の関与"] == "被疑薬"].copy()
sus["drug"] = sus["医薬品（一般名）"].fillna("").astype(str).str.strip()
sus = sus[sus["drug"] != ""][["識別番号", "drug"]].drop_duplicates()
vaccine_names = set(df_vaccine["vaccine_name"].dropna().astype(str).str.strip())
vaccine_ids = set(sus.loc[sus["drug"].isin(vaccine_names), "識別番号"])
DATASETS = {"Overall": reac, "Non-vaccine": reac[~reac["識別番号"].isin(vaccine_ids)], "Vaccine": reac[reac["識別番号"].isin(vaccine_ids)]}
for k, g in DATASETS.items():
    n = g["識別番号"].nunique()
    print(f"{k}: {n:,} cases ({len(g):,} reports)", "OK" if n == EXPECTED_CASES[k] else f"<< differs from Figure 1 ({EXPECTED_CASES[k]:,})")

# ---------------------------------------------------------------- top N suspected drugs (unique cases)
rows = []
for ds, aes in TARGET_AES.items():
    g = DATASETS[ds]
    for ae_en, ae_jp in aes:
        ids = g.loc[g["有害事象"] == ae_jp, "識別番号"].dropna().unique()
        if len(ids) == 0:
            print(f"[warning] {ds}: no cases for {ae_jp}"); continue
        cnt = (sus[sus["識別番号"].isin(ids)].groupby("drug")["識別番号"].nunique()
               .reset_index(name="cases").sort_values(["cases", "drug"], ascending=[False, True]).reset_index(drop=True))
        top = cnt.head(TOP_N).copy()
        cut = top["cases"].iloc[-1]
        n_tied_outside = int(((cnt["cases"] == cut).sum()) - (top["cases"] == cut).sum())
        top["rank"] = top["cases"].rank(method="min", ascending=False).astype(int)
        top["percent_of_AE_cases"] = (top["cases"] / len(ids) * 100).round(1)
        top["tie_at_rank10"] = (top["cases"] == cut) & (n_tied_outside > 0)
        top.insert(0, "AE_cases", len(ids)); top.insert(0, "AE_japanese", ae_jp); top.insert(0, "AE_english", ae_en); top.insert(0, "dataset", ds)
        top["drug_english"] = ""          # to be filled in for the table
        rows.append(top)
        print(f"\n[{ds}] {ae_en} ({ae_jp}): {len(ids):,} cases"
              + (f"  * {n_tied_outside} more drug(s) tied with the 10th ({cut} cases)" if n_tied_outside else ""))
        print(top[["rank", "drug", "cases", "percent_of_AE_cases"]].to_string(index=False))

res = pd.concat(rows, ignore_index=True)[["dataset", "AE_english", "AE_japanese", "AE_cases", "rank", "drug", "drug_english",
                                          "cases", "percent_of_AE_cases", "tie_at_rank10"]]
csv_path = OUT / f"{today}_SupplementaryTableS8_top10_suspected_drugs.csv"
res.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("\nsaved:", csv_path)
