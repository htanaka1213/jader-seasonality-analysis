# -*- coding: utf-8 -*-

"""
Data preparation code for vaccine-stratified monthly adverse event count data.

This script creates analysis-ready monthly count tables for:
1. non-vaccine dataset
2. vaccine dataset

Analysis environment used by the author:
Python version: 3.9.13
pandas version: 1.4.4
os module: Standard Library
datetime module: Standard Library
"""

import pandas as pd
import datetime
import os
from pathlib import Path

# =========================
# Path settings
# =========================
# Replace these paths with your local environment.
path = str(Path(__file__).resolve().parents[1] / "data")
jader_path = os.path.join(path, "raw_jader")

reac_file = os.path.join(jader_path, "reac202409.csv")
drug_file_utf8 = os.path.join(jader_path, "drug202409_utf8.csv")
drug_file_raw = os.path.join(jader_path, "drug202409.csv")
if os.path.isfile(drug_file_utf8):
    drug_file = drug_file_utf8
    drug_encoding = "utf-8"
elif os.path.isfile(drug_file_raw):
    drug_file = drug_file_raw
    drug_encoding = "cp932"  # original JADER DRUG encoding (no manual conversion needed)
else:
    raise FileNotFoundError(f"Neither drug202409.csv nor drug202409_utf8.csv found in {jader_path}")
vaccine_list_file = str(Path(__file__).resolve().parent / "vaccine_names_extracted.csv")

# Output directory
save_dir = os.path.join(path, "monthly")
os.makedirs(save_dir, exist_ok=True)

# Current date for output file names
dt_now = datetime.datetime.now()
today = dt_now.strftime("%Y%m%d")
print(today)

# =========================
# Load files
# =========================
df_reac = pd.read_csv(reac_file, encoding="shift-jis", dtype={"識別番号": str})
df_drug = pd.read_csv(drug_file, encoding=drug_encoding, dtype=str)
df_vaccine = pd.read_csv(vaccine_list_file, encoding="utf-8")

print("Number of rows in REAC:", len(df_reac))
print("Number of unique case IDs in REAC:", df_reac["識別番号"].nunique())
print("Number of rows in DRUG:", len(df_drug))
print("Number of unique case IDs in DRUG:", df_drug["識別番号"].nunique())

# =========================
# Process onset dates in REAC
# =========================
df_reac.rename(
    columns={"有害事象の発現日": "有害事象の発現日_元データ"},
    inplace=True
)

df_reac["有害事象の発現日"] = (
    df_reac["有害事象の発現日_元データ"]
    .fillna("")
    .astype(str)
    .str[:8]
)

df_8digit = df_reac[
    df_reac["有害事象の発現日"].str.match(r"^\d{8}$")
].copy()

df_6digit = df_reac[
    df_reac["有害事象の発現日"].str.match(r"^\d{6}$")
].copy()

print("Number of unique case IDs in all REAC data:", df_reac["識別番号"].nunique())
print("Number of unique case IDs with 8-digit dates:", df_8digit["識別番号"].nunique())
print("Number of unique case IDs with 6-digit dates:", df_6digit["識別番号"].nunique())

# Create onset year-month
df_8digit["有害事象の発現年月"] = pd.to_datetime(
    df_8digit["有害事象の発現日"],
    format="%Y%m%d"
).dt.strftime("%Y%m")

df_6digit["有害事象の発現年月"] = df_6digit["有害事象の発現日"]

# Combine records with 8-digit and 6-digit onset dates
df_valid_combined_all_period = pd.concat(
    [df_8digit, df_6digit],
    ignore_index=True
)

print(
    "Number of unique case IDs after date processing:",
    df_valid_combined_all_period["識別番号"].nunique()
)

# Restrict the analysis period to January 2005 through December 2019
df_valid_combined = df_valid_combined_all_period[
    df_valid_combined_all_period["有害事象の発現年月"]
    .astype(int)
    .between(200501, 201912)
].copy()

print(
    "Number of unique case IDs from 200501 to 201912:",
    df_valid_combined["識別番号"].nunique()
)

# =========================
# Extract suspected drugs from DRUG
# =========================
required_drug_cols = ["識別番号", "医薬品の関与", "医薬品（一般名）"]

for col in required_drug_cols:
    if col not in df_drug.columns:
        raise ValueError(f"Required column is missing in the DRUG file: {col}")

df_drug_suspect = df_drug[
    df_drug["医薬品の関与"] == "被疑薬"
].copy()

df_drug_suspect["医薬品（一般名）_norm"] = (
    df_drug_suspect["医薬品（一般名）"]
    .astype(str)
    .str.strip()
)

print("Number of rows for suspected drugs:", len(df_drug_suspect))
print("Number of unique case IDs for suspected drugs:", df_drug_suspect["識別番号"].nunique())

# =========================
# Identify vaccine-related case IDs
# =========================
if "vaccine_name" not in df_vaccine.columns:
    raise ValueError("The vaccine list file must contain a column named 'vaccine_name'.")

vaccine_names = set(
    df_vaccine["vaccine_name"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print("Number of vaccine names in the list:", len(vaccine_names))

vaccine_case_ids = set(
    df_drug_suspect.loc[
        df_drug_suspect["医薬品（一般名）_norm"].isin(vaccine_names),
        "識別番号"
    ].unique()
)

print("Number of vaccine-related case IDs:", len(vaccine_case_ids))

# =========================
# Split REAC into non-vaccine and vaccine datasets
# =========================
df_non_vaccine = df_valid_combined[
    ~df_valid_combined["識別番号"].isin(vaccine_case_ids)
].copy()

df_vaccine_only = df_valid_combined[
    df_valid_combined["識別番号"].isin(vaccine_case_ids)
].copy()

print("Number of unique case IDs in the non-vaccine dataset:", df_non_vaccine["識別番号"].nunique())
print("Number of unique case IDs in the vaccine dataset:", df_vaccine_only["識別番号"].nunique())

# =========================
# Create monthly adverse event count tables
# =========================
yearly_counts_non_vaccine = pd.pivot_table(
    df_non_vaccine,
    index="有害事象",
    columns="有害事象の発現年月",
    aggfunc="size",
    fill_value=0
).transpose()

yearly_counts_vaccine_only = pd.pivot_table(
    df_vaccine_only,
    index="有害事象",
    columns="有害事象の発現年月",
    aggfunc="size",
    fill_value=0
).transpose()

# Set index names to match the analysis CSV format
yearly_counts_non_vaccine.index.name = "年月"
yearly_counts_vaccine_only.index.name = "年月"

# =========================
# Save analysis-ready CSV files
# =========================
non_vaccine_file = os.path.join(
    save_dir,
    "20260411_yearly_counts_filtered_non_vaccine.csv"
)

vaccine_only_file = os.path.join(
    save_dir,
    "20260411_yearly_counts_filtered_vaccine_only.csv"
)

yearly_counts_non_vaccine.to_csv(
    non_vaccine_file,
    index=True,
    encoding="utf-8-sig"
)

yearly_counts_vaccine_only.to_csv(
    vaccine_only_file,
    index=True,
    encoding="utf-8-sig"
)

print("Analysis-ready monthly count CSV files saved:")
print("Non-vaccine dataset:", non_vaccine_file)
print("Vaccine dataset:", vaccine_only_file)