# -*- coding: utf-8 -*-

"""
Data preparation code for monthly adverse event count data.

This script creates the analysis-ready monthly count table used for
the overall JADER dataset.

Analysis environment used by the author:
Python version: 3.9.13
pandas version: 1.4.4
os module: Standard Library
datetime module: Standard Library
"""

import pandas as pd
import datetime
import os

# =========================
# Path settings
# =========================
# Replace these paths with your local environment.
path = r"C:/Users/YourName/YourFolder/"
jader_path = os.path.join(path, "pmdacasereport202409")

reac_file = os.path.join(jader_path, "reac202409.csv")

# Output directory
save_dir = r"C:/Users/YourName/YourOutputFolder"
os.makedirs(save_dir, exist_ok=True)

# Current date for output file name
dt_now = datetime.datetime.now()
today = dt_now.strftime("%Y%m%d")
print(today)

# =========================
# Load REAC file
# =========================
df_reac = pd.read_csv(reac_file, encoding="shift-jis")

print("Number of rows in REAC:", len(df_reac))
print("Number of unique case IDs in REAC:", df_reac["識別番号"].nunique())

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
# Create monthly adverse event count table
# =========================
yearly_counts = pd.pivot_table(
    df_valid_combined,
    index="有害事象",
    columns="有害事象の発現年月",
    aggfunc="size",
    fill_value=0
).transpose()

# Set index name to match the analysis CSV format
yearly_counts.index.name = "年月"

# =========================
# Save analysis-ready CSV
# =========================
out_file = os.path.join(
    save_dir,
    today + "_yearly_counts_filtered.csv"
)

yearly_counts.to_csv(
    out_file,
    index=True,
    encoding="utf-8-sig"
)

print("Analysis-ready monthly count CSV saved:")
print(out_file)