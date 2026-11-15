# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import os

path = ""
jader_path = path + "pmdacasereport202409/"

reac_file = jader_path + "reac202409.csv"
drug_file = jader_path + "drug202409_utf8.csv"
vaccine_list_file = ""

dt_now = datetime.datetime.now()
today = dt_now.strftime('%Y%m%d')
print(today)

save_dir = r""
os.makedirs(save_dir, exist_ok=True)

df_reac = pd.read_csv(reac_file, encoding="shift-jis")
df_drug = pd.read_csv(drug_file, encoding="utf-8", dtype=str)
df_vaccine = pd.read_csv(vaccine_list_file, encoding="utf-8")

print("REAC 行数:", len(df_reac))
print("REAC ID数:", df_reac["識別番号"].nunique())
print("DRUG 行数:", len(df_drug))
print("DRUG ID数:", df_drug["識別番号"].nunique())

df_reac = df_reac.copy()
df_reac.rename(columns={"有害事象の発現日": "有害事象の発現日_元データ"}, inplace=True)

df_reac["有害事象の発現日"] = (
    df_reac["有害事象の発現日_元データ"]
    .fillna("")
    .astype(str)
    .str[:8]
)

df_8digit = df_reac[df_reac["有害事象の発現日"].str.match(r"^\d{8}$")].copy()
df_6digit = df_reac[df_reac["有害事象の発現日"].str.match(r"^\d{6}$")].copy()

df_8digit["有害事象の発現年月"] = pd.to_datetime(
    df_8digit["有害事象の発現日"],
    format="%Y%m%d",
    errors="coerce"
).dt.strftime("%Y%m")

df_6digit["有害事象の発現年月"] = df_6digit["有害事象の発現日"]

df_8digit = df_8digit[df_8digit["有害事象の発現年月"].notna()].copy()

df_valid_combined_all_period = pd.concat([df_8digit, df_6digit], ignore_index=True)

df_valid_combined = df_valid_combined_all_period[
    pd.to_numeric(df_valid_combined_all_period["有害事象の発現年月"], errors="coerce")
    .between(200501, 201912, inclusive="both")
].copy()

print("日付処理後・期間制限後 REAC 行数:", len(df_valid_combined))
print("日付処理後・期間制限後 REAC ID数:", df_valid_combined["識別番号"].nunique())

required_drug_cols = ["識別番号", "医薬品の関与", "医薬品（一般名）"]
for col in required_drug_cols:
    if col not in df_drug.columns:
        raise ValueError(f"drug ファイルに必須列がありません: {col}")

df_drug_suspect = df_drug[df_drug["医薬品の関与"] == "被疑薬"].copy()
df_drug_suspect["医薬品（一般名）_norm"] = (
    df_drug_suspect["医薬品（一般名）"].fillna("").astype(str).str.strip()
)

print("被疑薬 DRUG 行数:", len(df_drug_suspect))
print("被疑薬 DRUG ID数:", df_drug_suspect["識別番号"].nunique())


if "vaccine_name" not in df_vaccine.columns:
    raise ValueError("vaccine_names_extracted.csv に 'vaccine_name' 列がありません。")

vaccine_names = set(
    df_vaccine["vaccine_name"].dropna().astype(str).str.strip().unique().tolist()
)

vaccine_case_ids = set(
    df_drug_suspect.loc[
        df_drug_suspect["医薬品（一般名）_norm"].isin(vaccine_names),
        "識別番号"
    ].dropna().unique()
)

print("ワクチン症例ID数:", len(vaccine_case_ids))

df_all = df_valid_combined.copy()
df_non_vaccine = df_valid_combined[
    ~df_valid_combined["識別番号"].isin(vaccine_case_ids)
].copy()
df_vaccine_only = df_valid_combined[
    df_valid_combined["識別番号"].isin(vaccine_case_ids)
].copy()

print("全データ ID数:", df_all["識別番号"].nunique())
print("ワクチン除外 ID数:", df_non_vaccine["識別番号"].nunique())
print("ワクチンのみ ID数:", df_vaccine_only["識別番号"].nunique())

target_aes = {
    "Overall": [
        {"en": "Aspiration",              "jp": "誤嚥"},
        {"en": "Guillain-Barre syndrome", "jp": "ギラン・バレー症候群"},
    ],
    "Non-vaccine": [
        {"en": "Appetite loss",                      "jp": "食欲減退"},
        # ▼ 追加：本文DiscussionでS7引用のICI関連AE
        {"en": "Hypothyroidism",                     "jp": "甲状腺機能低下症"},
        {"en": "Hypophysitis",                       "jp": "下垂体炎"},
        {"en": "Secondary adrenal insufficiency",    "jp": "続発性副腎皮質機能不全"},
        {"en": "Fulminant type 1 diabetes mellitus", "jp": "劇症１型糖尿病"},
    ],
    "Vaccine": [
        {"en": "Acute disseminated encephalomyelitis", "jp": "急性散在性脳脊髄炎"},
        {"en": "Guillain-Barre syndrome",              "jp": "ギラン・バレー症候群"},
    ],
}

dataset_dict = {
    "Overall":     df_all,
    "Non-vaccine": df_non_vaccine,
    "Vaccine":     df_vaccine_only,
}

def extract_top10_suspect_drugs_for_ae(
    reac_subset: pd.DataFrame,
    drug_suspect: pd.DataFrame,
    dataset_name: str,
    ae_en: str,
    ae_jp: str,
    top_n: int = 10
) -> pd.DataFrame:

    ae_case_ids = reac_subset.loc[
        reac_subset["有害事象"] == ae_jp, "識別番号"
    ].dropna().unique()

    ae_case_ids = list(ae_case_ids)

    if len(ae_case_ids) == 0:
        return pd.DataFrame([{
            "dataset":          dataset_name,
            "有害事象名_日本語": ae_jp,
            "有害事象名_英語":   ae_en,
            "AE症例ID数":       0,
            "rank":             None,
            "被疑薬名":         None,
            "報告件数":         0
        }])

    drug_tmp = drug_suspect[
        drug_suspect["識別番号"].isin(ae_case_ids)
    ][["識別番号", "医薬品（一般名）_norm"]].copy()

    drug_tmp = drug_tmp.drop_duplicates()

    top_df = (
        drug_tmp.groupby("医薬品（一般名）_norm")["識別番号"]
        .nunique()
        .reset_index(name="報告件数")
        .sort_values(["報告件数", "医薬品（一般名）_norm"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )

    top_df["rank"]          = top_df.index + 1
    top_df["dataset"]       = dataset_name
    top_df["有害事象名_日本語"] = ae_jp
    top_df["有害事象名_英語"]   = ae_en
    top_df["AE症例ID数"]    = len(ae_case_ids)

    top_df = top_df[
        ["dataset", "有害事象名_日本語", "有害事象名_英語", "AE症例ID数",
         "rank", "医薬品（一般名）_norm", "報告件数"]
    ].rename(columns={"医薬品（一般名）_norm": "被疑薬名"})

    return top_df

all_results = []

for dataset_name, reac_subset in dataset_dict.items():
    print(f"\n==================== {dataset_name} ====================")
    for ae_info in target_aes[dataset_name]:
        ae_en = ae_info["en"]
        ae_jp = ae_info["jp"]

        result_df = extract_top10_suspect_drugs_for_ae(
            reac_subset=reac_subset,
            drug_suspect=df_drug_suspect,
            dataset_name=dataset_name,
            ae_en=ae_en,
            ae_jp=ae_jp,
            top_n=10
        )

        all_results.append(result_df)

        print(f"\n[{dataset_name}] {ae_jp} / {ae_en}")
        print(result_df.to_string(index=False))

results_df = pd.concat(all_results, ignore_index=True)


out_file_all = os.path.join(
    save_dir, f"{today}_top10_suspect_drugs_for_7AEs_by_dataset.csv"
)
results_df.to_csv(out_file_all, index=False, encoding="utf-8-sig")
print(f"\n保存完了（全結果）: {out_file_all}")

for dataset_name in results_df["dataset"].dropna().unique():
    tmp = results_df[results_df["dataset"] == dataset_name].copy()
    safe_name = dataset_name.replace(" ", "_").replace("-", "_")
    out_file = os.path.join(
        save_dir, f"{today}_top10_suspect_drugs_{safe_name}.csv"
    )
    tmp.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"保存完了: {out_file}")