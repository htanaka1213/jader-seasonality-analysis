# -*- coding: utf-8 -*-
"""Scientific Reports / JADER (2026-11-12): RMSE & MAPE secondary metrics.

READ-ONLY re-aggregation of existing 11-origin x 12-month PRIMARY predictions.
No model fitting. No changes to MAE, IUT, BH/FDR, significance classification.
Uses original submitted metrics definitions:
 RMSE = sqrt(mean((observed - predicted)^2))
 MAPE (%) = 100*mean(abs((observed - predicted)/observed))
 If ANY observed count is zero in a year's 12-month evaluation window,
 that whole year is excluded from MAPE, as in the old ARIMA/SARIMA code.

Outputs annual-CV means (direct fold-wise analogue to prior code),
 and pooled 132-month RMSE (a different aggregation, clearly labelled).
Both MAPE fraction (Prophet legacy convention) and percent (ARIMA legacy
convention) are saved to disclose the old reporting-unit inconsistency.

To use in Spyder: edit PRIMARY_RESULT_SOURCE below (ZIP or directory), press F5.
Place this script beside SciRep_GitHub_REVISED_20261111_ENVFIX.zip to auto-find.
"""
from __future__ import annotations
import io
import codecs
import json
import hashlib
import math
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

# ---------------- SPYDER SETTINGS ----------------
# ZIP (recommended) or extracted folder containing the *main November 11*
# revised forecasts for BOTH Prophet and ARIMA/SARIMA. NOT the sensitivity
# analysis checkpoints. Empty string => auto-discover next to this script.
PRIMARY_RESULT_SOURCE = r""  # auto-detect outputs/main or a local forecast ZIP
OUTPUT_DIR = r""  # Empty => 'RMSE_MAPE_secondary_results' next to script.
STRICT_VALIDATE = True
# Additional reproducibility output: all original monthly predictions in a single ZIP.
# The values and columns are copied directly from the 2,952 MAIN-analysis files.
# No model refitting, no modification of RMSE/MAPE/MAE/IUT/FDR.
EXPORT_MONTHLY_PREDICTIONS = True
MONTHLY_ARCHIVE_NAME = "MAIN_MONTHLY_PREDICTIONS_ALL_AEs.zip"
# -----------------------------------------------

DATASETS = ("overall", "non_vaccine", "vaccine")
ENGINES = ("prophet", "arima_sarima")
EXPECTED_AE = {"overall": 727, "non_vaccine": 698, "vaccine": 51}
EXPECTED_READY = {"overall": 720, "non_vaccine": 692, "vaccine": 47}
FOCUS = {
    ("overall", "低ナトリウム血症"), ("non_vaccine", "低ナトリウム血症"),
    ("non_vaccine", "発熱"), ("vaccine", "注射部位紅斑"),
    ("vaccine", "注射部位腫脹"),
    ("overall", "ギラン・バレー症候群"), ("overall", "誤嚥"),
    ("vaccine", "ギラン・バレー症候群"), ("vaccine", "急性散在性脳脊髄炎"),
}
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_RE = re.compile(
    r"(?:^|/)(overall|non_vaccine|vaccine)/(prophet|arima_sarima)/[^/]+/revised_cv_monthly_paired\.csv$"
)
INFERENCE_RE = re.compile(r"(?:^|/)INFERENCE_BY_METHOD_(overall|non_vaccine|vaccine)\.csv$")


def select_source():
    if PRIMARY_RESULT_SOURCE.strip():
        p = Path(PRIMARY_RESULT_SOURCE).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"予測結果が見つかりません: {p}")
        return p
    # Prefer the sibling main-analysis output folder in the portable repository.
    local_main = SCRIPT_DIR.parent / "outputs" / "main"
    if local_main.is_dir():
        return local_main
    for parent in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "results", Path.cwd()):
        for leaf in ("SciRep_GitHub_REVISED_20261111_ENVFIX.zip",
                     "SciRep_GitHub_REVISED_20261111_ENVFIX"):
            p = parent / leaf
            if p.exists():
                return p
    raise FileNotFoundError(
        "11起点の保存済み予測結果ZIPが見つかりません。"
        "PRIMARY_RESULT_SOURCE に SciRep_GitHub_REVISED_20261111_ENVFIX.zip の絶対パスを指定してください。"
    )


class ReadOnlySource:
    def __init__(self, path):
        self.path = path
        self.zipfile = zipfile.ZipFile(path) if path.is_file() else None
        if self.zipfile:
            self.files = [x for x in self.zipfile.namelist() if x.endswith('.csv')]
        else:
            self.files = [p.relative_to(path).as_posix() for p in path.rglob('*.csv')]

    def read_csv(self, name):
        if self.zipfile:
            return pd.read_csv(io.BytesIO(self.zipfile.read(name)), encoding='utf-8-sig')
        return pd.read_csv(self.path / name, encoding='utf-8-sig')

    def read_bytes(self, name):
        if self.zipfile:
            return self.zipfile.read(name)
        return (self.path / name).read_bytes()

    def close(self):
        if self.zipfile:
            self.zipfile.close()


def t_interval(deltas):
    """Descriptive SECONDARY confidence interval; NEVER used for selection."""
    a = np.asarray(deltas, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return float('nan'), float('nan')
    sd = float(a.std(ddof=1))
    if sd == 0 or not math.isfinite(sd):
        return float('nan'), float('nan')
    m = float(a.mean())
    h = float(t.ppf(0.975, n-1) * sd / np.sqrt(n))
    return m-h, m+h


def summarize_one(df, dataset, engine, expected_status):
    """Fast 11x12 block calculation (read only; never re-fit a model)."""
    required = ('dataset', 'event', 'engine', 'origin_year', 'ds', 'y',
                'yhat_baseline', 'yhat_seasonal')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"月次予測ファイルに列がありません: {dataset}/{engine}: {missing}")
    if len(df) != 132:
        raise ValueError(f"132か月の予測ではありません: {dataset}/{engine}")
    if df['event'].nunique() != 1 or df['dataset'].nunique() != 1 or df['engine'].nunique() != 1:
        raise ValueError('ファイル内に複数のAE/データセット/手法が混在しています')
    event = str(df.event.iloc[0])
    if str(df.dataset.iloc[0]) != dataset or str(df.engine.iloc[0]) != engine:
        raise ValueError('パスとファイル内容のdataset/engineが一致しません')
    df = df.sort_values('ds')
    date_months = pd.to_datetime(df['ds'],errors='raise').to_numpy(dtype='datetime64[M]')
    expected = np.arange(np.datetime64('2009-01','M'),np.datetime64('2020-01','M'),dtype='datetime64[M]')
    if not np.array_equal(date_months,expected):
        raise ValueError(f"月次の欠落/重複: {dataset}/{engine}/{event}")
    years = df['origin_year'].to_numpy(dtype=int)
    if not np.array_equal(years,np.repeat(np.arange(2009,2020),12)):
        raise ValueError(f"起点年の不一致: {dataset}/{engine}/{event}")
    values = df[['y','yhat_baseline','yhat_seasonal']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.min(values[:,0]) < 0:
        raise ValueError(f"欠損/非有限/負の観測値: {dataset}/{engine}/{event}")
    actual = values[:,0].reshape(11,12)
    baseline = values[:,1].reshape(11,12)
    seasonal = values[:,2].reshape(11,12)
    sq_b = np.square(actual-baseline)
    sq_s = np.square(actual-seasonal)
    rmse_b = np.sqrt(np.mean(sq_b,axis=1))
    rmse_s = np.sqrt(np.mean(sq_s,axis=1))
    rmse_pooled_b = float(np.sqrt(np.mean(sq_b)))
    rmse_pooled_s = float(np.sqrt(np.mean(sq_s)))
    mape_valid = np.all(actual>0,axis=1)
    mape_b = np.full(11,np.nan)
    mape_s = np.full(11,np.nan)
    # Never divide by zero: only fully positive years are eligible.
    if mape_valid.any():
        obs_valid=actual[mape_valid]
        mape_b[mape_valid]=np.mean(np.abs((obs_valid-baseline[mape_valid])/obs_valid),axis=1)*100
        mape_s[mape_valid]=np.mean(np.abs((obs_valid-seasonal[mape_valid])/obs_valid),axis=1)*100
    delta_rmse = rmse_b-rmse_s
    delta_mape = mape_b-mape_s
    annual = [{
        'dataset':dataset,'event':event,'engine':engine,'year':int(year),
        'n_months':12,'n_zero_observed_months':int(np.sum(actual[i]==0)),
        'MAPE_year_eligible_no_zero':bool(mape_valid[i]),
        'RMSE_baseline':float(rmse_b[i]),'RMSE_seasonal':float(rmse_s[i]),
        'delta_RMSE':float(delta_rmse[i]),
        'MAPE_baseline_percent':float(mape_b[i]),
        'MAPE_seasonal_percent':float(mape_s[i]),
        'delta_MAPE_percentage_points':float(delta_mape[i]),
    } for i,year in enumerate(range(2009,2020))]
    mae_b=float(np.mean(np.abs(actual-baseline)))
    mae_s=float(np.mean(np.abs(actual-seasonal)))
    rmse_mean_b=float(rmse_b.mean())
    rmse_mean_s=float(rmse_s.mean())
    mape_mean_b=float(mape_b[mape_valid].mean()) if mape_valid.any() else float('nan')
    mape_mean_s=float(mape_s[mape_valid].mean()) if mape_valid.any() else float('nan')
    rmse_low,rmse_high=t_interval(delta_rmse)
    mape_low,mape_high=t_interval(delta_mape[mape_valid])
    summary={
        'dataset':dataset,'event':event,'engine':engine,'main_inference_status':expected_status,
        'n_forecast_months':132,'n_years':11,
        'n_observed_zero_months_132':int(np.sum(actual==0)),
        'n_MAPE_valid_years':int(mape_valid.sum()),
        'n_MAPE_excluded_years':int((~mape_valid).sum()),
        'MAE_baseline_QC_only':mae_b,'MAE_seasonal_QC_only':mae_s,
        'RMSE_baseline_mean_11years':rmse_mean_b,
        'RMSE_seasonal_mean_11years':rmse_mean_s,
        'delta_RMSE_mean_11years':rmse_mean_b-rmse_mean_s,
        'delta_RMSE_year_t95_low_secondary':rmse_low,
        'delta_RMSE_year_t95_high_secondary':rmse_high,
        'RMSE_baseline_pooled_132months':rmse_pooled_b,
        'RMSE_seasonal_pooled_132months':rmse_pooled_s,
        'delta_RMSE_pooled_132months':rmse_pooled_b-rmse_pooled_s,
        'MAPE_baseline_percent_mean_valid_years':mape_mean_b,
        'MAPE_seasonal_percent_mean_valid_years':mape_mean_s,
        'delta_MAPE_percentage_points_mean_valid_years':mape_mean_b-mape_mean_s,
        'delta_MAPE_year_t95_low_pctpts_secondary':mape_low,
        'delta_MAPE_year_t95_high_pctpts_secondary':mape_high,
        'MAPE_baseline_ratio_mean_valid_years':mape_mean_b/100,
        'MAPE_seasonal_ratio_mean_valid_years':mape_mean_s/100,
        'delta_MAPE_ratio_mean_valid_years':(mape_mean_b-mape_mean_s)/100,
    }
    return summary,annual



def export_monthly_predictions(source, predictions, output_dir):
    """Lossless consolidation of the original main-analysis paired prediction CSVs.

    No recalculation or rounding: records are copied from each source CSV as bytes.
    The header is retained once. Every origin-year, observation and forecast stays
    exactly as stored in the source ZIP. Streaming ZIP writer saves disk space.
    """
    import hashlib
    archive_path = output_dir / MONTHLY_ARCHIVE_NAME
    tmp_path = archive_path.with_suffix(".part")
    if tmp_path.exists():
        tmp_path.unlink()
    header_expected = None
    count_files = 0
    count_rows = 0
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as zout:
            with zout.open("MAIN_MONTHLY_PREDICTIONS_ALL_AEs.csv", "w", force_zip64=True) as dest:
                for name, dataset, engine in sorted(predictions):
                    raw = source.read_bytes(name)
                    lines = raw.splitlines(keepends=True)
                    if len(lines) != 133:
                        raise ValueError(f"Expected header + 132 rows, got {len(lines)}: {name}")
                    head = lines[0].lstrip(codecs.BOM_UTF8).rstrip(b"\r\n")
                    if header_expected is None:
                        header_expected = head
                        dest.write(codecs.BOM_UTF8 + head + b"\r\n")
                    elif head != header_expected:
                        raise ValueError(f"Prediction column names differ: {name}")
                    # Preserve each record's original bytes, including full precision.
                    for line in lines[1:]:
                        if not line.endswith((b"\n", b"\r")):
                            line += b"\r\n"
                        dest.write(line)
                    count_files += 1
                    count_rows += 132
                    if count_files % 500 == 0:
                        print(f"[MONTHLY EXPORT] {count_files}/{len(predictions)} source files", flush=True)
        if count_files != 2952 or count_rows != 389664:
            raise RuntimeError(f"Monthly export row count unexpectedly {count_files} files/{count_rows} rows")
        tmp_path.replace(archive_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    manifest = {
        "archive": archive_path.name,
        "file_inside_archive": "MAIN_MONTHLY_PREDICTIONS_ALL_AEs.csv",
        "n_source_csv": count_files,
        "n_prediction_rows": count_rows,
        "columns": header_expected.decode("utf-8").split(","),
        "method": "Lossless merge of original revised_cv_monthly_paired.csv records; one header; no modeling",
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    (output_dir / "MAIN_MONTHLY_PREDICTIONS_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[MONTHLY EXPORT COMPLETE] {count_rows:,} months, {count_files:,} series; {archive_path}", flush=True)
    return manifest


def main():
    src_path = select_source()
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR.strip() else SCRIPT_DIR / 'RMSE_MAPE_secondary_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    source = ReadOnlySource(src_path)
    try:
        inference_paths={}
        predictions=[]
        for name in source.files:
            m = INFERENCE_RE.search(name)
            if m:
                dataset=m.group(1)
                if dataset in inference_paths:
                    raise RuntimeError(f"同名推論表が複数あります: {dataset}")
                inference_paths[dataset]=name
            p=ROOT_RE.search(name)
            if p:
                predictions.append((name,p.group(1),p.group(2)))
        if set(inference_paths)!=set(DATASETS):
            raise RuntimeError(f"推論表3本が見つかりません: {inference_paths}")
        references={}
        for dataset in DATASETS:
            old=source.read_csv(inference_paths[dataset])
            if len(old)!=2*EXPECTED_AE[dataset] or old.duplicated(['event','engine']).any():
                raise ValueError(f"保存済み推論表の件数/重複異常: {dataset}")
            for r in old.itertuples(index=False):
                references[(dataset,str(r.event),str(r.engine))] = r
        print(f'[INPUT] {src_path} | monthly files={len(predictions)}', flush=True)
        if len(predictions) != 2*sum(EXPECTED_AE.values()):
            raise RuntimeError('月次予測ファイル数が想定と異なります。別バージョン/部分的なZIPを指定していないか確認してください')
        summary_rows=[]; annual_rows=[]; found=set(); qc_bad=[]
        for idx,(name,dataset,engine) in enumerate(predictions,1):
            df=source.read_csv(name)
            if len(df)==0:
                raise ValueError(f'空の月次ファイル: {name}')
            event=str(df.event.iloc[0]); key=(dataset,event,engine)
            if key in found:
                raise RuntimeError(f'重複したAE×手法: {key}')
            found.add(key)
            if key not in references:
                raise RuntimeError(f'推論表にないAE: {key}')
            reference=references[key]
            summ,years=summarize_one(df,dataset,engine,str(reference.status))
            if str(reference.status)=='INFERENCE_READY':
                for col_s,col_old in [('MAE_baseline_QC_only','mae_baseline_descriptive'),
                                      ('MAE_seasonal_QC_only','mae_seasonal_descriptive')]:
                    v=float(getattr(reference,col_old))
                    if not np.isclose(summ[col_s],v,rtol=1e-7,atol=1e-7):
                        qc_bad.append((key,col_s,summ[col_s],v))
                if not np.isclose(summ['MAE_baseline_QC_only']-summ['MAE_seasonal_QC_only'],
                                  float(reference.delta_mae),rtol=1e-7,atol=1e-7):
                    qc_bad.append((key,'delta_mae'))
            summary_rows.append(summ)
            annual_rows.extend(years)
            if idx%500==0 or idx==len(predictions):
                print(f'[READ-ONLY] {idx}/{len(predictions)} prediction files aggregated',flush=True)
        if found!=set(references):
            raise RuntimeError(f'予測ファイルと推論表が一致しません (missing={len(set(references)-found)})')
        if STRICT_VALIDATE and qc_bad:
            raise AssertionError(f'MAE original QC mismatch in {len(qc_bad)} entries. First: {qc_bad[:3]}')
        summary=pd.DataFrame(summary_rows).sort_values(['dataset','engine','event'])
        annual=pd.DataFrame(annual_rows).sort_values(['dataset','engine','event','year'])
        if len(summary)!=2952 or len(annual)!=2952*11:
            raise RuntimeError(f'Unexpected output lengths: {len(summary)}, {len(annual)}')
        for d in DATASETS:
            for e in ENGINES:
                part=summary[(summary.dataset==d)&(summary.engine==e)]
                if len(part)!=EXPECTED_AE[d] or int(part.main_inference_status.eq('INFERENCE_READY').sum())!=EXPECTED_READY[d]:
                    raise RuntimeError(f'Expected main inference counts mismatch: {d}/{e}')
        # Same observed counts for Prophet / ARIMA for every AE and forecast year.
        by=annual.groupby(['dataset','event','year']).n_zero_observed_months
        if (by.nunique()!=1).any():
            raise AssertionError('Prophet/ARIMA do not share the same actual monthly-zero counts')
        overview=summary.groupby(['dataset','engine'],sort=True).agg(
            n_AE=('event','size'), n_inference_ready=('main_inference_status',lambda x:(x=='INFERENCE_READY').sum()),
            n_MAPE_available=('n_MAPE_valid_years',lambda x:(x>0).sum()),
            n_MAPE_all11years=('n_MAPE_valid_years',lambda x:(x==11).sum()),
            n_RMSE_11year_available=('RMSE_baseline_mean_11years','count'),
            total_MAPE_excluded_years=('n_MAPE_excluded_years','sum')
        ).reset_index()
        monthly_manifest = None
        if EXPORT_MONTHLY_PREDICTIONS:
            monthly_manifest = export_monthly_predictions(source, predictions, output_dir)
        summary.to_csv(output_dir/'RMSE_MAPE_SECONDARY_ALL_AEs.csv',index=False,encoding='utf-8-sig')
        annual.to_csv(output_dir/'RMSE_MAPE_YEARLY_DETAILS.csv',index=False,encoding='utf-8-sig')
        overview.to_csv(output_dir/'RMSE_MAPE_DATASET_OVERVIEW.csv',index=False,encoding='utf-8-sig')
        summary[summary.apply(lambda r:(r.dataset,r.event) in FOCUS,axis=1)].to_csv(
            output_dir/'RMSE_MAPE_FOCUS_9.csv',index=False,encoding='utf-8-sig')
        manifest={
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'source':str(src_path),
            'source_zip_sha256':hashlib.sha256(src_path.read_bytes()).hexdigest() if src_path.is_file() else None,
            'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,
            'definitions':{
                'RMSE':'sqrt(mean((y-yhat)^2)) per year, then mean over 11 years',
                'MAPE':'100*mean(abs((y-yhat)/y)) per year; year excluded if any y=0; mean over available years',
                'CI':'secondary only, exploratory year-level t CI, never used for AE classification',
                'main_analysis':'MAE/IUT/BH/FDR neither recomputed nor modified',
                'pooled_RMSE':'sqrt(mean(squared error over all 132 months)), a separate aggregation',
                'old_units':'prior Prophet MAPE output fraction; ARIMA output percentage; both supplied here'
            },
            'monthly_prediction_export':monthly_manifest,
            'counts':overview.to_dict(orient='records'),
        }
        (output_dir/'RMSE_MAPE_RUN_MANIFEST.json').write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print('\n[COMPLETE] Read-only RMSE and MAPE from original main 11x12 forecasts',flush=True)
        print(overview.to_string(index=False),flush=True)
        print(f'[RESULTS] {output_dir}',flush=True)
        print(f'[MAE QC] {len(qc_bad)} discrepancies; no main inference modified',flush=True)
    finally:
        source.close()


if __name__=='__main__':
    main()
