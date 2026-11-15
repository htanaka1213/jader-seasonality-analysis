ARCHIVED NOTE: This document records an earlier local workflow. For paths and execution steps in this packaged repository, follow the root README.md instead.

# JADER adverse-event seasonality — Scientific Reports revision (2026-11-11)

## 何を改訂したか

これは投稿時の `GitHub_SourceCode` にある以下の6本を基礎とする**再解析版**です。投稿時のファイルは残し、本ZIPに含まれる `_REVISED_20261111.py` を新たな版として公開してください。

| 投稿時のコード | 改訂版 |
|---|---|
| `run_prophet_overall_analysis.py` | `run_prophet_overall_analysis_REVISED_20261111.py` |
| `run_prophet_non_vaccine_analysis.py` | `run_prophet_non_vaccine_analysis_REVISED_20261111.py` |
| `run_prophet_vaccine_analysis.py` | `run_prophet_vaccine_analysis_REVISED_20261111.py` |
| `run_arima_sarima_overall_analysis.py` | `run_arima_sarima_overall_analysis_REVISED_20261111.py` |
| `run_arima_sarima_non_vaccine_analysis.py` | `run_arima_sarima_non_vaccine_analysis_REVISED_20261111.py` |
| `run_arima_sarima_vaccine_analysis.py` | `run_arima_sarima_vaccine_analysis_REVISED_20261111.py` |

新たな共通処理は `scirep_revision_cv_core_20261111.py`、入力・出力設定は `scirep_revision_config_20261111.py`、推論集計は `summarize_IUT_BH_REVISED_20261111.py` にあります。GitHubでは**6本の実行ファイルと共通処理・設定・集計・READMEを同時に公開**してください。

**重要：今回は単純なファイル名変更ではありません。**

- 投稿時の Prophet `yearly_seasonality=True/False` と標準設定を保持しつつ、毎年12月末までの学習データから翌年1–12月を予測する **11起点×12か月** に変更（従来の`cross_validation(..., horizon='365 days')`ではなく、明示的な暦月スライス）。
- 投稿時の ARIMA/SARIMA 5-fold 30か月予測を **同じ11起点×12か月** に変更。
- ARIMA/SARIMAのAIC探索を、比較可能な尤度計算対象月数となるよう**差分次数固定** `(d=1,D=0)`、 `trend='n'`、定常性・可逆性制約ありへ変更。ARIMAは `(p,1,q)` 4候補、SARIMAは `(p,1,q)(P,0,Q)[12]` 12候補 (`P+Q>0`)。本仕様は先に検証したcontrolled AIC pilotに由来しますが、**あらゆるAEに最適だと証明されたものではありません**。研究上の主要解析として固定する前に研究者間で仕様を承認してください。
- 各起点ごとに予測値、絶対誤差、モデル選択、収束状況を保存。初期48か月0件のAEは記述的結果を残す一方、両手法とも主要なCI/p値なし。
- 生の132か月分の **ΔMAE = mean(|error_baseline|−|error_seasonal|)** を用いる。11個の年別ΔMAEから探索的t区間と両側p値、HAC(12)による感度分析を算出。
- 両手法の改善について、ΔMAEが正なら両側p値、0以下なら1とした後、**IUT = max(p_Prophet, p_ARIMA/SARIMA)** を算出。**各データセット内**でこの共同p値にBenjamini–Hochberg (BH)補正、q < 0.05を記述。
- 計算済みAEが一部しかない場合、**BH q値は計算しない**。

## Spyderでの実行方法

1. ZIPの中身を**すべて同じフォルダー**へ展開します。`scirep_revision_config_20261111.py`の`INPUT_FILES`と`OUTPUT_ROOT`を確認してください。初期値は `Path.home()/OneDrive/...` を使うため、Windowsログイン名はコードに埋め込まれていません。
2. 初期設定の `MAX_EVENTS_PER_DATASET = 1` のまま、Spyderで `run_ALL_6_REVISED_20261111_spyder.py` を開き、**F5**で動作確認します。6通りの解析で各1AEだけ処理します。利用環境にProphetがない場合は該当処理で停止します。
3. 出力された各AEの `years/2009_predictions.csv` から `years/2019_predictions.csv`、`revised_cv_monthly_paired.csv`、`revised_event_summary.csv` を確認します。
4. **全AE処理に進む場合**、設定ファイルで `MAX_EVENTS_PER_DATASET = None`、`SELECTED_EVENTS = None` に変更して同じスクリプトを再実行します。完了した年の結果は、入力・コード・環境のfingerprintが同一なら再利用されます。停止・再開可能ですが、全AEは非常に長時間かかる可能性があります。特にSARIMAは起点ごとに16候補を最適化します。小規模実測時間から全件の完了時間を断定しないでください。
5. **3データセット×2手法の出力がそろったら**、Spyderで `summarize_IUT_BH_REVISED_20261111.py` をF5で実行します。パイロット時にもこのファイルを実行できますが、`QC_PROGRESS_ALL_DATASETS.csv` と暫定QCのみが出力され、FDRの本番値は出ません。

6本を個別に実行したい場合は、それぞれの `_REVISED_20261111.py` をSpyderでF5実行します。Spyderの `runcell` が `__name__` の扱いを変更する場合は、F5の全ファイル実行を使用してください。

## 出力構成

```
SciRep_GitHub_REVISED_20261111/
  overall/
    prophet/
      RUN_MANIFEST.json
      RUN_STATUS.csv
      <AE名とハッシュ>/
        years/2009_predictions.csv ... 2019_predictions.csv
        years/2009_metrics.json ... 2019_metrics.json
        revised_cv_monthly_paired.csv
        revised_origin_metrics.csv
        revised_event_summary.csv
    arima_sarima/ ...同様...
  non_vaccine/ ...同様...
  vaccine/ ...同様...
  QC_PROGRESS_ALL_DATASETS.csv
  INFERENCE_BY_METHOD_overall.csv ... (各手法の数値)
  QC_JOINT_PARTIAL_overall.csv ... (全AE未完了時、qなし)
  IUT_BH_RESULTS_overall.csv ... (全AE完了時のみ)
  IUT_BH_RESULTS_ALL_DATASETS.csv (3データセットが全部完了時のみ)
```

**規則**：15年間に100件以上の全AEを対象（727 + 698 + 51）。初期48か月が0件の17組は**両手法とも**主要な推論判定不能として保存する。初期以外の全ゼロ学習期間にも、両モデルとも0を予測する補助処理を使う。報告が1件以上あるのに適合に失敗した起点にはゼロ予測を代入せず、`UNRESOLVED` とする。推定に失敗した起点が1つでもあればIUT/BHの推論対象には含めない（ただし得られた予測の記述的要約は残す）。

## 統計学上の限界・投稿原稿における表現

年別cutoffのΔMAEが独立とみなせる保証はなく（学習期間を共有するため）、11年t検定の実際の被覆率・第一種過誤率は保証されていません。**BHは個々のp値の問題を修復しません**。また、BHのFDR制御にはAE間の依存構造について仮定があります。HACの比較はあくまで感度分析です。元の平滑化`performance_metrics()`の33行を再標本化する投稿時のCIは、新しい主要解析には使用しません。

「両側95%CI下限 > 0」は調整前の探索的分類です。「q < 0.05」はIUT後の多重比較を考慮した指標です。3つのデータセットごとの補正は、3群全体でのFDRが5%であると主張するものではありません。初期全ゼロAEは有害事象に季節性がないという意味ではなく、評価に情報が不足していたことを示します。

**方法変更の履歴は論文・回答書に明記してください**。このコードは投稿時の結果を再現するための「旧コード」の代用品ではなく、査読対応の再解析版です。解析方法を2026-11-11の再解析段階で明文化したのであり、元の投稿時点で事前指定していたと表現してはいけません。

## 環境・公開時の注意

投稿時の `requirements.txt` は `numpy 1.26.4`, `pandas 2.2.2`, `statsmodels 0.14.2`, `prophet 1.3.0`, `Python 3.12.3` などでした。現コードは実行環境の各バージョンと元CSVのSHA256を`RUN_MANIFEST.json`に残します。GitHubには**JADERの生データ、OneDriveからの実データ、予測結果に含まれる機微情報や個人のローカルパス**を無条件で公開しないでください。公開範囲は元論文のデータ利用規則に従ってください。

シミュレーション用コード・過去の監査版とはファイル名も出力先も分けてあり、新旧コード混在を避けるため実行ファイルは`runpy.run_path`で同じフォルダーの共通処理を読み込みます。


## CmdStan version fingerprint 修正（2026-11-11 hotfix）

- `env_versions()` がバージョンを取得する前に `Prophet(yearly_seasonality=False)` を生成し、CmdStanバックエンドのパスを確定します。実際のAEデータでのfitは行いません。
- これにより、同一カーネルの初回と2回目で `cmdstan=None` / `(2, 37)` などに変わって再開できなくなる不具合を修正しました。
- **coreファイルの内容が変わったためSHA256指紋も変わります。** 既存出力と照合エラーになるのは意図した安全措置です。修正版はデフォルトで新しい `SciRep_GitHub_REVISED_20261111_ENVFIX` へ保存します。旧フォルダーはそのまま保持します。
- すでに旧版で長時間計算を済ませた場合は、旧フォルダーを手動コピー・指紋書き換えして再利用しないでください。互換性の検証が必要です。
- `test_cmdstan_env_fingerprint_REVISED_20261111.py` はバックエンド初期化前後の状態変化を模したテストです。実際のProphetによる全工程はお使いのWindows環境で確認してください。
