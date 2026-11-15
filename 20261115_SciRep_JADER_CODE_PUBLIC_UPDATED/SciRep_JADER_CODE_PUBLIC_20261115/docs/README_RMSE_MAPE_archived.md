ARCHIVED NOTE: This document records an earlier local workflow. For paths and execution steps in this packaged repository, follow the root README.md instead.

# Scientific Reports：RMSE・MAPEの追加計算（主解析変更なし）

## 目的
2026年5月の再投稿原稿ではMAEが主要評価指標、RMSE・MAPEが副次指標でした。新しい2009〜2019年の11起点×12か月の**保存済み月次予測値**から、RMSE・MAPEおよびΔ値を追加集計します。モデルの再推定やMAE/IUT/BH-FDRの変更は一切しません。全AEの差分次数感度分析とは独立です。

## 実行方法（Spyder）
1. `calculate_RMSE_MAPE_revised_main.py`を開きます。
2. `PRIMARY_RESULT_SOURCE`に、ご自身のPCにある主解析の予測結果アーカイブ `SciRep_GitHub_REVISED_20261111_ENVFIX.zip` の絶対パスを指定します（または展開フォルダのパスを指定します）。同じ場所にZIPとコードを置けば空欄のまま自動検出します。
3. F5で実行します。`[COMPLETE]`の表示を確認します。
4. スクリプトのあるフォルダの `RMSE_MAPE_secondary_results` にCSVと監査用マニフェストを保存します。**元ZIP、元CSV、元のPythonの結果は変更しません。**

例：`PRIMARY_RESULT_SOURCE = r"[local user directory]\...\SciRep_GitHub_REVISED_20261111_ENVFIX.zip"`

`INFERENCE_BY_METHOD_*.csv`の6本だけでは予測値が入っていないためRMSE/MAPEの計算はできません。必ず個々の月次予測値を含むZIPまたは展開フォルダが必要です。

## 指標の定義
各年の12か月予測で、
- RMSE = sqrt(mean((実測−予測)^2))
- MAPE(%) = 100 × mean(|実測−予測| / 実測)
- ΔRMSE = RMSE（季節なし）−RMSE（季節あり）
- ΔMAPE = MAPE（季節なし）−MAPE（季節あり）

**2009〜2019年の11個の年別RMSEを平均**します。これは旧ARIMA/SARIMAのfold別RMSEの平均と対応する集計です。比較可能性のため、Prophetにも同じ年別集計を用います。参考用に132か月をひとまとめにしたpooled RMSEも別の列に保存します。この2つは数学的に同一ではありません。

旧ARIMA/SARIMAコードと同様、**その年の実測値に0件が1か月でもあれば、その年のMAPEを欠測**として扱います。両モデルのMAPEは同じ有効年だけで比較します。11年全てが除外された場合、MAPEとΔMAPEは空欄になります。MAPEの有効年数は必ず併記してください。MAPEの0件を仮に1件と置いたり、0件の月だけ除いたりしません。

**重要な旧表の単位差：** 旧Prophetコードでは`performance_metrics`由来のMAPEを0〜1程度の「比率」で表示する一方、旧ARIMA/SARIMAコードでは100倍した「百分率」で表示していました。本コードは両方を別名で出力します。論文の新しい表では両手法とも**MAPE (%)**に統一することを推奨します。旧再投稿時と数値そのものは異なります（CV評価期間が異なるため）。

## 保存するファイル
- `RMSE_MAPE_SECONDARY_ALL_AEs.csv`：1476 AE×データセット×2手法＝2952行、RMSE・MAPE・Δ・95%CI（副次的）・MAPE使用年数・MAE照合用数値。
- `RMSE_MAPE_YEARLY_DETAILS.csv`：各年の予測誤差（32,472行）、実測0件月数。
- `RMSE_MAPE_DATASET_OVERVIEW.csv`：各群のMAPE計算可能数・使用年数。
- `RMSE_MAPE_FOCUS_9.csv`：重点9組のAE。
- `RMSE_MAPE_RUN_MANIFEST.json`：実行環境と入力の記録。

## 95%CIの扱い
再投稿時はブートストラップCIでしたが、旧方法はReviewer 2から推論の妥当性を指摘されています。本コードではRMSEとMAPEの年別Δ値から、**参考用の年別t区間**のみを追加出力します。これらの区間を用いた陽性判定や、RMSE/MAPEの追加IUT・FDR判定は**行いません**。実際に論文にCIを載せるかは、元論文の副次指標の提示範囲と査読対応方針に応じて判断します。

## 注意
元の`INFERENCE_BY_METHOD`のMAEと、再計算したMAEが一致するかも検証します。ただしこれらはQC用であり、元のMAE推論結果を変更しません。途中までの推論ZIPを指定した場合はファイル数検証で停止し、不完全な結果を論文用に保存しません。
