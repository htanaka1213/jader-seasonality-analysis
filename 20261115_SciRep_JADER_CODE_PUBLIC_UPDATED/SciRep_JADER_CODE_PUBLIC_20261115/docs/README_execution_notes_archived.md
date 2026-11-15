ARCHIVED NOTE: This document records an earlier local workflow. For paths and execution steps in this packaged repository, follow the root README.md instead.

# ARIMA/SARIMA全AE感度分析：集計ロジック修正版（2026-11-12）

## この版で修正した内容

1. **推定失敗と処理途中を分離**
   - `NOT_RUN`: チェックポイントなし（まだ計算していない）
   - `IN_PROGRESS`: チェックポイントはあるが処理中・未完了（この段階ではBHを計算しない）
   - `INCOMPLETE_OR_FAILED`: 11評価年すべて処理済みで、少なくとも1年のARIMA/SARIMAモデル推定・予測が失敗。推定不能としてBHの有効検定集合から除外
   - `UNTESTABLE_ZERO_INITIAL_48`: 初期48か月全0（既存の主解析と同じ推論対象外）
   - `UNTESTABLE_ZERO_VARIANCE`: 11個の年別ΔMAEの標準誤差が0で、t検定が定義できない
   - `INFERENCE_READY`: 11年すべての予測・t検定が成立

2. **FDRの確定条件を修正**
   各データセット×シナリオのすべてのAEが `INFERENCE_READY`、`INCOMPLETE_OR_FAILED`、`UNTESTABLE_ZERO_INITIAL_48`、`UNTESTABLE_ZERO_VARIANCE` のいずれかの確定状態となり、`NOT_RUN` と `IN_PROGRESS` がいずれも0の場合に限り、`INFERENCE_READY` のAEをBH補正する。`n_failed` と `n_zero_variance` を出力する。

3. **年別ΔMAEの標準誤差0の誤陽性を修正**
   平均ΔMAEが非0であっても、標準誤差0の場合はp=0とせず、CI/p値とも欠測として推論対象外にする。平均0で標準誤差0の場合も検定不能とする。

4. **推定失敗AEのQC記録も残す**
   `SELECTED_MODELS_AND_QC.csv` に、推定に失敗した年の `model_status`・候補ごとの非収束数なども含める。候補モデルが非収束だったことと、AE全体の推定失敗は異なる。

5. **比較不能なAEは『判定変更なし』と扱わない**
   結果比較表では、参照仕様・代替仕様のいずれかが推論不能なら `nominal_IUT_changed` を欠測にする。

## 実行

Windowsにあるご本人の6本の原本CSVを**絶対パスで直接読み込みます**。原本を上書きしません。`VERIFY_ORIGINAL_SHA256=True` でアップロード時の原本と一致することを検査します。

- `run_full_ae_sensitivity_USER_ORIGINALS_FIXED.py` をSpyderで開く。
- `RUN_MODE = "validate"` で入力の照合。
- `RUN_MODE = "pilot"` で少数AEについて試験。
- `RUN_MODE = "all"` で全件実行。
- `RUN_MODE = "collect"` で保存済みの結果を再集計。
- `GROUP_COMPLETENESS_AND_COUNTS.csv` の `group_complete` と `n_failed` / `n_in_progress` / `n_zero_variance` を必ず確認する。`group_complete=False` のq値は確定していない。

**既存のチェックポイントを再利用したい場合**：修正版を旧スクリプトと同じフォルダに置くか、結果フォルダの保存場所 `RESULTS_DIR` を従来と同じ場所に設定すること。モデル推定条件のチェックポイント・ハッシュは変更していないため、推論集計ロジックのみを再実行できる。旧版と新しい集計表は別名でバックアップすること。

## 解釈上の注意

- 推定失敗を除外する方針は主解析の取扱いとそろえたもの。失敗がAEの特徴と関連する場合、失敗除外後のBH補正だけでFDR保証が自動的に成立するわけではない。除外AEの件数・特徴・失敗理由を記載し、必要に応じ、失敗した検定をp=1として分母に含める保守的な補足計算を検討する。
- 元の主解析コードにない`llf == 0`の退化解除外と分散不正チェックは残しており、主解析の再推定値が元の値と異なる可能性がある。`REFITTED_REFERENCE_VS_SAVED_MAIN.csv` と `SELECTED_MODELS_AND_QC.csv` で検証する。原著主解析そのものとの完全な等価性を推定結果だけから仮定しない。
- 少数AEの試験が一致しても、全AEの再推定一致を保証しない。主解析の再推定を省略する場合は、既存の保存予測値の使用と比較表の参照先を別途設計し直す必要がある。この版にはその省略機能はない。
- 年間11起点の学習期間の重複に伴う時間依存性は解消されない。t区間の推定上の限界は本文に記載する。
- 本版はFDR・集計ロジックのテストを実施済み。**全1,476AEの新規感度分析を実際に完走したものではない。**

## 動作検証

- ご本人の6本のCSVについて、SHA-256、AE一覧、180か月、全期間報告件数、初期48か月、両手法推論結果の対応を確認。
- 模擬データを使い次の状態をテスト：未実行1件 → q値なし／処理中1件・未実行0件 → q値なし／推定失敗1件・ゼロ分散1件・他45件推論可能 → BH q値は45件のみ／年別ΔMAEの標準誤差0 → p値なし。模擬例の陽性数は論文の実結果ではない。
