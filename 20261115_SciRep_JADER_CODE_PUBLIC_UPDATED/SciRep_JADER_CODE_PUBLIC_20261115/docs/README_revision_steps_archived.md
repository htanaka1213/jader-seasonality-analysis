ARCHIVED NOTE: This document records an earlier local workflow. For paths and execution steps in this packaged repository, follow the root README.md instead.

# Scientific Reports ARIMA/SARIMA 11×12 パイロット — AIC比較診断付き V2

## なぜ改訂したか

旧版は投稿時のSARIMAX AIC探索を忠実に踏襲しますが、`enforce_stationarity=False` と `d=0/1`, `D=0/1` が混在する設定では、モデルごとに`loglikelihood_burn`（初期の尤度寄与から除かれる月数）が異なります。**このAICは無条件には比較できません。** また、`trend=None`（デフォルト）は切片なしです。特に非差分型が零平均となり、候補比較上の制約になります。

ただし、これは投稿時の**未見データの予測誤差比較が数学的に無効**という意味ではありません。SARIMA探索空間には季節次数が`(0,0,0,12)`の候補もあり、「SARIMAが必ず季節項を持つ」は一般論として正しくありません。実際の選択状況を診断してください。

## 使用するファイル・実行順序

1. `01_run_legacy_11x12_spyder.py`：投稿時と同じAIC仕様（8+64候補）で11×12予測。既存の`ARIMA_SARIMA_11x12_PILOT`の計算結果は**コード本体と環境が同じ場合に限り再利用**します。初期設定`MAX_CASES=1`。完走したら`MAX_CASES=9`。
2. `02_audit_selected_AIC_no_refit_spyder.py`：選択されたモデルの季節次数とAICの尤度計算月数を**再学習なし**で監査。旧版の計算結果にも使用可能。
3. `03_run_controlled_11x12_spyder.py`：**別のモデル仕様を用いた感度分析**。`d=1, D=0`で固定、`enforce_stationarity=True`, `enforce_invertibility=True`, `simple_differencing=False`, `trend='n'`（両方同一）。ARIMAは4候補、SARIMAは**季節ARまたはMA項が最低1つ入る12候補**。収束しない候補は採用しない。全候補の`loglikelihood_burn`が等しいことを検証。こちらも`MAX_CASES=1`で開始後、`9`に変更。
4. `04_compare_legacy_controlled_spyder.py`：投稿時仕様と感度分析仕様のΔMAE・年別t区間・改善方向の違いを、**再学習なし**で比較。
5. `test_aic_controlled.py`：実行前にF5でテスト（少量の模擬時系列で実際にSARIMAXを推定）。

`representative_AEs_9.csv`に9系列を登録。追加は**Non-vaccine「食欲減退」**です。2026-05-07投稿版Table 3の英語`Appetite loss`（5,677件）は、入力CSVのこの日本語列に対応することを報告件数から確認しました。**入力CSVに「食欲不振」という列はありません。**

## 出力先（旧版と混在させない）

- 投稿時仕様：`[local project directory]`
- 比較条件統一仕様：`[local project directory]`
- AIC監査・比較：`[local project directory]`

各モデルの年別結果は独立に保存され、中断しても再開できます。コード・入力・ライブラリの版が変わった場合は、混在を避けて停止します。

## 統計学的な位置づけ（重要）

- **比較条件統一仕様は感度分析**であって、`d=1, D=0`がすべてのAEに最適という意味ではありません。`D=1`の季節単位根、非定常性の程度、切片・ドリフトの有無は別途検討を要します。今回の`trend='n'`では差分系列にドリフトを加えません。
- `enforce_stationarity=True`のみでは、`D=1`の候補がある場合の尤度寄与期間差は解消しません。`d`と`D`を固定し、モデル選択後の実効標本数も確認してください。
- SARIMAは季節項が最低1つ、ARIMAは季節項なし、と事前に定義していますが、**季節項の有無と季節変動の有意性・生物学的原因は同じではありません**。
- 11起点は学習期間を共有するため、年単位t区間は依然として探索的です。FDR補正用の確定p値や主要分類にそのまま使用しません。
- B方針で主分類を新しいCIへ切り替える場合、**全適格AE**で一貫した仕様を適用し、Prophet/ARIMA予測値・CI・FDR・一致分類、本文・図表を更新します。実行時間（「12時間」など）は計測前には保証できません。

## 何を優先して確認するか

第一に既存8件＋食欲減退の選択季節項と`loglikelihood_burn`を監査する。第二に新しい`controlled`条件で9件を実行し、GBS、誤嚥、ワクチンADEM、食欲減退のΔMAEの符号と改善年数を対比する。**どちらの仕様を主解析に採用するかは、代表AEの結果に都合よく合わせるのではなく、モデル仕様の正当性に基づいて事前固定してください。**
