# NBA 球員表現預測模型

## 專案概覽

本專案利用機器學習技術，根據 NBA 球員的歷史統計數據來預測球員表現。模型結合了多種回歸算法，以達到精確預測，並與 NBA 球員的實際表現非常接近。

## 特色

- **數據清理**：過濾重複條目、移除不必要的欄位、處理缺失值。
- **數據轉換**：將字符串值轉換為數字，並對選定的特徵進行對數轉換以提高與目標變量的相關性。
- **預測建模**：使用 Ridge、Lasso 和 ElasticNet 回歸模型，以及 Voting 和 Stacking 等高級組合技術回歸器。
- **性能指標**：使用 RMSE 評估模型的準確性。
- **可視化**：生成熱圖和散點圖，可視化相關性和預測結果。

## 數據來源
### NBA 2K 值
利用 Python `requests` 和 `BeautifulSoup` 模組從 [Hoopshype](https://hoopshype.com/nba2k/) 網站抓取 NBA 球員的 2K 數值。
使用2023年球員數據作為模型訓練的基礎，來模擬球員在2024的遊戲評分。

### NBA 球員統計
利用 Python `requests` 和 `BeautifulSoup` 模組從 [Basketball Reference](https://www.basketball-reference.com/) 網站的每年賽季統計頁面抓取 NBA 球員的總體表現數據。例如，使用 URL `https://www.basketball-reference.com/leagues/NBA_2023_totals.html` 可獲取 2023 年的數據。

## 關鍵庫

- **Pandas** 用於數據操作和分析。
- **Seaborn** 和 **Matplotlib** 用於數據可視化。
- **Scikit-Learn** 用於實施機器學習模型。
- **NumPy** 用於數值運算。
- **Joblib** 用於保存和加載機器學習模型。

## 模型建構過程

### 數據摘要
數據集包含 369 筆條目，每個條目包含 32 個與球員表現相關的特徵。
樣本數據:
      Rk                    Player Pos Age   Tm     G    GS    MP   FG   FGA  ...  DRB   TRB   AST  STL  BLK  TOV   PF   PTS    year  2K_Value
0      1          Precious Achiuwa   C  23  TOR  55.0  12.0  20.7  3.6   7.3  ...  4.1   6.0   0.9  0.6  0.5  1.1  1.9   9.2  2023.0        76
1      2              Steven Adams   C  29  MEM  42.0  42.0  27.0  3.7   6.3  ...  6.5  11.5   2.3  0.9  1.1  1.9  2.3   8.6  2023.0        82
2      3               Bam Adebayo   C  25  MIA  75.0  75.0  34.6  8.0  14.9  ...  6.7   9.2   3.2  1.2  0.8  2.5  2.8  20.4  2023.0        87

## 相關性分析
![correlation_heatmap_of_stats_with_2k_values](https://github.com/nw7551762/2k-value-predict/assets/118497430/b4262d8d-f3c0-4457-9217-c0f2a960f557)







## 預測分析
2024 年球員預測分數與實際 2k 分數散布圖
![comparisn_od_actual_vs_predictd_2k_value](https://github.com/nw7551762/2k-value-predict/assets/118497430/3617d4ff-0659-415f-9c07-2f63bebd0c0c)

### 低分段球員預測偏高
在低分段球員的預測中，模型傾向於預測出較實際分數更高的值。這種偏差可能是因為模型在學習過程中給予高分段球員的數據更高的權重，導致模型在預測低分段球員時過度泛化其特徵。
### 高分段球員預測偏低
對於高分段球員，模型則常常預測出比實際分數低的結果。這表明在高分段的數據上模型可能已達到了一種過擬合的狀態，對於這些球員的特定特徵過度敏感。
