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

截至 2024 年 5 月 6 號實際 2K 分數 90 分以上球員預測數據:
                  Player  Actual_2K_Value  Predicted_2K_Value  Difference   
            Nikola Jokic               98           96.188489   -1.811511
             Joel Embiid               96           96.640695    0.640695
   Giannis Antetokounmpo               96           96.365043    0.365043
            LeBron James               96           95.568045   -0.431955
            Kevin Durant               96           92.580398   -3.419602
             Luka Doncic               95           97.535999    2.535999
            Jimmy Butler               95           89.804507   -5.195493
            Jayson Tatum               95           92.303765   -2.696235
          Damian Lillard               94           89.432404   -4.567596
           Kawhi Leonard               94           91.490098   -2.509902
            Devin Booker               94           91.235675   -2.764325
           Anthony Davis               93           93.849111    0.849111
 Shai Gilgeous-Alexander               93           94.130990    1.130990
        Donovan Mitchell               92           91.086434   -0.913566
            Kyrie Irving               90           90.765717    0.765717
         Zion Williamson               90           88.540953   -1.459047

            
### 低分段球員預測偏高
在低分段球員的預測中，模型傾向於預測出較實際分數更高的值。這種偏差可能是因為模型在學習過程中給予高分段球員的數據更高的權重，導致模型在預測低分段球員時過度泛化其特徵。
### 高分段球員預測偏低
對於高分段球員，模型則常常預測出比實際分數低的結果。這表明在高分段的數據上模型可能已達到了一種過擬合的狀態，對於這些球員的特定特徵過度敏感。


## 可能的改善方法
### 數據增強
為了解決模型在低分和高分球員預測上的偏差，可以考慮增加數據集的多樣性和量，特別是在低分和高分的球員數據上。透過增加這些分段的訓練樣本，可以幫助模型更好地學習這些群體的特點。
### 特徵工程
重新審視特徵工程過程，尋找能更好反映球員表現的特徵。例如，可以加入球員的效率指標、賽季進步率等數據作為新特徵，這可能有助於模型捕捉到更深層次的表現趨勢。
### 調整模型結構
考慮使用不同的機器學習模型或優化現有模型的參數。例如，可以實驗深度學習方法或更複雜的非線性模型，看是否可以改善預測的準確性。此外，調整模型的正則化參數或損失函數也可能有所幫助。
### 優化損失函數
針對觀察到的偏差問題，可以設計或修改損失函數，使其對預測低分或高分球員時的誤差給予更大的懲罰。這種方法可以讓模型在訓練過程中更加關注這些範圍的預測準確性。
