# 匯入必要的庫
from joblib import load
import pandas as pd
from scrawl import get_stats
import numpy as np
from sklearn.preprocessing import StandardScaler

# 載入預先訓練的模型
vote = load('./model/voting_regressor.joblib')
stack_mod = load('./model/stacking_regressor.joblib')

# 獲取 2024 年球員數據
player_stats_2024 = get_stats('2024')

# 處理缺失值
nan_columns = player_stats_2024.columns[player_stats_2024.isna().any()].tolist()
player_stats_2024[nan_columns] = player_stats_2024[nan_columns].fillna(0)
# 確保所有預期為數字的列確實為數字，並清理任何非數字條目
numeric_columns = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'year']
for column in numeric_columns:
    # 將字符串轉換為數字，移除非數字字符，轉換為浮點數
    player_stats_2024[column] = pd.to_numeric(player_stats_2024[column], errors='coerce').fillna(0)

# 定義需要對數轉換的列
columns_to_log_transform = ['FG%', '3P%', '2P%', 'eFG%', 'ORB', 'BLK']

# 進行對數轉換，處理零或負值
for col in columns_to_log_transform:
    # 檢查並處理小於等於零的值
    if any(player_stats_2024[col] <= 0):
        player_stats_2024[col] = np.log(player_stats_2024[col] + 1)  # 加1以避免log(0)和log(負數)
    else:
        player_stats_2024[col]  = np.log(player_stats_2024[col])



# 正規化用於預測的數據
scaler = StandardScaler()
player_stats_2024_scaled = StandardScaler().fit_transform(player_stats_2024.select_dtypes(include=[np.number]))

# 使用加載的模型進行預測
predictions_vote = vote.predict(player_stats_2024_scaled)
predictions_stack = stack_mod.predict(player_stats_2024_scaled)

# 混合这兩種預測的方式，这里使用之前找到的最佳混合權重
final_pred_blending = 0.42 * predictions_vote + 0.58 * predictions_stack

# 原始目標變量需要反轉換
final_pred_blending = np.expm1(final_pred_blending)

print(player_stats_2024)

# 創建 DataFrame 來存儲預測結果
predictions_2024 = pd.DataFrame({
    'Player': player_stats_2024['Player'],
    'Vote_Predicted_2K_Value_2024': predictions_vote,
    'Stack_Predicted_2K_Value_2024': predictions_stack,
    'final_pred_blending': final_pred_blending
})

# 預測結果
# 打印全部数据
print(predictions_2024.to_string())

# predictions_2024.to_csv('predictions_2024.csv', index=False)  # 可選：保存為 CSV 檔案
