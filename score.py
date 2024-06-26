from scrawl import fetch_nba_player_stats, fetch_player_values, get_stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import math
import sklearn.metrics as sklm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import VotingRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor



player_stats = get_stats('2023')
player_stats_2024 = get_stats('2024')
player_values = fetch_player_values(2023)

#刪除當年度季中轉隊球員重複資料
player_stats['filter_1'] = player_stats.groupby(['Player', 'year'])['Rk'].transform('count')
player_stats_2024['filter_1'] = player_stats.groupby(['Player', 'year'])['Rk'].transform('count')
#移除nba中重複的欄位名稱，如：Rk、AST、TM
player_stats=player_stats[((player_stats.Age != 'Age') & ((player_stats.filter_1==1) | (player_stats.Tm=='TOT')))]
player_stats_2024=player_stats_2024[((player_stats.Age != 'Age') & ((player_stats.filter_1==1) | (player_stats.Tm=='TOT')))]
# 檢查空值
nan_columns_list = player_stats.columns[player_stats.isna().any()].tolist()
print(nan_columns_list)

# 填充空值數據
player_stats['3P%'] = player_stats['3P%'].fillna(0)
player_stats_2024['3P%'] = player_stats_2024['3P%'].fillna(0)

# 轉換str為float
for idx in ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'year']:
    player_stats[idx]=player_stats[idx].astype(float)
    player_stats_2024[idx]=player_stats_2024[idx].astype(float)

# 篩除出賽數不足15場的球員
player_stats=player_stats[(player_stats.G>=15) ]
player_stats_2024=player_stats_2024[(player_stats.G>=15) ]
# 重設index
player_stats=player_stats.reset_index(drop=True)
player_stats_2024=player_stats_2024.reset_index(drop=True)

#比率項目加權
# for idx in ['FG', 'FGA', '3P', '3PA','FT', 'FTA']:
#     player_stats[idx+'_m'] = player_stats.groupby(['year'])[idx].transform('mean')
# player_stats['FG%_m']=player_stats['FG_m']/player_stats['FGA_m']
# player_stats['3P%_m']=player_stats['3P_m']/player_stats['3PA_m']
# player_stats['FT%_m']=player_stats['FT_m']/player_stats['FTA_m']
# player_stats['FG%_w']=(player_stats['FG%']-player_stats['FG%_m'])*player_stats['FGA']
# player_stats['3P%_w']=(player_stats['3P%']-player_stats['3P%_m'])*player_stats['3PA']
# player_stats['FT%_w']=(player_stats['FT%']-player_stats['FT%_m'])*player_stats['FTA']

print(player_stats)
print(player_values)

# merge 2k數值到資料中
player_values_df = pd.DataFrame(list(player_values.items()), columns=['Player', '2K_Value'])
player_stats = pd.merge(player_stats, player_values_df, on='Player', how='left')
player_stats = player_stats.dropna(subset=['2K_Value'])
# Ensure '2K_Value' is numeric
player_stats['2K_Value'] = pd.to_numeric(player_stats['2K_Value'], errors='coerce')

print(player_stats)

# 設定畫布大小，依據數字型欄位的數量來調整
# plt.figure(figsize=(15, 10))
# numeric_cols = player_stats.select_dtypes(include=['number']).columns
# print(numeric_cols)
# 使用循環來生成每個數字型欄位的箱形圖
# for index, col in enumerate(numeric_cols):
#     print(player_stats[col].values)
#     plt.subplot((len(player_stats) + 2) // 3, 3, index + 1)  # 調整子圖的排列
#     sns.boxplot(x=player_stats[col].values)
#     plt.title(col)
#     plt.xlabel('')

# 假設你的DataFrame叫做df，選擇數字型欄位
# numeric_cols = player_stats.select_dtypes(include=['number'])
# 使用melt函數將DataFrame轉換成長格式
# melted_data = numeric_cols.melt()
# 繪製箱形圖，每個變量對應一個箱
# plt.figure(figsize=(15, 10))  # 設定圖的大小
# sns.boxplot(data=melted_data, x='variable', y='value')
# plt.xticks(rotation=90)  # 旋轉X軸標籤以便閱讀
# plt.title('Box Plot of All Numeric Columns')
# # 設定 y 軸範圍，例如從 0 到 100
# plt.ylim(0, 120)
# plt.show()

# 為每個數字型欄位繪製圖表
# for col in numeric_cols:
#     # 檢查並處理零或負值
#     if any(player_stats[col] <= 0):
#         player_stats['log_' + col] = np.log(player_stats[col] + 1)  # 加1以避免log(0)和log(負數)
#     else:
#         player_stats['log_' + col] = np.log(player_stats[col])
    
#     # 繪製原始和轉換後的數據分佈
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     sns.histplot(player_stats[col], kde=True, ax=ax[0])
#     ax[0].set_title('Original ' + col + ' Distribution')
#     sns.histplot(player_stats['log_' + col], kde=True, ax=ax[1])
#     ax[1].set_title('Log-transformed ' + col + ' Distribution')
    
#     plt.show()

# 定義需要考慮的列
columns_to_log_transform = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA',
                       '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                       'BLK', 'TOV', 'PF', 'PTS']


# Select only numeric columns again to include '2K_Value'
player_numeric_stats = player_stats.select_dtypes(include=[np.number])
player_numeric_stats_2024 = player_stats_2024.select_dtypes(include=[np.number])
# print(numeric_stats.columns)

# 進行對數轉換，處理零或負值
for col in columns_to_log_transform:
    # 檢查並處理小於等於零的值
    if any(player_numeric_stats[col] <= 0):
        player_numeric_stats['log_'+ col] = np.log(player_numeric_stats[col] + 1)  # 加1以避免log(0)和log(負數)
    else:
        player_numeric_stats[ 'log_'+ col]  = np.log(player_numeric_stats[col])


    # 計算原始数据和對數轉換後數據 '2K_Value' 的相關係數
    correlation_ori = player_numeric_stats[col].corr(player_numeric_stats['2K_Value'])
    correlation_log = player_numeric_stats['log_'+col].corr(player_numeric_stats['2K_Value'])
    if correlation_ori<correlation_log:
        player_numeric_stats[col] = player_numeric_stats['log_'+col]
        print(f'col: {col}')
        print(f'correlation_ori: {correlation_ori}')
        print(f'correlation_log: {correlation_log}')
    player_numeric_stats.drop('log_' + col, axis=1, inplace=True)


# Compute the correlation matrix with '2K_Value'
correlation_with_2k = player_numeric_stats.corr()['2K_Value'].sort_values(ascending=False)
print(correlation_with_2k)


# Convert to DataFrame for better visualization
correlation_df = pd.DataFrame(correlation_with_2k).transpose()

# Generate a heatmap for correlations with '2K_Value'
# plt.figure(figsize=(20,5))  # Adjusting the figure size for better visibility
# sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap of NBA Player Stats with 2K Values')
# plt.show()


# 定義特徵 目標變量
X = player_numeric_stats.drop('2K_Value', axis=1)  # 特徵是除了 '2K_Value' 的其他列
y = player_numeric_stats['2K_Value']  # 目標變量

# 劃分訓練集 測試集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 打印结果确认
if len(X) > 1:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set dimensions:", x_train.shape)
    print("Testing set dimensions:", x_test.shape)
else:
    print("Not enough data to split. Consider gathering more data.")



# 正規化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
player_stats_2024_scaled = scaler.transform(player_stats_2024.select_dtypes(include=[np.number]))


# set cross-validation alpha
alpha=[0.0001,0.001,0.01,0.1,1,10,100]
# find the best alpha and build model
Ridge = RidgeCV(cv=5, alphas=alpha)
Ridge_fit = Ridge.fit(x_train_scaled, y_train)
y_ridge_train = Ridge_fit.predict(x_train_scaled)
y_ridge_test = Ridge_fit.predict(x_test_scaled)
# validation( train data and validate data)
print('RMSE_train_Ridge = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_ridge_train))))
print('RMSE_test_Ridge = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_ridge_test))))


# set cross-validation alpha
alpha=[0.0001,0.001,0.01,0.1,1,10,100]
# find the best alpha and build model
Lasso = LassoCV(cv=5, alphas=alpha)
Lasso_fit=Lasso.fit(x_train,y_train)
y_lasso_train=Lasso_fit.predict(x_train)
y_lasso_test=Lasso_fit.predict(x_test)
# validation( train data and validate data)
print('RMSE_train_Lasso = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train))))
print('RMSE_test_Lasso = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_lasso_test))))

# set cross-validation alpha and l1ratio
alpha=[0.0001,0.001,0.01,0.1,1,10,100]
l1ratio = [0.1, 0.5, 0.9, 0.95, 0.99, 1]
# find the best alpha/l1ratio and build model
elastic_cv = ElasticNetCV(cv=5, max_iter=10000000, alphas=alpha,  l1_ratio=l1ratio)
elastic_fit = elastic_cv.fit(x_train_scaled, y_train)
y_el_train=elastic_fit.predict(x_train_scaled)
y_el_test=elastic_fit.predict(x_test_scaled)
# validation( train data and validate data)
print('RMSE_train_ElasticNet = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_el_train))))
print('RMSE_test_ElasticNet = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_el_test))))



# Build Model
vote_mod = VotingRegressor([
    ('Ridge', Ridge_fit),   # Ridge 模型
    ('Lasso', Lasso_fit),   # Lasso 模型
    ('Elastic', elastic_fit)  # ElasticNet 模型
])

# Fit model
vote = vote_mod.fit(x_train_scaled, y_train.ravel())

# Predict train/test y
vote_pred_train = vote.predict(x_train_scaled)
vote_pred_test = vote.predict(x_test_scaled)

# Validation (train data and validate data)
print('RMSE_train_Voting = ' + str(math.sqrt(sklm.mean_squared_error(y_train, vote_pred_train))))
print('RMSE_test_Voting = ' + str(math.sqrt(sklm.mean_squared_error(y_test, vote_pred_test))))



# 更新回歸器參數
gbdt = GradientBoostingRegressor(learning_rate=0.07,   # 0.7
                                 max_leaf_nodes=4,     # 4
                                 n_estimators=50)      # 50
# 更新堆叠回歸器
stregr = StackingRegressor(regressors=[Ridge_fit, Lasso_fit, elastic_fit], 
                           meta_regressor=gbdt, 
                           use_features_in_secondary=True)

# 重新訓練模型
stack_mod = stregr.fit(x_train_scaled, y_train.ravel())
stacking_pred_train = stack_mod.predict(x_train_scaled)
stacking_pred_test = stack_mod.predict(x_test_scaled)

# 重新計算 RMSE
print('Adjusted RMSE_train_Stacking = ' + str(math.sqrt(sklm.mean_squared_error(y_train, stacking_pred_train))))
print('Adjusted RMSE_test_Stacking = ' + str(math.sqrt(sklm.mean_squared_error(y_test, stacking_pred_test))))


# 把第二層的Voting、Stacking兩個模型用Blending的方式結合，嘗試找出最佳混和權重
weight=list(np.linspace(0.1,1,91))
# create outcome dataframe 
train_mse=[]
test_mse=[]
for i in weight:
    blending_pred_train=(i*vote_pred_train)+((1-i)*stacking_pred_train)
    blending_pred_test=(i*vote_pred_test)+((1-i)*stacking_pred_test)
    train_mse.append(math.sqrt(sklm.mean_squared_error(y_train, blending_pred_train)))
    test_mse.append(math.sqrt(sklm.mean_squared_error(y_test, blending_pred_test)))
blending_output=pd.DataFrame({'weight':weight,
                              'train_mse':train_mse,
                              'test_mse':test_mse})
# print top 10 weight value
print (blending_output.sort_values(by=['test_mse'],ascending=True).head(10))

# # predict and blending
# # 使用VotingRegressor进行预测
# final_pred_voting_2024 = vote.predict(player_stats_2024_scaled)
# # 使用StackingRegressor进行预测
# final_pred_stacking_2024 = stack_mod.predict(player_stats_2024_scaled)
# final_pred_blending=0.94*final_pred_voting+0.06*final_pred_stacking
# # reverse log y
# final_pred_blending=np.expm1(final_pred_blending)


# # 混合这两种预测的方式，这里使用之前找到的最佳混合权重
# final_pred_blending_2024 = 0.94 * final_pred_voting_2024 + 0.06 * final_pred_stacking_2024

# # 如果原始目标变量进行了对数转换，需要逆转换
# final_pred_blending_2024 = np.expm1(final_pred_blending_2024)

# # 创建提交或检查的DataFrame
# predictions_2024 = pd.DataFrame({
#     'Player': player_stats_2024['Player'],
#     'Predicted_2K_Value_2024': final_pred_blending_2024
# })

# # 显示或保存结果
# print(predictions_2024.head())