import numpy as np

def calculate_weights(y):
    # 设置高分和低分的焦点
    low_score_focus = 72
    high_score_focus = 95

    # 计算权重，使用高斯函数
    # sigma 控制曲线的宽度，sigma 值越小，曲线越窄
    sigma = 10
    weights_low = np.exp(-0.5 * ((y - low_score_focus) ** 2 / sigma ** 2))
    weights_high = np.exp(-0.5 * ((y - high_score_focus) ** 2 / sigma ** 2))
    
    # 将两个权重相加，确保低分和高分都有较高权重
    weights = weights_low + weights_high

    # 标准化权重，使得最小权重为1
    weights = weights / np.min(weights)

    return weights


def calculate_weights_custom(y):
    weights = np.zeros_like(y)
    for i, score in enumerate(y):
        if score <= 74:  
            weights[i] = 2 # 低分段權重3
        if score > 85 and score<90 :  
            weights[i] = 3  # 中高分段權重3
        if score >= 90:  
            weights[i] = 7  # 高分段權重7
        else:
            weights[i] = 1  # 中間分段權重1

    return weights


def calculate_weights_gaussian(y):
    # 設置高分和低分的焦點
    low_score_focus = 67
    high_score_focus = 99

    # 計算權重，使用高斯函數
    # 減小 sigma 值，使得高斯函數曲線更加陡峭
    sigma = 5  # 調整 sigma 參數的值
    weights_low = np.exp(-0.5 * ((y - low_score_focus) ** 2 / sigma ** 2))
    weights_high = np.exp(-0.5 * ((y - high_score_focus) ** 2 / sigma ** 2))
    
    # 將兩個權重相加，確保低分和高分都有較高權重
    weights = weights_low + weights_high

    # 標準化權重，使得最小權重為1
    weights = weights / np.min(weights)

    return weights

