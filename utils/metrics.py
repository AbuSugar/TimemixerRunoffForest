import numpy as np
from sklearn.metrics import r2_score


def RSE(pred, true):
    denominator = np.sum((true - true.mean()) ** 2)
    if denominator == 0:
        return np.inf
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(denominator)


# 在 utils/metrics.py 中

# ... (其他函数定义) ...

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # 避免除以零
    # 确保 d 不为 0，如果 d 为 0，则该特征的 corr 为 0
    non_zero_d = d != 0

    # 初始化一个与 u 形状相同的全零数组来存储相关性
    corr_per_feature = np.zeros_like(u, dtype=float)
    # 只在 d 不为零的特征上计算相关性
    corr_per_feature[non_zero_d] = u[non_zero_d] / d[non_zero_d]

    # 再次强调：确保返回的是一个标量浮点数
    # 如果 pred 和 true 是多变量的（例如 (N, L, F)），那么 corr_per_feature 会是 (F,)
    # .mean() 会计算这些相关性的平均值，最终应该是一个标量。
    # 加上 .item() 可以强制转换为 Python 标量，如果它是一个单元素 NumPy 数组的话。
    return corr_per_feature.mean().item() if corr_per_feature.size == 1 else corr_per_feature.mean()


# ... (metric 函数及其他函数定义) ...


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(np.abs(true) < 1e-8, 0, mape)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    mspe = np.square((pred - true) / true)
    mspe = np.where(np.abs(true) < 1e-8, 0, mspe)
    return np.mean(mspe)


def R2(pred, true):
    return r2_score(true.reshape(-1), pred.reshape(-1))


def MASE(pred, true, naive_forecast=None, seasonality=1):
    if naive_forecast is None:
        true_flat = true.reshape(-1)
        if len(true_flat) <= seasonality:
            return np.inf

        errors_naive = np.abs(true_flat[seasonality:] - true_flat[:-seasonality])
        if np.sum(errors_naive) == 0:
            return np.inf

        pred_flat = pred.reshape(-1)
        return np.mean(np.abs(pred_flat - true_flat)) / np.mean(errors_naive)
    else:
        errors_naive = np.abs(true - naive_forecast)
        if np.sum(errors_naive) == 0:
            return np.inf
        return np.mean(np.abs(pred - true)) / np.mean(errors_naive)


def SMAPE(pred, true):
    denominator = (np.abs(true) + np.abs(pred)) / 2
    denominator[denominator == 0] = np.finfo(float).eps

    return np.mean(np.abs(pred - true) / denominator) * 100


# Modified metric function to accept seasonality
def metric(pred, true, seasonality=1):  # Add seasonality parameter here
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    r2 = R2(pred, true)
    # Pass the seasonality directly to MASE
    mase = MASE(pred, true, seasonality=seasonality)
    smape = SMAPE(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, r2, mase, smape