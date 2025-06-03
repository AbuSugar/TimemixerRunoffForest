import numpy as np
from sklearn.metrics import r2_score  # 导入 r2_score


def RSE(pred, true):
    # 避免除以零，如果真实值的方差为零，则返回无穷大
    denominator = np.sum((true - true.mean()) ** 2)
    if denominator == 0:
        return np.inf
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(denominator)


def CORR(pred, true):
    # 避免除以零
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # 将分母为零的项设为零以避免 NaN
    corr_per_feature = np.where(d == 0, 0, u / d)
    return corr_per_feature.mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    # 避免除以零，并处理极端值
    mape = np.abs((pred - true) / true)
    # 将真实值为零或接近零导致 MAPE 变得非常大的情况设为 0
    # 另一种常见做法是将其设为 np.nan 并从平均值中排除
    mape = np.where(np.abs(true) < 1e-8, 0, mape)  # 针对非常小的值
    mape = np.where(mape > 5, 0, mape)  # 移除异常大的值，可根据数据调整阈值
    return np.mean(mape)


def MSPE(pred, true):
    # 避免除以零
    mspe = np.square((pred - true) / true)
    mspe = np.where(np.abs(true) < 1e-8, 0, mspe)  # 针对非常小的值
    return np.mean(mspe)


# --- 新增评估指标 ---

def R2(pred, true):
    """
    计算 R2 Score。
    R2 Score 衡量模型对数据方差的解释程度。
    """
    return r2_score(true.reshape(-1), pred.reshape(-1))


def MASE(pred, true, naive_forecast=None, seasonality=1):
    """
    计算平均绝对标度误差 (MASE)。
    MASE 将预测误差与简单的季节性朴素预测进行比较，使其具有标度独立性。
    Args:
        pred (np.array): 预测值。
        true (np.array): 真实值。
        naive_forecast (np.array, optional): 朴素预测的基准。如果未提供，将使用基于 'true' 的滞后季节性朴素预测。
        seasonality (int, optional): 数据的时间序列季节性周期（例如，月度数据为 12）。默认为 1。
    Returns:
        float: MASE 值。
    """
    if naive_forecast is None:
        # 使用基于 'true' 的滞后季节性朴素预测作为基准
        # 注意：严格的 MASE 通常使用训练集上的朴素预测误差作为分母
        # 这里的实现是为了方便在评估阶段使用
        true_flat = true.reshape(-1)
        # 确保有足够的历史数据来计算朴素误差
        if len(true_flat) <= seasonality:
            return np.inf  # 如果数据太短无法计算朴素误差，返回无穷大

        errors_naive = np.abs(true_flat[seasonality:] - true_flat[:-seasonality])
        if np.sum(errors_naive) == 0:
            return np.inf  # 避免除以零，如果朴素预测是完美的

        pred_flat = pred.reshape(-1)
        return np.mean(np.abs(pred_flat - true_flat)) / np.mean(errors_naive)
    else:
        errors_naive = np.abs(true - naive_forecast)
        if np.sum(errors_naive) == 0:
            return np.inf
        return np.mean(np.abs(pred - true)) / np.mean(errors_naive)


def SMAPE(pred, true):
    """
    计算对称平均绝对百分比误差 (SMAPE)。
    SMAPE 是 MAPE 的替代品，可以处理真实值为零或接近零的情况。
    """
    # 避免除以零，将分母为零的项设为很小的数
    denominator = (np.abs(true) + np.abs(pred)) / 2
    denominator[denominator == 0] = np.finfo(float).eps

    return np.mean(np.abs(pred - true) / denominator) * 100


def metric(pred, true):
    """
    计算所有评估指标。
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    # 新增指标
    r2 = R2(pred, true)
    # MASE 需要指定季节性，例如如果数据是月度的，则 seasonality=12
    # 您可能需要根据您的数据周期调整 seasonality
    mase = MASE(pred, true, seasonality=12 if 'm' in args.freq else 1)  # 假设月度数据季节性为12
    smape = SMAPE(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, r2, mase, smape