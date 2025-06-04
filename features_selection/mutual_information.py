import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def calculate_mutual_information(X, y, discrete_features='auto', random_state=0):
    """
    计算 X 中每个特征与目标 y 之间的互信息。
    会处理 X 为空的情况。
    """
    if X.empty:
        return pd.Series(dtype=float)

    # 确保 y 是1维数组
    if hasattr(y, 'ravel'):
        y_1d = y.ravel()
    else:  # 如果 y 已经是1维numpy数组或列表
        y_1d = y

    mi = mutual_info_regression(X, y_1d, discrete_features=discrete_features, random_state=random_state)
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series


def mrmr_feature_selection(features_df, target_series, num_features_to_select):
    """
    使用 mRMR (最小冗余最大相关性) 原则选择特征。

    参数:
        features_df (pd.DataFrame): 包含特征列的 DataFrame。
        target_series (pd.Series): 包含目标变量的 Series。
        num_features_to_select (int): 要选择的特征数量。

    返回:
        list: 包含所选特征名称的列表。
    """
    if features_df.empty:
        print("警告: 特征 DataFrame 为空。无法选择特征。")
        return []
    if num_features_to_select <= 0:
        return []
    if num_features_to_select > len(features_df.columns):
        print(
            f"警告: num_features_to_select ({num_features_to_select}) 大于可用特征数 ({len(features_df.columns)})。将选择所有可用特征。")
        num_features_to_select = len(features_df.columns)

    remaining_features = features_df.columns.tolist()
    selected_features = []

    # 1. 计算所有特征与目标变量之间的互信息
    print("正在计算特征与目标之间的互信息...")
    feature_target_mi = calculate_mutual_information(features_df, target_series)

    # 处理互信息可能为 NaN 或负数的情况 (scikit-learn的MI应为非负)
    feature_target_mi = feature_target_mi.fillna(0)
    feature_target_mi[feature_target_mi < 0] = 0

    # 2. 选择第一个特征：与目标MI最高的特征
    if not feature_target_mi.empty and feature_target_mi.sum() > 0:  # 检查是否有任何非零的MI值
        first_feature = feature_target_mi.idxmax()
        selected_features.append(first_feature)
        if first_feature in remaining_features:
            remaining_features.remove(first_feature)
        else:  # 正常情况下不应发生
            print(f"警告: 第一个特征 {first_feature} 在剩余特征中未找到。")
            if remaining_features:  # 后备方案
                selected_features.append(remaining_features.pop(0))
            else:
                print("错误: 初始互信息计算后没有可用特征进行选择。")
                return []
    elif remaining_features:  # 如果所有MI都为零，选择第一个可用特征作为起点
        print("警告: 所有特征与目标的互信息值均为零。将选择第一个可用的特征。")
        first_feature = remaining_features.pop(0)
        selected_features.append(first_feature)
    else:
        print("错误: 特征-目标互信息计算未产生结果或无可用特征。无法选择特征。")
        return []

    # 3. 迭代选择其余特征
    for i in range(1, num_features_to_select):
        if not remaining_features:
            print(f"已选择 {len(selected_features)} 个特征。没有更多特征可供选择。")
            break

        best_next_feature = None
        max_mrmr_score = -np.inf

        for feature_k in remaining_features:
            # 相关性项: I(Fk; Y)
            relevance = feature_target_mi.get(feature_k, 0)

            # 冗余项: mean(I(Fk; Fj)) for Fj in selected_features
            redundancy_sum = 0
            if selected_features:
                for feature_j in selected_features:
                    # 确保 feature_k 和 feature_j 的数据有效
                    if features_df[[feature_k]].isnull().all().all() or features_df[feature_j].isnull().all():
                        mi_feature_feature = 0  # 如果一个特征全是NaN，则无法计算MI
                    else:
                        mi_feature_feature = mutual_info_regression(
                            features_df[[feature_k]], features_df[feature_j].ravel(),  # 确保y是1维
                            discrete_features='auto', random_state=0
                        )[0]
                    redundancy_sum += mi_feature_feature

                redundancy_avg = redundancy_sum / len(selected_features)
            else:
                redundancy_avg = 0

            mrmr_score = relevance - redundancy_avg

            if mrmr_score > max_mrmr_score:
                max_mrmr_score = mrmr_score
                best_next_feature = feature_k

        if best_next_feature:
            selected_features.append(best_next_feature)
            if best_next_feature in remaining_features:  # 确保它还在
                remaining_features.remove(best_next_feature)
        else:
            # 未找到合适的特征 (例如, 所有剩余特征的 mRMR 分数均为非正数，或已无剩余特征)
            print(f"提前停止: 没有更多特征能提高 mRMR 分数或已无剩余特征。已选择 {len(selected_features)} 个特征。")
            break

    print(f"\n使用 mRMR 选择了 {len(selected_features)} 个特征。")
    return selected_features


if __name__ == "__main__":
    # --- 配置 ---
    file_path = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\merged_monthly_hydrological_data.csv"
    target_variable_name = "Runoff"
    # 需要从特征中排除的列 (标识符等), 会进行不区分大小写的检查
    identifier_cols_to_exclude = ["DATE", "STATION", "Date", "Station_ID", "time", "YEAR", "MONTH", "Year", "Month"]
    num_features_to_select = 10  # 根据需要调整

    # --- 1. 加载数据 ---
    print(f"正在从以下路径加载数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("数据加载成功。")
        print(f"原始 DataFrame 形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"🚨 错误: 文件未在路径 {file_path} 找到")
        exit()
    except Exception as e:
        print(f"🚨 加载 CSV 文件时出错: {e}")
        exit()

    # --- 2. 初始数据检查与目标变量处理 ---
    if target_variable_name not in df.columns:
        print(f"🚨 错误: 目标列 '{target_variable_name}' 在 CSV 文件中未找到。")
        print(f"可用的列有: {df.columns.tolist()}")
        exit()

    df[target_variable_name] = pd.to_numeric(df[target_variable_name], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=[target_variable_name], inplace=True)
    if len(df) < initial_rows:
        print(f"因目标变量 '{target_variable_name}' 中存在 NaN 值，已删除 {initial_rows - len(df)} 行。")

    if df.empty:
        print(f"🚨 错误: 处理目标变量 '{target_variable_name}' 中的 NaN 后无剩余数据。")
        exit()

    Y = df[target_variable_name]

    # --- 3. 特征准备 ---
    # 初步选择所有列作为特征，然后进行筛选
    # 对排除列表进行小写处理，以便进行不区分大小写的比较
    identifier_cols_to_exclude_lower = [id_col.lower() for id_col in identifier_cols_to_exclude]
    potential_feature_cols = [col for col in df.columns if
                              col.lower() not in identifier_cols_to_exclude_lower and col != target_variable_name]

    X = pd.DataFrame()  # 初始化空的X DataFrame
    if not potential_feature_cols:
        print("🚨 错误: 排除标识符和目标列后，未识别到潜在的特征列。")
        exit()

    print(f"\n已识别的潜在特征列: {potential_feature_cols}")

    for col in potential_feature_cols:
        if col in df.columns:  # 确保列存在
            # 尝试转换为数值型。如果某列确实是分类文本，它会变成NaN并被处理，
            # 或者您可能需要特定的编码。
            X[col] = pd.to_numeric(df[col], errors='coerce')
        else:  # 这不应该发生，因为 potential_feature_cols 来自 df.columns
            print(f"警告: 潜在特征列列表中的列 '{col}' 在 DataFrame 中未找到。")

    # 处理特征中的 NaN 值
    # 选项 1: 删除包含过多 NaN 值的列 (例如, >50%)
    # 计算阈值前检查X是否为空
    if not X.empty:
        threshold = 0.5 * len(X)
        X.dropna(axis=1, thresh=int(threshold), inplace=True)  # thresh 需要整数
        print(f"删除超过50% NaN的列后剩余特征: {X.columns.tolist()}")
    else:
        print("警告: 特征 DataFrame X 在尝试转换数值类型后为空。")

    # 选项 2: 填充剩余的 NaN 值 (例如, 使用均值或中位数)
    if not X.empty and X.isnull().any().any():
        print("警告: 特征中发现 NaN 值。将使用列均值进行填充。")
        for col in X.columns[X.isnull().any()]:  # 仅对仍有NaN的列进行插补
            X[col] = X[col].fillna(X[col].mean())

    # 如果插补后特征中仍有 NaN 值，则删除这些行 (例如，如果整列都是 NaN，则均值也是 NaN)
    if not X.empty:
        initial_rows_X = len(X)
        X.dropna(axis=0, how='any', inplace=True)
        if len(X) < initial_rows_X:
            print(f"因特征插补后仍存在 NaN 值，已删除 {initial_rows_X - len(X)} 行。")

    # 在 X 中删除行后，对齐 Y
    if not X.empty:
        Y = Y[X.index]
    else:  # 如果X变为空，Y也应该变为空或给出错误
        Y = pd.Series(dtype=Y.dtype)  # 使Y为空但保持类型

    if X.empty or Y.empty:
        print("🚨 错误: 特征预处理后无剩余数据。无法执行特征选择。")
        exit()

    print(f"\n最终用于选择的特征 ({X.shape[1]}): {X.columns.tolist()}")
    print(f"X 的形状: {X.shape}, Y 的形状: {Y.shape}")

    # --- 4. 特征选择 ---
    print(f"\n正尝试使用 mRMR 选择 {num_features_to_select} 个特征...")
    selected_features_mrmr = mrmr_feature_selection(X.copy(), Y.copy(), num_features_to_select)  # 传递副本

    # --- 5. 输出结果 ---
    print("\n✅ --- mRMR 选择的特征 ---")
    if selected_features_mrmr:
        for i, feature in enumerate(selected_features_mrmr):
            print(f"{i + 1}. {feature}")
    else:
        print("mRMR 未选择任何特征。")

    # 为了比较，显示按与目标变量的互信息排序的特征
    print("\n📊 --- 按与目标变量的互信息排序的特征 (供比较) ---")
    if not X.empty:
        mi_with_target = calculate_mutual_information(X.copy(), Y.copy())  # 传递副本
        if not mi_with_target.empty:
            mi_with_target_sorted = mi_with_target.sort_values(ascending=False)
            print(mi_with_target_sorted)
        else:
            print("与目标变量的互信息计算返回空结果。")
    else:
        print("没有可用于计算与目标变量互信息的特征。")