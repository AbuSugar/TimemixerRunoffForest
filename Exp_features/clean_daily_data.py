import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def clean_daily_weather_data(input_filepath, output_filename="cleaned_CHANGBEI_daily_weather_data.csv"):
    """
    对逐日气象数据进行清洗，包括日期处理、缺失值填充、异常值处理
    以及属性列和天气现象编码列的处理。

    参数:
        input_filepath (str): 待清洗的原始日气象数据CSV文件路径。
        output_filename (str): 清洗后数据保存的CSV文件名。
    """
    print(f"正在加载数据文件: {input_filepath}")
    try:
        # 加载数据，确保日期列能够被正确解析
        # 再次确认分隔符和跳过初始空格的设置，以防数据再次出现格式问题
        df = pd.read_csv(input_filepath, sep=',', skipinitialspace=True, encoding='utf-8')
        print("数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径是否正确: {input_filepath}")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    print("\n原始数据信息:\n", df.info())
    print("\n原始数据头部:\n", df.head())

    # --- 清洗步骤 ---

    # 1. 日期时间列处理与设置索引
    print("\n--- 步骤1: 处理日期时间列并设置索引 ---")
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')  # errors='coerce' 会将无法解析的日期变为 NaT
        df.set_index('DATE', inplace=True)
        df.sort_index(inplace=True)
        # 检查是否有 NaT（Not a Time）日期，这表示有日期解析失败
        if df.index.isnull().any():
            print("警告: 存在无法解析的日期，已被转换为 NaT，请检查原始数据。")
            # 可以选择删除这些行，或进一步处理
            df.dropna(subset=[df.index.name], inplace=True)
            print(f"已删除包含无效日期的行。当前数据行数: {len(df)}")
    else:
        print("错误: 数据中缺少 'DATE' 列，无法进行时间序列处理。")
        return

    # 2. 数值型缺失值处理 (999.9 转换为 NaN 及其他 NaN 填充)
    print("\n--- 步骤2: 处理数值型缺失值 (999.9 -> NaN -> 填充) ---")
    numerical_cols = [
        "TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "MXSPD", "GUST",
        "MAX", "MIN", "PRCP", "SNDP"
    ]

    for col in numerical_cols:
        if col in df.columns:
            # 将 999.9 (字符串或数值) 替换为 NaN
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().replace('999.9', np.nan),
                errors='coerce'  # 确保无法转换的也变为 NaN
            )

    print("替换 999.9 后的缺失值数量:\n", df[numerical_cols].isnull().sum())

    # 对数值列进行线性插值填充，并处理头尾缺失值
    for col in numerical_cols:
        if df[col].isnull().any():
            initial_nans = df[col].isnull().sum()
            # 线性插值，限制连续填充7天
            df[col].interpolate(method='linear', limit_direction='both', inplace=True, limit=7)
            # 对于开头或结尾仍然存在的缺失值，使用前后填充
            if df[col].isnull().any():
                df[col].fillna(method='ffill', inplace=True)
            if df[col].isnull().any():
                df[col].fillna(method='bfill', inplace=True)
            print(f"列 '{col}' 缺失值处理: 从 {initial_nans} 减少到 {df[col].isnull().sum()}")
    print("缺失值填充完成。")

    # 3. 异常值检测与处理 (使用 IQR 方法进行截断)
    print("\n--- 步骤3: 异常值检测与处理 ---")
    # 设定一个阈值（例如 3倍 IQR），根据数据特性调整
    iqr_multiplier = 3

    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            # 识别异常值
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                print(f"列 '{col}' 发现 {len(outliers)} 个异常值 (示例: {outliers.head().tolist()}).")
                # 将异常值截断到上下界
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                print(f"列 '{col}' 的异常值已通过截断处理。")

    print("异常值处理完成。")

    # 4. 属性列 (_ATTRIBUTES) 和分类编码 (FRSHTT) 处理
    print("\n--- 步骤4: 处理属性列和天气现象编码列 ---")

    # 丢弃 _ATTRIBUTES 列，因为它们的具体含义可能需要详细文档且不直接用于预测
    cols_to_drop_attributes = [
        "TEMP_ATTRIBUTES", "DEWP_ATTRIBUTES", "SLP_ATTRIBUTES", "STP_ATTRIBUTES",
        "VISIB_ATTRIBUTES", "WDSP_ATTRIBUTES", "MAX_ATTRIBUTES", "MIN_ATTRIBUTES",
        "PRCP_ATTRIBUTES"
    ]
    for col in cols_to_drop_attributes:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"已删除属性列: {col}")

    # 处理 FRSHTT 列，拆分为多个二进制标志列
    if 'FRSHTT' in df.columns:
        # 确保 FRSHTT 是字符串类型，并用 '0' 填充到 6 位
        df['FRSHTT'] = df['FRSHTT'].astype(str).str.zfill(6)

        # 拆分并创建新列 (对应：雾, 雨, 雪, 冰雹, 雷暴, 龙卷风/漏斗云)
        df['FRSHTT_Fog'] = df['FRSHTT'].str[0].astype(int)
        df['FRSHTT_Rain'] = df['FRSHTT'].str[1].astype(int)
        df['FRSHTT_Snow'] = df['FRSHTT'].str[2].astype(int)
        df['FRSHTT_Hail'] = df['FRSHTT'].str[3].astype(int)
        df['FRSHTT_Thunder'] = df['FRSHTT'].str[4].astype(int)
        df['FRSHTT_Tornado'] = df['FRSHTT'].str[5].astype(int)

        df.drop(columns=['FRSHTT'], inplace=True)
        print("已将 'FRSHTT' 列拆分为独立的二进制标志列。")
    else:
        print("未找到 'FRSHTT' 列，跳过其处理。")

    # 5. 其他列 (STATION, LATITUDE, LONGITUDE, ELEVATION, NAME) 的处理
    # 对于单个站点数据，这些列可能保持不变。
    # 确认它们的数据类型是正确的。
    # 如果这些是固定值，可以在后续特征工程阶段决定是否删除它们以简化模型。
    # 在清洗阶段，通常保留，除非明确不需要。

    # 确保像 STATION, NAME 这类非数值列没有被错误地转换
    non_numerical_fixed_cols = ["STATION", "NAME"]
    for col in non_numerical_fixed_cols:
        if col in df.columns:
            # 如果这些列在某些情况下可能出现 NaN，可以填充为字符串 'Unknown' 或其他标记
            df[col] = df[col].fillna('Unknown').astype(str)

    # 确保 LATITUDE, LONGITUDE, ELEVATION 是数值类型
    geo_cols = ["LATITUDE", "LONGITUDE", "ELEVATION"]
    for col in geo_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 对于这些固定地理信息，如果出现 NaN，很可能是原始数据问题，
            # 可以填充为该站点的已知值，或删除这些行（如果量少）
            if df[col].isnull().any():
                print(f"警告: 列 '{col}' 存在缺失值。")
                # 简单填充为该列的众数或平均值（因为对于单个站点通常是固定值）
                if df[col].mode().empty:  # 检查众数是否存在
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)  # 用众数填充
                print(f"列 '{col}' 的缺失值已填充。")

    print("\n--- 清洗完成后的数据概览 ---")
    print("\n清洗后的数据信息:\n", df.info())
    print("\n清洗后的数据头部:\n", df.head())
    print("\n清洗后的数据尾部:\n", df.tail())
    print("\n清洗后的数据缺失值汇总:\n", df.isnull().sum())

    # 保存清洗后的数据
    output_directory = os.path.dirname(input_filepath)  # 获取输入文件所在的目录
    output_path = os.path.join(output_directory, output_filename)
    df.to_csv(output_path, encoding='utf-8')
    print(f"\n清洗后的日数据已成功保存到: {output_path}")

    return df


# --- 设置您的文件路径 ---
# 这是您合并后的数据文件的完整路径
input_csv_path = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\merged_CHANGBEI_daily_weather_data.csv"
# 清洗后输出的文件名，默认会保存在与输入文件相同的目录下
output_cleaned_filename = "cleaned_CHANGBEI_daily_weather_data.csv"

# 运行清洗函数
cleaned_df = clean_daily_weather_data(input_csv_path, output_cleaned_filename)

# 如果 cleaned_df 不是 None，表示清洗成功，可以继续后续操作
if cleaned_df is not None:
    print("\n数据清洗流程已执行完毕。")
    # 这里您可以继续进行月度聚合等操作，例如：
    # monthly_df = cleaned_df.resample('M').agg({
    #     'TEMP': 'mean',
    #     'PRCP': 'sum',
    #     'WDSP': 'mean',
    #     'MAX': 'max',
    #     'MIN': 'min',
    #     'FRSHTT_Rain': 'sum', # 统计月总降雨天数
    #     # ... 其他列的聚合方式
    # })
    # print("\n月度聚合数据示例:\n", monthly_df.head())
    # monthly_df.to_csv(os.path.join(os.path.dirname(input_csv_path), "monthly_CHANGBEI_weather_data.csv"))