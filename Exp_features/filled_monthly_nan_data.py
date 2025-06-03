import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def fill_missing_data_seasonally(input_filepath, missing_start_year=1965, missing_end_year=1972,
                                 output_filename="monthly_CHANGBEI_weather_data_filled_seasonal.csv"):
    """
    加载月度气象数据，并使用季节性插值（按月平均值）填充指定年份范围内的缺失数据。

    参数:
        input_filepath (str): 待填充的月度气象数据CSV文件路径。
        missing_start_year (int): 缺失数据起始年份。
        missing_end_year (int): 缺失数据结束年份。
        output_filename (str): 填充后数据保存的CSV文件名。
    """
    print(f"正在加载数据文件: {input_filepath}")
    try:
        # 加载月度数据，确保 'DATE' 是索引且为 datetime 类型
        df_monthly = pd.read_csv(input_filepath, index_col='DATE', parse_dates=True, encoding='utf-8')
        print("月度数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径是否正确: {input_filepath}")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    print("\n原始月度数据信息:")
    df_monthly.info()
    print("\n原始月度数据头部:")
    print(df_monthly.head())

    # --- 识别缺失数据时间段 ---
    missing_period_start_date = pd.to_datetime(f'{missing_start_year}-01-01')
    missing_period_end_date = pd.to_datetime(f'{missing_end_year}-12-31')

    print(f"\n--- 识别并填充 {missing_start_year}-{missing_end_year} 年间的缺失数据 ---")

    # 识别需要填充的数值列（排除地理信息等通常不缺失的固定列）
    # 假设除了STATION, NAME, LATITUDE, LONGITUDE, ELEVATION，其他都是需要填充的气象指标
    numerical_cols_to_fill = df_monthly.select_dtypes(include=np.number).columns.tolist()
    # 移除地理信息等可能不需要季节性填充的固定列
    for col in ['LATITUDE', 'LONGITUDE', 'ELEVATION']:
        if col in numerical_cols_to_fill:
            numerical_cols_to_fill.remove(col)

    # 检查目标填充期内的缺失值数量
    print(f"\n{missing_start_year}-{missing_end_year} 年间的缺失值数量（填充前）:")
    print(df_monthly.loc[missing_period_start_date:missing_period_end_date, numerical_cols_to_fill].isnull().sum())

    # --- 计算季节性平均值 ---
    # 为了避免使用缺失期内可能存在的少量数据（或如果整个时期都缺失）来计算平均值，
    # 我们只使用缺失期之前和之后的数据来计算月度平均。
    df_before_missing = df_monthly.loc[:pd.to_datetime(f'{missing_start_year - 1}-12-31')]
    df_after_missing = df_monthly.loc[pd.to_datetime(f'{missing_end_year + 1}-01-01'):]

    # 合并缺失期前后的数据，用于计算每个月份的季节性平均值
    df_for_seasonal_avg = pd.concat([df_before_missing, df_after_missing])

    # 确保用于计算平均值的数据不包含 NaN，避免 NaN 传播
    monthly_averages = df_for_seasonal_avg.groupby(df_for_seasonal_avg.index.month)[numerical_cols_to_fill].mean()

    print("\n计算出的各月份平均值 (部分示例):\n", monthly_averages.head())

    # --- 执行季节性填充 ---
    df_filled_seasonal = df_monthly.copy()  # 创建副本进行填充，保留原始数据

    for col in numerical_cols_to_fill:
        initial_nans_in_period = df_filled_seasonal.loc[missing_period_start_date:missing_period_end_date,
                                 col].isnull().sum()
        if initial_nans_in_period > 0:
            for month_num in range(1, 13):
                # 找到该列在指定缺失期内，并且是当前月份的缺失值索引
                mask = (df_filled_seasonal.index.month == month_num) & \
                       (df_filled_seasonal.index >= missing_period_start_date) & \
                       (df_filled_seasonal.index <= missing_period_end_date) & \
                       (df_filled_seasonal[col].isnull())

                # 如果该月份的平均值存在，则进行填充
                if month_num in monthly_averages.index and not pd.isna(monthly_averages.loc[month_num, col]):
                    df_filled_seasonal.loc[mask, col] = monthly_averages.loc[month_num, col]
                else:
                    print(f"警告: 列 '{col}' 在 {month_num} 月没有足够的非缺失数据来计算平均值。此月份的缺失值将保留。")
            print(
                f"列 '{col}' 在 {missing_start_year}-{missing_end_year} 期间的缺失值已使用季节性平均值填充，共 {initial_nans_in_period} 个。")
        else:
            print(f"列 '{col}' 在 {missing_start_year}-{missing_end_year} 期间无缺失值。")

    print("\n季节性填充后 1965-1972 年数据缺失情况（再次检查）:")
    print(df_filled_seasonal.loc[missing_period_start_date:missing_period_end_date,
          numerical_cols_to_fill].isnull().sum())

    # --- 最终检查并处理填充后可能残余的 NaN （通常不应该有，除非计算季节平均值时就缺失） ---
    # 再次对所有数值列进行一次线性插值，以防在计算月平均时出现某种原因导致的NaN，
    # 导致某个月份的缺失未能被季节性平均值填充到。
    print("\n--- 对可能残余的 NaN 进行最终检查和处理 ---")
    for col in numerical_cols_to_fill:
        if df_filled_seasonal[col].isnull().any():
            remaining_nans = df_filled_seasonal[col].isnull().sum()
            df_filled_seasonal[col].interpolate(method='linear', limit_direction='both', inplace=True)
            df_filled_seasonal[col].fillna(method='ffill', inplace=True)
            df_filled_seasonal[col].fillna(method='bfill', inplace=True)
            if remaining_nans > 0:
                print(f"列 '{col}' 经过季节性填充后仍有 {remaining_nans} 个 NaN，已通过线性/前后填充处理。")

    print("\n--- 填充完成后的数据概览 ---")
    print("\n填充后的数据信息:\n", df_filled_seasonal.info())
    print("\n填充后的数据头部:\n", df_filled_seasonal.head())
    print("\n填充后的数据尾部:\n", df_filled_seasonal.tail())
    print("\n填充后的数据缺失值汇总:\n", df_filled_seasonal.isnull().sum())

    # --- 可视化填充结果 ---
    plt.figure(figsize=(18, 8))
    for i, col in enumerate(['PRCP', 'TEMP']):  # 只展示两个关键指标
        if col in df_monthly.columns:
            plt.subplot(2, 1, i + 1)
            plt.plot(df_monthly[col], 'b--', alpha=0.5, label=f'原始 {col}')
            plt.plot(df_filled_seasonal[col], 'g-', label=f'填充后 {col} (季节性平均)')
            plt.title(f'{col} 填充结果对比 (季节性平均)')
            plt.axvspan(missing_period_start_date, missing_period_end_date, color='red', alpha=0.2, label='缺失期')
            plt.legend()
            plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 保存填充后的数据 ---
    output_directory = os.path.dirname(input_filepath)
    output_path = os.path.join(output_directory, output_filename)
    df_filled_seasonal.to_csv(output_path, encoding='utf-8')
    print(f"\n填充后的月度数据已成功保存到: {output_path}")

    return df_filled_seasonal


# --- 设置您的文件路径和缺失年份范围 ---
input_monthly_csv_path = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\monthly_CHANGBEI_weather_data.csv"
missing_start_year = 1965
missing_end_year = 1972
output_filled_filename = "monthly_CHANGBEI_weather_data_filled_seasonal.csv"

# 运行填充函数
filled_monthly_df = fill_missing_data_seasonally(
    input_monthly_csv_path,
    missing_start_year,
    missing_end_year,
    output_filled_filename
)

if filled_monthly_df is not None:
    print("\n数据填充流程已执行完毕。")