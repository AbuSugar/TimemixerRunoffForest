import pandas as pd
import os
import numpy as np


def transform_monthly_runoff_data(input_excel_filepath, output_csv_filename="monthly_runoff_data_transformed.csv"):
    """
    加载月径流数据Excel文件，将其从宽格式（年份/月份）转换为长格式时间序列数据，
    并保存为CSV文件，格式与气象数据对齐。
    此版本明确指定只读取第3行到第67行的数据。
    专门处理Excel文件第一列是'年份'的情况，并手动指定一个站点ID。

    参数:
        input_excel_filepath (str): 待转换的月径流数据Excel文件路径。
        output_csv_filename (str): 转换后数据保存的CSV文件名。
    """
    print(f"正在加载径流数据文件: {input_excel_filepath}")
    try:
        # 正确跳过前两行，使用第3行为列名 (header=2表示0-indexed的第三行)
        # 并且只读取从header行开始的65行数据 (67 - 2 = 65行)
        df_runoff_raw = pd.read_excel(input_excel_filepath, header=2, nrows=65, engine='openpyxl')
        print("径流数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径是否正确: {input_excel_filepath}")
        return None
    except Exception as e:
        print(f"加载Excel文件时发生错误: {e}")
        return None

    print("\n原始径流数据头部（加载后）:")
    print(df_runoff_raw.head())
    print("\n原始径流数据信息:")
    df_runoff_raw.info()

    # --- 步骤1: 清理列名和重命名 ---
    print("\n--- 步骤1: 清理列名和重命名 ---")

    # 清理所有列的字符串空格并去除可能的换行符
    df_runoff_raw.columns = [str(col).strip().replace(" ", "").replace("\n", "") for col in df_runoff_raw.columns]

    # --- 关键修改部分：处理年份列和站号列 ---
    # 假设 '年份' 列是第一列，且它是实际的年份数据
    # 将 '年份' 列名标准化为 'Year'
    if '年份' in df_runoff_raw.columns:
        df_runoff_raw.rename(columns={'年份': 'Year'}, inplace=True)
        print("已将 '年份' 列重命名为 'Year'。")
    elif 'Year' not in df_runoff_raw.columns:
        print("错误: 未找到 '年份' 或 'Year' 列作为年份数据。请检查Excel文件中的年份列名。")
        return None

    # 手动添加站号列，因为原始Excel中可能没有明确的站号列
    # 如果您的径流数据也只包含昌北国际站的数据，这里可以使用气象数据的站号
    # 如果是其他站，请根据实际情况修改
    # 假设此径流文件只对应一个站，且该站的ID是 '58606099999'
    df_runoff_raw['STATION'] = '58606099999'  # <--- 请在此处手动设置正确的站号

    # --- 保持其他清理逻辑不变 ---
    # 去除“均值”列（如果存在）
    if '均值' in df_runoff_raw.columns:
        df_runoff_raw.drop(columns=['均值'], inplace=True)
        print("已删除 '均值' 列。")

    # 将 "1月"~"12月" 这些列名标准化为数字 "1"~"12"
    month_column_mapping = {}
    for i in range(1, 13):
        month_col_chinese = f"{i}月"
        if month_col_chinese in df_runoff_raw.columns:
            month_column_mapping[month_col_chinese] = str(i)  # 映射为字符串形式的数字月份

    if month_column_mapping:  # 只有当找到中文月份列时才重命名
        df_runoff_raw.rename(columns=month_column_mapping, inplace=True)
        print("已将中文月份列名转换为数字。")
    else:
        # 如果没有找到中文月份，检查是否已经是数字列名
        # 注意：这里需要确保STATION列不被误认为是月份列
        current_cols_for_check = [c for c in df_runoff_raw.columns if c not in ['Year', 'STATION']]
        current_cols_are_numbers = all(str(col).isdigit() for col in current_cols_for_check)
        if not current_cols_are_numbers:
            print("警告: 未找到 '1月'~'12月' 格式的月份列名，且现有月份列名似乎不是纯数字。")
            print("请手动检查Excel文件中的月份列名，并调整脚本中的列名转换逻辑。")
            print("当前识别的非Year/站号列名示例: ", current_cols_for_check)
            return None  # 无法确定月份列，终止执行

    # 检查是否缺失必要列（Year 和 1-12 月）
    expected_month_cols = [str(i) for i in range(1, 13)]
    if 'Year' not in df_runoff_raw.columns or not all(col in df_runoff_raw.columns for col in expected_month_cols):
        print(f"错误: 缺失 'Year' 列或部分月份列 ({expected_month_cols})。请检查列名。")
        return None

    # --- 步骤2: 转换为长格式（Melt） ---
    print("\n--- 步骤2: 转换为长格式 ---")
    # id_vars_melt 现在包含 'Year' 和我们手动添加的 'STATION'
    id_vars_melt = ['Year', 'STATION']
    value_vars_melt = expected_month_cols  # 要融化的月份列

    df_melted = df_runoff_raw.melt(id_vars=id_vars_melt,
                                   value_vars=value_vars_melt,
                                   var_name='Month',
                                   value_name='Runoff')  # 径流量的列名，可以自定义

    print("融化后的数据头部:")
    print(df_melted.head())

    # --- 步骤3: 构建完整的日期列 ---
    print("\n--- 步骤3: 构建完整的日期列 ---")

    # 类型转换
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce').astype('Int64')
    df_melted['Month'] = pd.to_numeric(df_melted['Month'], errors='coerce').astype('Int64')
    df_melted['Runoff'] = pd.to_numeric(df_melted['Runoff'], errors='coerce').astype(float)

    # 在日期构建前，去除缺失值 (Year, Month 或 Runoff 为 NaN 的行)
    df_melted.dropna(subset=['Year', 'Month', 'Runoff'], inplace=True)

    # 组合成日期字符串 'YYYY-MM-01'
    df_melted['DATE'] = df_melted['Year'].astype(str) + '-' + \
                        df_melted['Month'].astype(str).str.zfill(2) + '-01'

    # 转换为 datetime 对象
    df_melted['DATE'] = pd.to_datetime(df_melted['DATE'])

    # --- 步骤4: 最终格式调整和保存 ---
    print("\n--- 步骤4: 最终格式调整和保存 ---")
    # 选择需要的列并排序，使 'DATE' 为索引，并保持一致性
    df_transformed = df_melted[['DATE', 'STATION', 'Runoff']].copy()
    df_transformed.set_index('DATE', inplace=True)
    df_transformed.sort_index(inplace=True)

    # 处理径流数据中的 NaN (如果融化后仍有NaN，或原始Excel中是空白)
    # 对于径流数据，通常将 NaN 填充为 0，表示无径流或枯水期
    initial_runoff_nans = df_transformed['Runoff'].isnull().sum()
    if initial_runoff_nans > 0:
        print(f"径流数据中发现 {initial_runoff_nans} 个缺失值，将填充为 0。")
        df_transformed['Runoff'].fillna(0, inplace=True)
    else:
        print("径流数据中无缺失值需要填充。")

    print("\n转换并排序后的数据头部:")
    print(df_transformed.head())
    print("\n转换并排序后的数据信息:")
    df_transformed.info()
    print("\n转换后数据最终缺失值检查:")
    print(df_transformed.isnull().sum())

    # 保存到CSV
    output_directory = os.path.dirname(input_excel_filepath)
    output_path = os.path.join(output_directory, output_csv_filename)
    # encoding='utf-8-sig' 是为了处理中文CSV文件，防止在Excel中打开时出现乱码
    df_transformed.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n转换完成，已保存为: {output_path}")

    return df_transformed


# --- 配置参数 ---
input_excel_path = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\HydrologicalDataWaiZhouStations.xlsx"
output_csv_filename = "monthly_runoff_data_transformed.csv"

# 运行转换函数
transformed_runoff_df = transform_monthly_runoff_data(input_excel_path, output_csv_filename)

if transformed_runoff_df is not None:
    print("\n月径流数据转换流程已完成。")