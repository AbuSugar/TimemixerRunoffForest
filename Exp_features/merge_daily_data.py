import pandas as pd
import os
import glob

def merge_specific_weather_data(base_dir, specific_filename, output_filename="merged_specific_weather_data.csv"):
    """
    遍历指定目录及其子目录，只读取并合并特定名称的CSV文件，
    将它们合并为一个DataFrame并保存为新的CSV文件。

    参数:
        base_dir (str): 包含年份文件夹的根目录路径。
        specific_filename (str): 您想要合并的特定CSV文件名（例如 "58606099999_CHANGBEI INTERNATIONAL_CHINA_(115.9,28.865).csv"）。
        output_filename (str): 合并后的CSV文件名，默认为 "merged_specific_weather_data.csv"。
    """
    all_files = []
    # 精确匹配指定文件名的搜索模式
    search_pattern = os.path.join(base_dir, '*', specific_filename)
    print(f"正在搜索指定文件，模式为: {search_pattern}")

    # glob.glob 返回所有匹配的文件路径列表
    matched_files = glob.glob(search_pattern)

    if not matched_files:
        print(f"在 {base_dir} 或其子目录中未找到任何名为 '{specific_filename}' 的文件。")
        return

    # 遍历所有找到的特定CSV文件
    for f in matched_files:
        try:
            # 读取CSV文件，指定逗号为分隔符，并跳过值前的初始空格
            df = pd.read_csv(f, sep=',', skipinitialspace=True, encoding='utf-8')

            # 检查文件是否为空，或者是否只有标题行
            if df.empty or (len(df.columns) == 1 and "STATION" in df.columns[0]):
                print(f"跳过空文件或格式不正确的文件: {f}")
                continue

            all_files.append(df)
            print(f"成功加载文件: {f}")
        except Exception as e:
            print(f"加载文件 {f} 时出错: {e}")

    if not all_files:
        print("没有有效的DataFrame可以合并。")
        return

    # 拼接所有DataFrame
    merged_df = pd.concat(all_files, ignore_index=True)

    # 清理数据：根据你提供的示例，一些数值型字段可能包含999.9等特殊值，
    # 这些值通常表示缺失或无效数据。将其转换为 NaN，方便后续处理。
    numerical_cols = ["TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", "SNDP"]
    for col in numerical_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(
                merged_df[col].astype(str).str.strip().replace('999.9', pd.NA),
                errors='coerce'
            )

    # 保存合并后的DataFrame到CSV文件
    output_path = os.path.join(base_dir, output_filename)
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n所有指定文件已成功合并到: {output_path}")
    print(f"合并后的数据形状: {merged_df.shape}")

# --- 配置参数 ---
# 设置你的基础目录
base_directory = r"C:\Users\MOSS\Desktop\1942-2024年我国观测站点的逐日气象指标"
# 指定你想要合并的CSV文件名
target_filename = "58606099999_CHANGBEI INTERNATIONAL_CHINA_(115.9,28.865).csv"
# 你可以自定义输出文件名
output_file = "merged_CHANGBEI_daily_weather_data.csv"

# 运行合并函数
merge_specific_weather_data(base_directory, target_filename, output_file)