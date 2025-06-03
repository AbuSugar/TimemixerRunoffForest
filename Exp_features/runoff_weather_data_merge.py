import pandas as pd
import os

# --- 配置您的文件路径 ---
# 清洗并填充后的月度气象数据文件路径
weather_data_path = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\monthly_CHANGBEI_weather_data_filled_seasonal.csv"
# 转换后的月度径流数据文件路径
runoff_data_path = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\monthly_runoff_data_transformed.csv"
# 合并后输出文件名
merged_output_filename = "merged_monthly_hydrological_data.csv"

# --- 步骤1: 加载数据 ---
print(f"正在加载气象数据: {weather_data_path}")
try:
    df_weather = pd.read_csv(weather_data_path, index_col='DATE', parse_dates=True, encoding='utf-8')
    print("气象数据加载成功。")
except Exception as e:
    print(f"加载气象数据时出错: {e}")
    exit()

print(f"正在加载径流数据: {runoff_data_path}")
try:
    df_runoff = pd.read_csv(runoff_data_path, index_col='DATE', parse_dates=True, encoding='utf-8')
    print("径流数据加载成功。")
except Exception as e:
    print(f"加载径流数据时出错: {e}")
    exit()

print("\n气象数据头部:\n", df_weather.head())
print("\n径流数据头部:\n", df_runoff.head())

# --- 步骤2: 合并数据 ---
# 确保在合并前，两个DataFrame的索引名一致，以便在merge时正确识别为日期
df_weather.index.name = 'DATE'
df_runoff.index.name = 'DATE'

# --- 核心修改部分 ---
# 统一为两个DataFrame添加或确认 'STATION' 列
# 假设是单站数据，我们手动添加一个站号
default_station_id = 'GANJIANG_STATION_001' # 你可以根据实际情况修改这个站号

if 'STATION' not in df_weather.columns:
    df_weather['STATION'] = default_station_id
    print(f"气象数据中已添加 STATION 列，站号为: {default_station_id}")
else:
    # 如果气象数据已有 STATION 列，确保它的值是唯一且与径流数据匹配的，
    # 或者如果它本身就是多站数据，那么这里需要更复杂的匹配逻辑
    print("气象数据已包含 'STATION' 列，请确保其站号与径流数据匹配或后续处理得当。")
    # 可以选择检查一下
    if df_weather['STATION'].nunique() > 1:
        print(f"警告：气象数据包含多个站号：{df_weather['STATION'].unique()}")
    elif df_weather['STATION'].iloc[0] != default_station_id:
        print(f"注意：气象数据的 STATION 列值为 '{df_weather['STATION'].iloc[0]}' 与预设值 '{default_station_id}' 不同。")


# 为径流数据添加 'STATION' 列
if 'STATION' not in df_runoff.columns:
    df_runoff['STATION'] = default_station_id
    print(f"径流数据中已添加 STATION 列，站号为: {default_station_id}")
else:
    print("径流数据已包含 'STATION' 列，继续使用现有列。")
    # 同样可以进行检查
    if df_runoff['STATION'].nunique() > 1:
        print(f"警告：径流数据包含多个站号：{df_runoff['STATION'].unique()}")
    elif df_runoff['STATION'].iloc[0] != default_station_id:
        print(f"注意：径流数据的 STATION 列值为 '{df_runoff['STATION'].iloc[0]}' 与预设值 '{default_station_id}' 不同。")


# 合并数据：基于日期和站号进行外连接
# 注意：这里我们明确指定了要从 df_runoff 中选择 'STATION' 和 'Runoff' 列进行合并
# 这样即使 df_runoff 有其他列，也不会意外地被合并进来。
merged_df = pd.merge(df_weather, df_runoff[['STATION', 'Runoff']], on=['DATE', 'STATION'], how='outer', suffixes=('_weather', '_runoff'))


# 排序以确保时间序列正确
merged_df.sort_values(by=['DATE', 'STATION'], inplace=True) # 排序时也要考虑站号

print("\n合并后的数据头部:\n", merged_df.head())
print("\n合并后的数据尾部:\n", merged_df.tail())
print("\n合并后的数据信息:\n", merged_df.info())
print("\n合并后的数据缺失值汇总:\n", merged_df.isnull().sum())

# --- 步骤3: 缺失值处理 (合并后可能出现新的缺失，例如某个时间点只有气象数据没有径流数据) ---
# 对于径流 Runoff 列的缺失，通常填充为 0
if 'Runoff' in merged_df.columns and merged_df['Runoff'].isnull().any():
    print(f"Runoff 列有 {merged_df['Runoff'].isnull().sum()} 个缺失值，填充为 0。")
    merged_df['Runoff'].fillna(0, inplace=True)

# 对于合并后气象数据中可能存在的少数缺失（如果 outer merge 导致），可以再次填充
# 例如，使用线性插值
for col in merged_df.columns:
    # 站号不插值，且只对数值型列进行插值
    if merged_df[col].isnull().any() and col != 'STATION' and pd.api.types.is_numeric_dtype(merged_df[col]):
        initial_nans = merged_df[col].isnull().sum()
        # 尝试线性插值
        merged_df[col].interpolate(method='linear', limit_direction='both', inplace=True)
        # 如果插值后仍有NaN（例如序列开头或结尾的NaN），使用前后填充
        merged_df[col].fillna(method='ffill', inplace=True)
        merged_df[col].fillna(method='bfill', inplace=True)
        if initial_nans > 0 and merged_df[col].isnull().sum() == 0: # 确认确实填充了
            print(f"列 '{col}' 填充了 {initial_nans} 个缺失值。")
        elif merged_df[col].isnull().sum() > 0:
            print(f"警告：列 '{col}' 填充后仍有 {merged_df[col].isnull().sum()} 个缺失值，请检查数据。")
    elif merged_df[col].isnull().any() and col != 'STATION' and pd.api.types.is_string_dtype(merged_df[col]):
        merged_df[col].fillna('Unknown', inplace=True)
        print(f"列 '{col}' 填充了字符串缺失值。")


# 保存合并后的数据
output_directory = os.path.dirname(weather_data_path)
merged_output_path = os.path.join(output_directory, merged_output_filename)
merged_df.to_csv(merged_output_path, encoding='utf-8-sig', index=True) # 确保日期索引也被保存
print(f"\n合并后的月度水文气象数据已保存到: {merged_output_path}")