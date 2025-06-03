import pandas as pd
import logging
from pathlib import Path

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def aggregate_to_monthly(input_cleaned_filepath, output_monthly_filename="monthly_CHANGBEI_weather_data.csv"):
    """
    将清洗后的日气象数据聚合为月度数据。

    参数:
        input_cleaned_filepath (str or Path): 清洗后的日气象数据CSV路径。
        output_monthly_filename (str): 聚合后月度数据文件名。
    返回:
        pandas.DataFrame: 聚合后的月度数据，错误时返回 None。
    """
    input_path = Path(input_cleaned_filepath)
    logging.info(f"加载数据文件: {input_path}")

    try:
        df = pd.read_csv(input_path, index_col='DATE', parse_dates=True, encoding='utf-8')
        df.sort_index(inplace=True)
        logging.info("日数据加载并排序完成。")
    except FileNotFoundError:
        logging.error(f"未找到文件: {input_path}")
        return None
    except Exception as e:
        logging.error(f"读取文件时出错: {e}")
        return None

    if df.empty:
        logging.warning("加载的数据为空，终止聚合。")
        return None

    aggregation_rules = {
        'TEMP': 'mean', 'DEWP': 'mean', 'SLP': 'mean', 'STP': 'mean',
        'VISIB': 'mean', 'WDSP': 'mean', 'MXSPD': 'max', 'GUST': 'max',
        'MAX': 'max', 'MIN': 'min', 'PRCP': 'sum', 'SNDP': 'max',
        'FRSHTT_Fog': 'sum', 'FRSHTT_Rain': 'sum', 'FRSHTT_Snow': 'sum',
        'FRSHTT_Hail': 'sum', 'FRSHTT_Thunder': 'sum', 'FRSHTT_Tornado': 'sum',
        'LATITUDE': 'first', 'LONGITUDE': 'first', 'ELEVATION': 'first',
        # 'STATION': 'first', 'NAME': 'first',
    }

    actual_rules = {col: rule for col, rule in aggregation_rules.items() if col in df.columns}

    logging.info("执行月度聚合...")
    df_monthly = df.resample('MS').agg(actual_rules)
    df_monthly.dropna(subset=['PRCP', 'TEMP'], how='all', inplace=True)
    df_monthly.reset_index(inplace=True)

    df_monthly['YEAR'] = df_monthly['DATE'].dt.year
    df_monthly['MONTH'] = df_monthly['DATE'].dt.month

    logging.info("聚合完成。样例数据:\n%s", df_monthly.head())
    logging.info("数据概况:\n%s", df_monthly.info())
    logging.info("空值计数:\n%s", df_monthly.isnull().sum())

    output_path = input_path.parent / output_monthly_filename
    df_monthly.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"已保存月度数据至: {output_path}")

    return df_monthly


# === 用法示例 ===
if __name__ == "__main__":
    input_csv = r"C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\cleaned_CHANGBEI_daily_weather_data.csv"
    output_csv = "monthly_CHANGBEI_weather_data.csv"

    result_df = aggregate_to_monthly(input_csv, output_csv)

    if result_df is not None:
        logging.info("月度聚合成功完成。")
