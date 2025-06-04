import os
import pandas as pd
from datetime import datetime

def log_experiment_results(args, metrics_dict, file_name='experiment_results.csv'):
    """
    记录实验的参数和输出指标到一个CSV文件中。

    Args:
        args: 一个包含所有实验参数的对象（例如，argparse.Namespace 对象）。
              需要能够通过 args.parameter_name 访问参数。
        metrics_dict: 一个字典，包含评估指标，例如：
                      {'mse': 0.7395, 'mae': 0.5199, 'rmse': 0.8599, ...}
        file_name: 要保存结果的CSV文件名，默认为 'experiment_results.csv'。
    """
    # 确定文件保存路径（项目主目录）
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(project_root, file_name)

    # 提取需要的参数
    # 这里列出了你之前提到的主要参数，如果还有其他重要参数需要记录，请自行添加
    params_to_log = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'task_name': getattr(args, 'task_name', ''),
        'model_id': getattr(args, 'model_id', ''),
        'model': getattr(args, 'model', ''),
        'seq_len': getattr(args, 'seq_len', ''),
        'label_len': getattr(args, 'label_len', ''),
        'pred_len': getattr(args, 'pred_len', ''),
        'e_layers': getattr(args, 'e_layers', ''),
        'd_layers': getattr(args, 'd_layers', ''),
        'enc_in': getattr(args, 'enc_in', ''),
        'dec_in': getattr(args, 'dec_in', ''),
        'c_out': getattr(args, 'c_out', ''),
        'batch_size': getattr(args, 'batch_size', ''),
        'num_workers': getattr(args, 'num_workers', ''),
        'moving_avg': getattr(args, 'moving_avg', ''),
        'down_sampling_layers': getattr(args, 'down_sampling_layers', ''),
        'down_sampling_window': getattr(args, 'down_sampling_window', ''),
        'features': getattr(args, 'features', ''),
        'target': getattr(args, 'target', ''),
        'freq': getattr(args, 'freq', ''),
        'channel_independence': getattr(args, 'channel_independence', ''),
        'decomp_method': getattr(args, 'decomp_method', ''),
        'use_norm': getattr(args, 'use_norm', ''),
        'comment': getattr(args, 'comment', ''),
        'learning_rate': getattr(args, 'learning_rate', ''),
        'patience': getattr(args, 'patience', ''),
        'train_epochs': getattr(args, 'train_epochs', ''),
        # 如果有其他你觉得重要的参数，请在这里添加
    }

    # 合并参数和指标
    # 注意：metrics_dict 应该包含 mse, mae, rmse, mape, mspe, rse, corr, r2, mase, smape
    record_data = {**params_to_log, **metrics_dict}

    # 将字典转换为DataFrame的一行
    df_new_record = pd.DataFrame([record_data])

    # 检查文件是否存在，如果存在则追加，否则创建新文件
    if os.path.exists(log_file_path):
        # 如果文件存在，读取现有数据，然后追加
        df_existing = pd.read_csv(log_file_path)
        # 确保新记录的列顺序与现有文件一致，防止错位
        df_new_record = df_new_record[df_existing.columns.tolist()]
        df_combined = pd.concat([df_existing, df_new_record], ignore_index=True)
        df_combined.to_csv(log_file_path, index=False)
    else:
        # 如果文件不存在，直接写入（包含表头）
        df_new_record.to_csv(log_file_path, index=False)

    print(f"实验结果已记录到: {log_file_path}")

if __name__ == '__main__':
    # 这是一个示例用法，你需要在你的 runTest.py 或 exp_long_term_forecasting.py 中调用它

    # 模拟一个 args 对象 (实际运行时会是 argparse.Namespace 对象)
    class MockArgs:
        def __init__(self):
            self.task_name = 'long_term_forecast'
            self.model_id = 'ganjiang_river_runoff_forecast_test'
            self.model = 'TimeMixer'
            self.seq_len = 96
            self.label_len = 24
            self.pred_len = 12
            self.e_layers = 3
            self.d_layers = 2
            self.enc_in = 24
            self.dec_in = 24
            self.c_out = 1
            self.batch_size = 16
            self.num_workers = 4
            self.moving_avg = 12
            self.down_sampling_layers = 3
            self.down_sampling_window = 2
            self.features = 'MS'
            self.target = 'Runoff'
            self.freq = 'm'
            self.channel_independence = 0
            self.decomp_method = 'moving_avg'
            self.use_norm = 1
            self.comment = 'Predicting_Runoff_with_Weather_Data_Test'
            self.learning_rate = 0.0002
            self.patience = 10
            self.train_epochs = 30

    mock_args = MockArgs()

    # 模拟评估指标
    mock_metrics = {
        'mse': 0.7395,
        'mae': 0.5199,
        'rmse': 0.8599,
        'mape': 0.5728,
        'mspe': 10219.4961,
        'rse': 0.4781,
        'corr': 0.8449,
        'r2': 0.7714,
        'mase': 0.3155,
        'smape': 80.0558
    }

    print("--- 运行示例 ---")
    log_experiment_results(mock_args, mock_metrics)
    print("--- 示例结束 ---")

    # 再次运行以模拟追加记录
    mock_args.seq_len = 48
    mock_metrics['mse'] = 0.65
    mock_metrics['mae'] = 0.48
    print("\n--- 再次运行示例（模拟不同参数和结果）---")
    log_experiment_results(mock_args, mock_metrics)
    print("--- 再次示例结束 ---")