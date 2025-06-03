import argparse
import os
import torch
import pandas as pd
import numpy as np
import random
import glob  # 导入 glob 模块用于文件查找

# 从你的 TimeMixer 项目中导入必要的模块
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from data_provider.data_factory import data_provider  # 导入数据提供函数
from utils.metrics import metric  # 导入评估指标函数

# --- （你的 argparse 参数定义部分，与 run.py 中完全一致） ---
parser = argparse.ArgumentParser(description='TimeMixer')
# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TimeMixer',
                    help='model name, options: [Autoformer, Transformer, TimesNet, TimeMixer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='Hydrology', help='dataset type')
parser.add_argument('--root_path', type=str,
                    default=r'C:\Users\MOSS\Desktop\TimeMixer\dataset\ganjiang_river_forecast\\',
                    help='root path of the data file')
parser.add_argument('--data_path', type=str, default='merged_monthly_hydrological_data.csv',
                    help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='Runoff',
                    help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='m',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, ms:milliseconds], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=36, help='input sequence length (e.g., 36 months)')
parser.add_argument('--label_len', type=int, default=12,
                    help='start token length for decoder (e.g., last 12 months of input)')
parser.add_argument('--pred_len', type=int, default=12,
                    help='prediction sequence length (e.g., predict next 12 months)')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--enc_in', type=int, default=None, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=None, help='decoder input size')
parser.add_argument('--c_out', type=int, default=None, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=12,
                    help='window size of moving average for monthly data')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=0,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')

# imputation task
parser.add_argument('--mask_rate', type=float, default=0.125, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--drop_last', type=bool, default=True, help='drop last')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate (e.g., type1 for step decay)')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--comment', type=str, default='ganjiang_river_forecast', help='com')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# --- （以下是你 run.py 中 if __name__ == '__main__': 的内容） ---
if __name__ == '__main__':
    # 固定随机种子 (保持不变)
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 动态计算 enc_in, dec_in, c_out (保持不变)
    try:
        temp_df = pd.read_csv(os.path.join(args.root_path, args.data_path))
        if 'DATE' in temp_df.columns:
            temp_df.rename(columns={'DATE': 'date'}, inplace=True)
        if 'STATION' in temp_df.columns:
            temp_df.drop(columns=['STATION'], inplace=True)

        feature_columns = [col for col in temp_df.columns if col != 'date']
        args.enc_in = len(feature_columns)
        args.dec_in = len(feature_columns)

        if args.features == 'S':
            args.c_out = 1
        else:
            args.c_out = len(feature_columns)

        print(f"Detected enc_in (input features): {args.enc_in}")
        print(f"Set dec_in (decoder input features): {args.dec_in}")
        print(f"Set c_out (output features): {args.c_out}")

    except Exception as e:
        print(f"Error determining data dimensions: {e}")
        print("Please ensure your merged_monthly_hydrological_data.csv is correctly formatted and accessible.")
        exit()

    print('Args in experiment:')
    print(args)

    # 选择实验类 (保持不变)
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast  # 默认值

    # --- 原始的训练和测试逻辑 ---
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                # 添加了ll for label_len
                args.task_name,
                args.model_id,
                args.comment,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print(f"Current training setting: {setting}")  # <-- 添加这一行，打印出本次训练的 setting

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)  # 训练模型

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)  # 在训练结束后立即进行测试集评估
            torch.cuda.empty_cache()
    else:
        # 如果 is_training 为 0，则仅进行测试 (保持不变)
        ii = 0
        setting = '{}_{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    # --- 新增的代码块：加载最佳模型并在验证集上评估 ---
    print('\n>>>>>>> Re-evaluating best model on Validation Set <<<<<<<<<')
    try:
        # 重新初始化 Exp 类，但这次的目的是加载模型
        eval_exp = Exp(args)  # 创建一个新的实验实例

        # 设置 is_training 为 False，以便 _get_data 获取验证集数据
        # 临时创建用于数据加载的 args 对象，不修改原 args
        temp_args_eval = argparse.Namespace(**vars(args))  # 复制 args 的所有属性
        temp_args_eval.is_training = 0

        # 获取验证集数据加载器，注意这里 flag='val'
        val_data, val_loader = data_provider(temp_args_eval, flag='val')  #

        # --------------------------------------------------------------------------
        # 关键修正：确保这里使用的 setting 与训练时实际保存的模型文件名一致
        # 最直接的方式是使用训练循环中定义的那个 `setting` 变量
        # 确保你在 train 循环外部能访问到它，或者重新生成完全一致的 setting
        # 因为它在循环中定义，所以如果你在循环外使用，需要确保它是最新一次训练的 setting
        # 简单起见，如果只跑 itr=1 次，可以直接用它。
        # 如果跑 itr > 1 次，可能需要修改一下逻辑来获取对应 itr 的 setting。
        # 这里假设只跑 itr=1，所以直接用当前 setting 变量。
        # --------------------------------------------------------------------------

        # 构建保存模型文件名的路径
        # TimeMixer 的 train 方法通常会保存为 f'{self.args.checkpoints}/{setting}_checkpoint.pth'
        # 或者 f'{self.args.checkpoints}/{setting}_best_model.pth'
        # 检查你的 exp_long_term_forecasting.py 中的 _save_checkpoint 方法确认文件名格式。
        # 常见的命名是 {setting}_checkpoint.pth

        path_to_best_model = os.path.join(args.checkpoints, setting + '_checkpoint.pth')

        # 额外的查找逻辑，以防命名略有不同，查找与 setting 匹配的最新文件
        if not os.path.exists(path_to_best_model):
            print(
                f"Warning: Specific checkpoint '{path_to_best_model}' not found. Searching for any matching checkpoint for setting: {setting}")
            # 查找所有以当前 setting 开头的 .pth 文件
            # 注意：glob.glob 路径需要匹配操作系统的斜杠风格
            search_pattern = os.path.join(args.checkpoints, setting + '*.pth')
            list_of_files = glob.glob(search_pattern)

            if list_of_files:
                path_to_best_model = max(list_of_files, key=os.path.getctime)  # 获取最新修改的文件
                print(f"Found latest checkpoint: {path_to_best_model}")
            else:
                raise FileNotFoundError(f"No checkpoint found for setting: {setting} in {args.checkpoints}")

        # 加载最佳模型权重
        eval_exp.model.load_state_dict(torch.load(path_to_best_model))
        print(f"Loaded best model from {path_to_best_model}")

        # 将模型设置为评估模式
        eval_exp.model.eval()

        preds = []
        trues = []

        with torch.no_grad():  # 在评估时不计算梯度
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(eval_exp.device)
                batch_y = batch_y.float().to(eval_exp.device)
                batch_x_mark = batch_x_mark.float().to(eval_exp.device)
                batch_y_mark = batch_y_mark.float().to(eval_exp.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -eval_exp.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, eval_exp.args.label_len - eval_exp.args.pred_len:eval_exp.args.label_len, :], dec_inp],
                    dim=1).float()

                # 前向传播
                outputs = eval_exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 如果有逆变换，进行逆变换
                if eval_exp.args.inverse:
                    outputs = val_data.inverse_transform(outputs)
                    batch_y = val_data.inverse_transform(batch_y)

                # 收集预测结果和真实值
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        # 将 preds 和 trues 从 (num_batches, batch_size, ...) 拼接成 (total_samples, ...)
        # val_loader 的 drop_last 默认为 True，所以每个 batch_size 都是固定的
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 只取真实值中对应 pred_len 的部分
        trues_for_eval = trues[:, -eval_exp.args.pred_len:, :]

        # 计算评估指标
        mae, mse, rmse, mape, mspe = metric(preds, trues_for_eval)  #
        print(f"Validation Set Evaluation:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Could not find the best model checkpoint. Please ensure training completed successfully and saved a model.")
    except Exception as e:
        print(f"An error occurred during re-evaluation: {e}")

    # 最后清空 CUDA 缓存 (保持不变)
    torch.cuda.empty_cache()