from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):

    # super() 是一个内置函数，用于调用父类的方法。
    # Exp_Short_Term_Forecast 是当前类。
    # self 是当前类的实例。
    # __init__(args) 调用父类 Exp_Basic 的构造函数，并将 args 参数传递给它。
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self): # 构建模型， 可以理解为初始化模型的参数， 像是参数获取， 模型选择， 序列长度的选择

        if self.args.data == 'm4': # 如果实例中data == m4时候， 针对m4数据集特殊处理
            # m4是一个经典的时间序列预测基准数据集， 包含不同频率（年、季、月、周、日、小时）的数据
            # 每种时间序列都有一个固定的预测长度（例如：年=6步，月=18步）

            # self.args.seasonal_patterns代表数据的频率， 例如：年、季、月、周、日、小时
            # self.args.pred_len代表模型输出序列的长度， 与m4一致
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            # 输入序列长度，这里设为 2 倍预测长度，是一种经验性设定
            # 输入序列长度表示模型能看到的历史时间的步数，
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len

            # 将标签序列的长度设置为预测序列的长度， 对齐输入输出
            # 在时间序列预测中，label_len 通常表示解码器的输入标签长度，而 pred_len 表示模型需要预测的未来时间步数
            # label_len指的是在计算损失函数时候， 模型使用的真实标签的长度为label_len的部分
            # 时间轴 →
            # |---seq_len=48---|---label_len=24---|---pred_len=24---|
            #                  ↑ decoder输入       ↑ 预测目标
            self.args.label_len = self.args.pred_len

            # 从 M4Meta.frequency_map 中获取与当前时间序列的季节性模式（self.args.seasonal_patterns）对应的频率值，并赋值给 self.args.frequency_map。
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]

        # 根据 self.args.model 指定的模型名称，从 self.model_dict 中获取对应的模型类，并实例化该模型。
        # 将模型的参数转换为 float 类型（32 位浮点数）。
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 是否启用了多gpu
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 捞点数据
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器
    def _select_optimizer(self):
        # 短期预测的优化器使用Adam优化器，学习率为args.learning_rate
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        # 选择损失函数
        # 这里使用了均方误差（MSE）作为默认损失函数
        if loss_name == 'MSE':
            return nn.MSELoss()

        # 这里使用了均方根误差（RMSE）作为损失函数
        elif loss_name == 'MAPE':
            return mape_loss()

        # 这里使用了平均绝对误差（MAE）作为损失函数
        elif loss_name == 'MASE':
            return mase_loss()

        # 这里使用了对称平均绝对百分比误差（SMAPE）作为损失函数
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        """ 1. 准备训练/验证数据
            2. 构建保存目录
            3. 初始化训练控制组件（early stopping / optimizer / scheduler / criterion）
            4. 开始训练循环（多个 epoch）
               ├── 遍历数据迭代器
               │   ├── 前向传播
               │   ├── 损失计算
               │   ├── 反向传播
               │   └── 参数更新
               ├── 验证模型
               ├── early stopping 判断
               └── 学习率更新
            5. 加载最佳模型返回
        """
        # 短期预测训练流程
        # 调用get_data方法获取训练和验证数据集
        # train_loader和vali_loader是数据加载器（DataLoader），用于按批次（batch）加载数据，提高训练效率
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # 检查点目录用于保存训练过程中生成的模型权重文件。
        # 如果目录不存在，创建目录。
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # time_now记录当前时间，用于后续计算训练速度
        time_now = time.time()

        # train_loader 是 PyTorch 的 DataLoader
        # len(train_loader) 返回一个 epoch 中 batch 数量
        # 用于指导scheduler知道一个epoch中包含多少步
        train_steps = len(train_loader)

        # Early Stopping（早停）是一种正则化手段，用于防止过拟合。
        # patience：在验证集上若干个 epoch（patience 次）内指标不再提升，就提前终止训练。
        # verbose=True：每次监测到更优模型时，打印保存 checkpoint 的日志；并在触发早停时打印提示。
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 功能：根据命令行或配置中的 args.optimizer（如 "Adam"、"SGD"）等参数，调用内部封装方法返回相应的 PyTorch 优化器实例
        # 也就是优化器的选择
        model_optim = self._select_optimizer()

        # 损失函数的选择
        criterion = self._select_criterion(self.args.loss)

        # 学习器
        # 在训练初期逐步将学习率从一个较低值线性/三角地升高到 max_lr，
        # 然后再线性/三角地降回原始的或者更低的学习率。
        # 参数 | 含义
        # optimizer | 关联的优化器实例
        # steps_per_epoch | 每个 epoch 的迭代步数（即 train_steps），用以计算 total steps = steps_per_epoch * epochs
        # epochs | 总训练 epoch 数
        # max_lr | 学习率上限（peak LR），scheduler 会在训练过程中达到此值
        # pct_start | 上升阶段所占比重（0~1），例如 0.3 表示前 30% 的 steps 用于 LR 升至 max_lr，后 70% 用于下降
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):# 外层epoch循环
            iter_count = 0 # 统计本次大循环中已经处理的batch
            train_loss = [] # 统计本次大循环中每个batch的loss， 用于后面球均值

            self.model.train() # 设置模型为训练模式
            epoch_time = time.time() # 记录本 Epoch 开始时刻，用于打印本 Epoch 耗时

            # enumerate(train_loader)：按批次（batch）从 DataLoader 中取出训练数据。
            # iter_count += 1：累加迭代次数，用于速率估算。
            # model_optim.zero_grad()：清空上一轮梯度，避免累加。
            # 每轮enumerate()返回一个迭代器， 产出一个元组（index， element）
            # i是迭代步数（batch的索引）， 从0累加
            # batch就是train_loader产出的“元素”， 在这里是一个包含四个张量的tuple
            # batch_x：输入数据，包含历史时间序列, batch_y：标签数据，包含预测时间序列
            # batch_x_mark：输入数据的时间标记， batch_y_mark：标签数据的时间标记
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1# 累加迭代次数，用于速率估算
                model_optim.zero_grad() # 清空上一轮梯度，避免累加

                # 底下三句话是将数据迁移到指定设备上（GPU 或 CPU）
                # 顺便都转换成 float32 类型
                # .to(self.device)：将张量移动到 GPU/MPS/CPU
                # batch_x_mark（时间特征）通常不需要送入模型前置处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构造Decoder输入
                # 在时间预测任务中， 通常使用编码器-解码器
                # 编码器负责接受输入序列，提取特征并生成隐藏状态
                # 解码器负责根据编码器的隐藏状态和解码器资深的历史预测值去生成预测序列
                # 解码器需要输入两部分信息：已知的历史标签（label_len）：代表预测开始之前的已知真实值，未来时间序列的占位符（pred_len）：用于生成未来的预测值。

                # 创建一个与batch_y中预测长度部分形状一致的全零张量。
                # 用于占位未来的预测值pred_len。
                # 解码器在训练时需要输入的目标序列，但未来数据是未知的，因此用零张量进行占位
                # batch_y是目标序列（Ground Truth），其形状通常为(batch_size, seq_len, feature_dim)。
                # batch_y[:, -self.args.pred_len:, :]提取目标序列的最后pred_len个时间步。
                # torch.zeros_like创建一个与上述张量形状相同但值全为零的张量。
                # .float()确保数据类型为浮点数。
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

                # 将历史标签（label_len）和全零占位符（pred_len）沿时间维度拼接，构造出解码器的输入。
                # 提取历史标签：batch_y[:, :self.args.label_len, :]提取目标序列的前label_len个时间步，作为已知的历史标签。
                # 拼接：使用torch.cat函数，将历史标签和全零张量沿时间维度（dim=1）拼接，拼接完成后前半段是decoder
                # 拼接后的张量形状为(batch_size, label_len + pred_len, feature_dim)。
                # 转换为浮点数：.float()确保数据类型为浮点数
                # 在batch_y中，前self.args.label_len个时间步是已知的历史标签，后self.args.pred_len个时间步是全零占位符
                # batch_y[:, :self.args.label_len, :], 已知部分
                # dec_inp：全零占位符
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # batch_x：编码器输入序列（形状 [B, seq_len, C]）
                # None：在 TimeMixer 中通常忽略时间特征，此处占位
                # dec_inp：解码器输入（拼接了真实标签与占位，形状 [B, label_len+pred_len, C]）
                # 最后的 None：解码器时间特征，同样占位
                # 前向调用模型，得到预测输出
                outputs = self.model(batch_x, None, dec_inp, None)

                # 当多变量预测时，各个特征在最后一个维度拼接，取 -1: 意味着从最后一个变量开始（或取所有变量，看具体实现）；
                # 当单变量预测时，只需取第 0 号通道，故用 0:
                f_dim = -1 if self.args.features == 'MS' else 0

                # 裁剪出预测部分的输出
                # outputs 初始形状 [B, label_len+pred_len, C_out]
                # :（第 0 维）：保留所有 batch
                # -pred_len:（第 1 维）：选取最后 pred_len 个时间步，丢掉前 label_len 个。因为前 label_len 是拼接进 decoder 的真实标签占位逻辑，并不是真正的预测
                # f_dim:（第 2 维）：当 f_dim = 0 时，取从第 0 号通道到末尾（所有通道）；当 f_dim = -1 时，取从倒数第 1 号通道到末尾（可能只取最后一个变量或多个看实现）。
                # -k: —— 从倒数第 k 个一直取到末尾
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                # 对 batch_y（原标签序列）执行同样的切片：
                # 取最后 pred_len 步的真实值
                # 取与 outputs 相同的通道
                # .to(self.device)：确保真实标签也在同一设备上，方便后续 loss 计算。
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 切片操作 [:, -pred_len:, f_dim:]：
                # :（第 0 维）保留全部 B 个样本；
                # -pred_len:（第 1 维）取最后 pred_len 步，即未来时间段对应的时间特征；
                # f_dim:（第 2 维）同前面解释，兼容单/多变量场景，选择需要的通道子维度。
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 计算损失
                # 对于某些自定义 loss（如 MASE、SMAPE），函数签名可能需要额外信息：
                # batch_x：有时用于归一化或尺度计算（例如 MASE 使用历史序列差分）；
                # frequency_map：时间频率（如 'D', 'M'）用于构造基准预测；
                # outputs、batch_y：模型预测与真实标签。
                # batch_y_mark：未来时间特征，用于含时间成分的损失计算。
                # 返回值 loss_value：是一个标量张量（torch.Tensor），包含了此次 batch 的主要损失。
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)

                # 下方注释是可选的平滑度正则项， 计算了“预测与真实序列的一阶差分”之间的 MSE，称作 sharpness loss，用于鼓励预测曲线的平滑度与真实曲线一致。
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                # 权重 1e-5 是经验超参，用来平衡主损失与平滑度正则。
                loss = loss_value  # + loss_sharpness * 1e-5

                # loss.item()：将标量张量转换为 Python 浮点数，便于统计与打印；
                # train_loss 是一个 Python 列表，用来累积本 Epoch 中每个 batch 的 loss；
                # 后续会对 train_loss 求平均，得到该 Epoch 的平均训练误差，便于监控收敛情况。
                train_loss.append(loss.item())


                # 打印用于监控训练过程的日志信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播
                # 器学习的目标是最小化损失函数，需要根据梯度信息来更新参数。
                loss.backward()
                # 执行参数更新：优化器（如 Adam、SGD）根据每个参数的 .grad，按照预设的学习率和更新规则，修改参数值
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            # 打印本轮epoch训练时间
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # train_loss 是一个列表，记录每个 batch 的损失 loss.item()，通过 np.average(...) 计算 本轮 epoch 的训练集平均损失，作为本轮的最终训练结果。
            train_loss = np.average(train_loss)
            # 调用模型类中定义的 self.vali() 方法，对验证集进行评估，返回验证集的平均损失。
            vali_loss = self.vali(train_loader, vali_loader, criterion)

            # 实际上这个 test_loss 是 等于验证集损失 vali_loss 的，也就是说这时候并没有单独评估测试集。
            test_loss = vali_loss

            # 打印训练集、验证集和测试集的损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 这一行调用提前停止机制 early_stopping 实例，用当前 vali_loss（验证集损失）来判断到 path/checkpoint.pth；否则，内部会计数“连续未提升次数”
            # vali_loss：当前 epoch 的验证损失；
            # self.model：当前模型；
            # path：保存模型的路径，最佳模型会被保存为 checkpoint.pth。
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # self.args.lradj 是一个策略标志：
            # 如果不是 'TST'，就使用自定义的 adjust_learning_rate(...) 函数进行学习率调整；
            # 例如可以实现周期性下降、warmup、分段下降等策略。
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 加载在 EarlyStopping 过程中记录的“验证集表现最优的模型权重”。
        # 这一操作确保返回的模型是表现最好的，而不是最后一次 epoch 的模型。
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 返回已经完成训练（且恢复到最佳权重）的模型，供外部调用或测试使用。
        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        '''
            1.数据准备
            目的: 从训练集和验证集提取输入序列和目标时间序列，并将数据转换为适配模型的格式。
            具体操作: 提取训练集的最后一个窗口作为输入序列 (x)，提取验证集的目标时间序列 (y)，并调整张量形状。

            2.模型评估模式设置
            目的: 禁用梯度计算和训练特性（如 dropout），以节省内存并确保推理结果稳定。
            具体操作: 调用 self.model.eval() 和 torch.no_grad()。

            3.解码器输入构造
            目的: 为解码器提供输入，包括历史标签和未来时间步的占位符。
            具体操作: 拼接历史标签和全零张量，形成解码器输入。

            4.模型前向传播
            目的: 使用模型生成预测值。
            具体操作: 将输入数据分批送入模型，避免内存溢出，并存储预测结果。

            5.预测值和真实值处理
            目的: 提取预测值的目标维度，并与真实值对齐，准备计算损失。
            具体操作: 根据任务类型（单变量或多变量）选择特定维度的预测值。

            6.损失计算
            目的: 评估模型预测值与真实值之间的误差。
            具体操作: 调用损失函数 criterion 计算损失。

            7.恢复训练模式
            目的: 恢复模型为训练模式，确保后续训练不受影响。
            具体操作: 调用 self.model.train()。

            8.返回损失值
            目的: 将验证集的损失值返回给调用者，用于评估模型性能。
        '''
        # train_loader：训练集的数据加载器（用来获取最后一段输入）。
        # vali_loader：验证集数据加载器（用于评估模型）。
        # criterion：损失函数，可能是自定义的支持时间序列评估的函数。
        # 验证函数， 该函数的主要目的是在验证集上计算模型的损失值（loss），以评估模型的性能。
        # 它通过调用模型的前向传播，生成预测值（pred），并与真实值（true）进行比较，计算损失。
        x, _ = train_loader.dataset.last_insample_window()  #  从训练集的最后一个窗口中提取输入序列（last_insample_window），作为验证的输入数据。
        y = vali_loader.dataset.timeseries  # 从验证集提取目标时间序列（真实值）
        x = torch.tensor(x, dtype=torch.float32).to(self.device) # x转换成张量， 提取到设备上
        x = x.unsqueeze(-1) # 输入数据增加一个特征维度，形状从 [B, T] 变为 [B, T, 1]，以适配模型的输入格式。

        # self.model.eval(): 将模型设置为评估模式，禁用 dropout 和 batch normalization 的训练行为。
        # torch.no_grad(): 禁用梯度计算，减少内存消耗，加速推理。
        self.model.eval()

        # 在该with代码块中， 所有张量都不会被跟踪梯度，不会创建计算图
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape # 获取输入数据的形状，B是batch size，C是特征维度
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            # 前半部分是历史标签（label_len），从输入序列的最后部分提取。
            # 后半部分是全零张量（pred_len），作为未来时间步的占位符
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()

            # encoder - decoder
            # 初始化输出容器， 创建一个全零张量来储存输出结果， Batch size、预测长度和特征维度， pred_len：要预测的时间步数， c：特征数
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)

            # 将验证集 batch 分割成若干子批，每段 500 条（或你自己设定的数量）；
            # 避免一次性送入全部数据导致 GPU 内存溢出（out of memory）。
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)

            # x_enc: 当前小段的输入；
            # dec_inp[batch_slice]: 对应的 decoder 输入（注意要和 encoder 对应！）；
            # self.model(...): 调用模型的前向传播（forward）；
            # .detach()：切断梯度追踪；
            # .cpu()：结果放回 CPU；
            # 存入 outputs 的对应位置。
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x_enc, None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()

            # features == 'MS' 时表示多变量（multivariate series）预测多变量，我们取所有特征维度（-1 表示最后一个维度开始）；
            # 否则，我们预测的是单变量（S），我们只保留第一个维度。
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]

            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            # 计算损失
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x_enc, None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]], None)

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return

