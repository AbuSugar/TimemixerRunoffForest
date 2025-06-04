import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, setting='default_setting'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.setting = setting

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 这里的 path 已经是像 './checkpoints/long_term_forecast_..._0' 这样的目录了
        # 我们要在这个目录下保存一个名为 'checkpoint.pth' 的文件
        final_save_path = os.path.join(path, 'checkpoint.pth') # <-- 关键修改在这里！

        # 确保目标目录存在
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True) # exist_ok=True 避免目录已存在时报错

        torch.save(model.state_dict(), final_save_path)
        self.val_loss_min = val_loss

class AdjustLearningRate:
    def __init__(self, optimizer, lr_adjust_type='type1', patience=10, cos_lr=False, warm_up_epochs=0, min_lr=1e-6,
                 factor=0.2, gamma=0.1, verbose=False, **kwargs):
        # ... (此部分代码保持不变，与您仓库中的原代码一致)
        if lr_adjust_type == 'type1':
            self.schedule = {
                0: 0.0005,
                1: 0.00025,
                2: 0.000125,
                3: 0.0000625,
                4: 0.00003125,
                5: 0.000015625,
                6: 0.0000078125,
                7: 0.00000390625,
                8: 0.000001953125,
                9: 0.0000009765625
            }
        elif lr_adjust_type == 'type2':
            # ... (其他 lr_adjust_type 保持不变)
            pass
        elif lr_adjust_type == 'type3':
            # ... (其他 lr_adjust_type 保持不变)
            pass
        elif lr_adjust_type == 'constant':
            # ... (其他 lr_adjust_type 保持不变)
            pass
        self.optimizer = optimizer
        self.factor = factor
        self.gamma = gamma
        self.verbose = verbose
        self.lr_adjust_type = lr_adjust_type
        self.patience = patience
        self.cos_lr = cos_lr
        self.warm_up_epochs = warm_up_epochs
        self.min_lr = min_lr
        self.current_lr = 0
        self.counter = 0

    def __call__(self, epoch, vali_loss, current_lr):
        if self.lr_adjust_type == 'type1':
            new_lr = self.schedule.get(epoch, self.min_lr)
            if new_lr != self.optimizer.param_groups[0]['lr']:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.current_lr = new_lr
                if self.verbose:
                    print('Updating learning rate to %f' % new_lr)
        elif self.lr_adjust_type == 'type2':
            # ... (其他 lr_adjust_type 逻辑保持不变)
            if epoch >= self.warm_up_epochs:
                if self.cos_lr:
                    new_lr = self.min_lr + 0.5 * (current_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * (epoch - self.warm_up_epochs) / (self.patience * 2)))
                else:
                    new_lr = current_lr * self.factor
                if new_lr >= self.min_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.current_lr = new_lr
                    if self.verbose:
                        print('Updating learning rate to %f' % new_lr)
                else:
                    new_lr = self.min_lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.current_lr = new_lr
                    if self.verbose:
                        print('Updating learning rate to %f' % new_lr)
            else:
                self.current_lr = current_lr
        elif self.lr_adjust_type == 'type3':
            # ... (其他 lr_adjust_type 逻辑保持不变)
            if epoch == 0:
                self.current_lr = current_lr
            if (epoch + 1) % 1 == 0:
                new_lr = current_lr * 0.5
                if new_lr >= self.min_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.current_lr = new_lr
                    if self.verbose:
                        print('Updating learning rate to %f' % new_lr)
                else:
                    new_lr = self.min_lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.current_lr = new_lr
                    if self.verbose:
                        print('Updating learning rate to %f' % new_lr)
            else:
                self.current_lr = current_lr
        elif self.lr_adjust_type == 'constant':
            # ... (其他 lr_adjust_type 逻辑保持不变)
            new_lr = current_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
        return self.current_lr


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
