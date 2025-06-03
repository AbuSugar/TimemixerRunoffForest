import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

"""TimeMixer 预测流程概览
TimeMixer 的预测流程可以分为以下主要步骤：
未来时间特征编码：如果设置了使用未来时间特征（use_future_temporal_feature=True），首先对未来时间特征 x_mark_dec 进行嵌入编码（使用 enc_embedding），并在通道独立模式下复制到每个通道上​
file-akgywscpdfvslm1rasdzvx
。
多尺度下采样：对历史输入序列 x_enc（形状为 $B\times T\times N$）和历史时间特征 x_mark_enc 进行多尺度下采样。具体地，使用 MaxPool1D/AvgPool1D 或卷积按窗口 w 进行下采样，得到多层尺度的数据序列​
file-akgywscpdfvslm1rasdzvx
。例如，下采样层数为 2 时，会生成尺度0：$(B\times T\times N)$、尺度1：$(B\times T/w\times N)$、尺度2：$(B\times T/w^2\times N)$ 的列表​
file-akgywscpdfvslm1rasdzvx
。
数据归一化：对每个尺度的序列分别使用 Normalize 层进行归一化处理​
file-akgywscpdfvslm1rasdzvx
。这一步保持输入各特征维度的尺度一致，便于后续处理。
通道独立变换（如果开启通道独立模式）：在通道独立模式下，将每个通道视为单独序列进行处理。具体操作是将归一化后的张量从 $(B,T,N)$ 重塑为 $(B\times N,,T,1)$，并将对应的时间特征按通道复制​
file-akgywscpdfvslm1rasdzvx
。
序列分解：对归一化后的多尺度序列进行趋势/季节性分解（使用 series_decomp）​
file-akgywscpdfvslm1rasdzvx
​
file-akgywscpdfvslm1rasdzvx
。在多通道模式下，将每个尺度的序列拆分为趋势分量和残差分量（列表形式）；在单通道独立模式下，跳过此步骤，直接使用原序列。
输入嵌入：将每个尺度的序列（以及对应的时间特征）输入到嵌入层 DataEmbedding_wo_pos 中，转换为高维特征向量​
file-akgywscpdfvslm1rasdzvx
。输出形状为 $(B,T,d_model)$（或通道独立模式下的 $(B\times N,,T,d_model)$）。此时，每个尺度的序列已被编码为 d_model 维特征。
过去可分解混合 (PDM) 编码：将各尺度的嵌入特征作为列表，依次输入 pdm_blocks 进行处理​
file-akgywscpdfvslm1rasdzvx
。PDM 模块以可分解设计混合不同尺度的序列成分，分别在细到粗、粗到细方向上交互季节性和趋势信息​
blog.csdn.net
​
blog.csdn.net
。经过 $e_layers$ 层 PDM 处理后，得到编码后的多尺度特征列表。
未来多预测混合 (FMM) 解码：调用 future_multi_mixing 对编码后的多尺度特征进行解码。对于每个尺度 $i$，首先通过线性层 predict_layers[i] 将时间维度从原始长度映射到预测长度 pred_len​
file-akgywscpdfvslm1rasdzvx
。在有未来时间特征时，将步骤1中嵌入的时间特征加到预测结果上，并通过 projection_layer 生成通道数输出；在非通道独立模式下，还使用 out_projection 将残差成分加入预测结果​
file-akgywscpdfvslm1rasdzvx
。每个尺度得到输出张量 $B\times \text{pred_len}\times C_{out}$。
多尺度输出融合：将各尺度的预测结果列表沿最后一维堆叠后求和（Stack & Sum），得到形状 $(B,\text{pred_len},C_{out})$ 的预测值。最后，对融合后的输出应用逆归一化（denorm，即使用 normalize_layers[0]）​
file-akgywscpdfvslm1rasdzvx
。得到的 dec_out 即为最终的预测序列。
以上步骤概括了 forecast() 函数的执行流程，其中数据形状变化如下（以多尺度为例）：
输入：历史序列 $x_{enc}\in\mathbb{R}^{B\times T\times N}$，历史时间特征 $x_{mark_enc}$（可选），未来时间特征 $x_{mark_dec}$（可选）；输出：预测序列 $Y\in\mathbb{R}^{B\times \text{pred_len}\times C_{out}}$。
在下采样后，第 $i$ 个尺度的历史序列形状为 $B\times \frac{T}{w^i}\times N$。嵌入后变为 $B\times \frac{T}{w^i}\times d_{model}$。最终经 FMM 解码后，每个尺度输出形状为 $B\times \text{pred_len}\times C_{out}$，融合后形状为 $B\times \text{pred_len}\times C_{out}$​
file-akgywscpdfvslm1rasdzvx
​
file-akgywscpdfvslm1rasdzvx
。
其他模块简介
数据嵌入 (Data Embedding)：使用 DataEmbedding_wo_pos 层将原始时序数据和（可选的）时间特征映射到高维空间​
file-akgywscpdfvslm1rasdzvx
。其作用类似于编码器输入层，将每个时间点的标量特征编码为 $d_{model}$ 维向量。
过去可分解混合 (PDM)：PDM 模块负责提取历史信息并混合不同尺度上的趋势和季节成分​
blog.csdn.net
。它首先对序列进行分解，然后在多个尺度上从细到粗、从粗到细分别混合季节性和趋势成分，捕获微观和宏观的时序模式​
blog.csdn.net
​
blog.csdn.net
。
未来多预测混合 (FMM)：FMM 模块在预测阶段使用多个线性预测器对不同尺度的特征进行并行预测，并将它们的输出进行融合​
blog.csdn.net
​
file-akgywscpdfvslm1rasdzvx
。每个预测器基于不同尺度的信息生成未来值，通过叠加多尺度预测结果增强预测能力。
归一化 (Normalize)：在每个尺度上对序列进行标准化或归一化，使不同特征维度的数值范围一致，有助于模型稳定训练​
file-akgywscpdfvslm1rasdzvx
。训练结束后对输出进行反归一化，还原预测值的原始尺度。​
file-akgywscpdfvslm1rasdzvx
​
file-akgywscpdfvslm1rasdzvx
下采样 (Downsampling)：使用如 MaxPool1D、AvgPool1D 或带步幅的卷积层按设定窗口对时序数据进行下采样​
file-akgywscpdfvslm1rasdzvx
。通过下采样，原始序列被分解成多个采样频率不同的新序列，实现多尺度表示。
多尺度处理：TimeMixer 通过多尺度视角捕获时间序列的不同周期信息​
blog.csdn.net
。例如，对电力负荷序列分别按小时和按天采样，会体现不同的周期性特征​
blog.csdn.net
；多尺度策略使得模型既能关注短期微观变化，也能捕捉长期宏观趋势。
示例说明
假设原始历史序列长度 $T=96$，下采样窗口 $w=2$，下采样层数为 2，则：
尺度0（原始）：长度 $T_0=96$；尺度1：长度 $T_1=96/2=48$；尺度2：长度 $T_2=96/2^2=24$。
归一化后，每个尺度分别输入嵌入和 PDM。最终，每个尺度输出一个 $B\times \text{pred_len}\times C_{out}$ 的预测序列，叠加后得到总预测。
这个示例说明通过多尺度下采样，原本只在尺度0捕获不到的更粗周期变化（如天/周周期）会在尺度1、2中得到体现，从而提高预测能力​
blog.csdn.net
​
blog.csdn.net
。
流程图汇总
下图概括了 TimeMixer 预测流程中的各模块和数据流向：
mathematica
复制
编辑
原始输入 X_enc (B×T×N) + 时间特征 X_mark_enc
    │
    ↓  多尺度下采样 (MaxPool1D/AvgPool1D/Conv1D)  
┌──────────────┬───────────────┬───────────────┐
│ 尺度0: B×T×N │ 尺度1: B×T/w×N │ 尺度2: B×T/w^2×N …│  
└──────────────┴───────────────┴───────────────┘
    ↓  归一化 (各尺度单独归一化)  
    ↓  序列分解（趋势+季节性）  
    ↓  数据嵌入 (DataEmbedding)  → 得到嵌入特征列表 [B×T×d_model]  
    ↓  PDM编码 (多尺度混合)  
    ↓  FMM解码 (多预测器生成预测)  
    ↓  多尺度输出融合 (Stack + Sum)  
    ↓  反归一化 (Denorm)  
    ↓  最终输出 Y (B×pred_len×C_out)
该流程图清晰展示了从原始序列到多尺度输出预测的每一步骤，包括数据形状的变化和模块功能。通过多尺度下采样、归一化、分解、嵌入、PDM 和 FMM 等处理，TimeMixer 能同时利用不同时间尺度上的信息进行高质量预测​
blog.csdn.net
​
file-akgywscpdfvslm1rasdzvx
。"""
""" 
    输入序列首先通过多尺度预处理生成不同下采样尺度的历史观测，
    接着分别进行过去可分解混合（PDM）编码（分离季节性和趋势成分并跨尺度混合）和未来多预测者混合（FMM）解码（集成来自各尺度的预测器输出），
    最终得到预测结果
"""

class DFT_series_decomp(nn.Module):
    """
    基于离散傅里叶的时间序列分解模块，其功能主要是讲输入的时间序列分解为 季节性分量 和 趋势性分量
    季节性分量：通过保留频域中最显著的频率成分（top_k 个频率）来提取周期性变化。----》
    趋势分量：通过从原始序列中减去季节性分量，提取长期变化趋势。
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k # top_k参数指定了在频域中保留的最显著频率成分的数量。用于控制季节性分量的复杂度

    def forward(self, x): #  前向传播
        xf = torch.fft.rfft(x) # 对输入时间序列 x 进行快速傅里叶变换 (torch.fft.rfft)，将其从时域转换到频域，得到复数频谱 xf

        # # 计算频谱的幅值（freq = abs(xf)），并将直流分量（频率为 0 的部分）置为 0（freq[0] = 0），以避免直流偏移对结果的影响
        freq = abs(xf)
        freq[0] = 0

        # 通过 torch.topk 找到幅值最大的 top_k 个频率及其对应的索引
        top_k_freq, top_list = torch.topk(freq, self.top_k)

        # 将频谱中幅值小于 top_k 最小值的频率分量置为 0（xf[freq <= top_k_freq.min()] = 0），以保留主要的周期性成分
        xf[freq <= top_k_freq.min()] = 0

        # 对处理后的频谱进行逆快速傅里叶变换 (torch.fft.irfft)，将其转换回时域，得到季节性分量 x_season。
        x_season = torch.fft.irfft(xf)
        # 通过从原始序列 x 中减去季节性分量 x_season，计算趋势分量 x_trend。
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    从细到粗，每两层之间的融合过程，下层负责提取细粒度的特征，上层负责吸收（融合）并输出更粗粒度的表示。
    用于季节分量的多尺度混合
    通过逐层下采样的方式，从高分辨率（高频）到低分辨率（低频）逐步混合季节性特征。
    Bottom-up mixing season pattern
    1. 初始化阶段 (__init__)：
    构建了一组下采样处理模块 down_sampling_layers，使用 nn.ModuleList 统一管理。
    每个模块都是一个由以下结构组成的 nn.Sequential：
    Linear 层：对时间序列进行下采样（降低时间分辨率）。
    GELU 激活：引入非线性，使特征表达更丰富。
    Linear 层：在下采样后的特征空间进行转换处理。
    这些模块可以看作是 “不同尺度之间的映射器”，专门负责从高分辨率向低分辨率传递信息。

    2. 前向传播阶段 (forward)：
    传入多个不同时间尺度的“季节性分量”（比如从小时级到天级、周级）。

    然后：

    从高分辨率（细粒度）开始，逐层调用对应的 down_sampling_layer，将当前高频表示转换为下一级尺度；

    与当前低频表示进行残差融合（out_low + out_low_res）；

    迭代更新 out_high 和 out_low；

    每一步融合结果都保留下来，输出全体融合序列。
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        """
        down_sampling_layers 是一个 ModuleList，包含多个下采样层，每层由以下组成：
            线性变换：将输入序列的长度从当前尺度下采样到下一尺度。
            激活函数：使用 GELU 激活函数引入非线性。
            线性变换：进一步处理下采样后的特征
        """
        self.down_sampling_layers = torch.nn.ModuleList(# ModuleList 是 PyTorch 的容器，用来存储多个 nn.Module，会自动被添加到模型参数中，参与训练
            [
                nn.Sequential(# nn.Sequential 是一个模块容器，用来把多个子模块顺序连接。每一层结构由 下采样+激活函数+特征变换 三部分组成
                    # 创建一个线性层（全连接层），用来做下采样（也就是降低时间维度长度）
                    # torch.nn.Linear(in_features, out_features)， 接收一个大小为 in_features 的向量，输出一个大小为 out_features 的向量，
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),# 当前下采样层的输入时间步数（也就是当前层的时间维度长度）
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)), # 下采样后的输出时间步数（也就是下一层的时间维度长度）
                    ),
                    nn.GELU(), # 激活函数
                    # 在不改变维度的前提下，对下采样后的数据进行 线性变换，
                    # 这里的线性变换是为了进一步学习时序特征之间的组合关系
                    # 本质上是特征之间的重混合（feature mixing
                    torch.nn.Linear(# 在下采样后的时间尺度上，保持序列长度不变，进一步学习时序特征之间的组合关系。
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                # 动态创建多个下采样层的配置，每层处理一个新的时间尺度。
                # 例如：第一层从 96 点 -> 48 点，第二层从 48 点 -> 24 点... 依此类推。
                for i in range(configs.down_sampling_layers)
            ]
        )
    """是的，你的理解是正确的。每一步融合的流程如下：

        ### **详细流程**
        1. **Step 1**:
           - `season_0`（高频）作为 `out_high`。
           - 通过 `down_sampling_layers[0]` 提取特征，得到 `out_low_res`。
           - 将 `out_low_res` 融合到 `season_1`（中频）中，更新 `out_low`。
           - 更新后的 `out_low` 作为新的 `out_high`。
        
        2. **Step 2**:
           - 上一步融合后的 `out_high`（即更新后的 `season_1`）作为输入。
           - 通过 `down_sampling_layers[1]` 提取特征，得到 `out_low_res`。
           - 将 `out_low_res` 融合到 `season_2`（低频）中，更新 `out_low`。
           - 更新后的 `out_low` 作为新的 `out_high`。
        
        ### **总结**
        - 每一步中，`out_high` 是当前的高频分量，`out_low` 是当前的低频分量。
        - 通过 `down_sampling_layers[i]` 提取高频特征，并将其融合到低频分量中，逐步完成从高频到低频的多尺度季节性特征混合。
        """
    def forward(self, season_list):
        # 实现了一个从高频率到低频率，利用多层 down_sampling_layers，逐层融合不同尺度的时间序列信息（高频 -> 低频），并输出每一步融合后的结果

        # mixing high->low

        # season_list 是一个列表，每个元素是一个表示不同时间尺度下的特征张量

        out_high = season_list[0]   # out_high 是当前最高分辨率（最长时间维度）的输入
        out_low = season_list[1]    # out_low 是下一个尺度的输入（准备融合）
        # out_season_list 初始化一个输出列表，记录每一层融合结果
        # .permute(0, 2, 1) 是在将 [B, C, T] → [B, T, C]，方便后续操作
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            #  self.down_sampling_layers 是一个 torch.nn.ModuleList，其中的每个元素是一个 nn.Sequential 模块
            # self.down_sampling_layers[i](out_high) 的调用实际上是将 out_high 作为输入，依次通过第 i 个 nn.Sequential 中的所有子模块，完成下采样、激活和特征变换操作，最终返回处理后的结果
            out_low_res = self.down_sampling_layers[i](out_high)
            # 和当前的低频表示相加，实现特征融合（有点像残差连接）
            out_low = out_low + out_low_res
            # 当前的“融合结果”变成下一轮的输入 out_high
            out_high = out_low
            # 继续准备下一层低频输入，和下次融合作比较
            # 逻辑上就是 season_list[i+2] 是下下一层
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            # 把当前融合后的表示（[B, C, T] → [B, T, C]）保存起来
            # 用于后续模型的使用或输出
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    这是用来实现趋势分量逐层从粗粒度到细粒度逐步融合，每次融合一次后输出当前的结果，并将其作为下次融合的输入，最终，它会输出所有融合后的趋势分量列表
    从低分辨率（低频）到高分辨率（高频）逐步融合趋势特征。这种设计适用于时间序列建模中的趋势分量处理，能够逐层上采样并融合不同尺度的趋势特征
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                # Linear(低分辨率长度 → 高分辨率长度) → GELU → Linear(高分辨率 → 高分辨率)
                nn.Sequential(
                    # 第一次线性层假设输入是 [batch_size, channels, low_seq_len]；
                    # 你需要把它变成 [batch_size, channels, high_seq_len]；
                    # 由于 PyTorch 的 nn.Linear 是作用在最后一维上的，实际你需要先 permute 变成 [batch_size, low_seq_len, channels] 形式，
                    # 让每个时间步都是一个向量，才能用线性层扩展它的“时间维度”。
                    # 例：configs.seq_len = 16， configs.down_sampling_window = 2， configs.down_sampling_layers = 2
                    # 假设 i = 0 时，configs.seq_len // (configs.down_sampling_window ** i + 1) = 8， configs.seq_len // (configs.down_sampling_window ** i) = 16
                    # 实现从粗到细
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),

                    # 激活函数，用于引入非线性
                    nn.GELU(),

                    #  nn.Linear(高分辨率 → 高分辨率)
                    # 一个投影层，相当于对上采样之后的特征再处理一次；
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                # 构建的是从低分辨率 → 高分辨率的上采样层，所以顺序要反过来
                # configs.seq_len = 64
                # configs.down_sampling_window = 2
                # configs.down_sampling_layers = 3
                # 则 trend_list 的长度按层为：
                # Layer 0: 64 → 32
                # Layer 1: 32 → 16
                # Layer 2: 16 → 8
                # [trend_8, trend_16, trend_32, trend_64]   # 从底层到顶层
                # 如果你要从 trend_8 开始重建 trend_16（即从底层往上重建），那你应该依次创建：
                # Linear(8 → 16) → GELU → Linear(16 → 16)
                # Linear(16 → 32) → GELU → Linear(32 → 32)
                # Linear(32 → 64) → GELU → Linear(64 → 64)
                # 所以顺序：i = 2 → i = 1 → i = 0
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        """
          trend_8  ──> 上采样 ─┐
                             ├─> 融合 trend_16 ──> 上采样 ─┐
        trend_16────────────┘                           ├─> 融合 trend_32 ──> 上采样 ─┐
                                                     trend_32──────────────────────┘
                                                                  ...
        """
        # 从低分辨率到高分辨率逐步融合趋势特征
        # mixing low->high
        # 输入的trend_list是一个趋势特征列表，假设：trend_list = [trend_64, trend_32, trend_16, trend_8]
        # 每个trend_list[i] shape 是 [batch_size, channels, seq_len=i]
        # 创建 trend_list 的副本 trend_list_reverse，并将其顺序反转。这样可以从低分辨率（低频）到高分辨率（高频）逐步处理
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse() # 反转后[trend_8, trend_16, trend_32, trend_64]



        out_low = trend_list_reverse[0] # 此时这是最粗粒度的趋势特征
        out_high = trend_list_reverse[1] # 最细粒度的趋势特征， 初始化为第二低分辨率层， 用在第一个融合过程

        # 初始化输出列表 out_trend_list，存储每次融合后的趋势特征
        # 将 out_low 从 [B, C, L] → [B, L, C]，便于后续拼接或对齐
        # out_low.permute(0, 2, 1)：将张量维度从 [batch_size, channels, seq_len] 转换为 [batch_size, seq_len, channels]，方便后续处理
        # permute 是 PyTorch 中用于重排维度顺序的函数。它不会改变数据内容，只会改变维度的顺序（排列）
        # [B, C, L]----> batch_size, channel, seq_len
        # permute(0, 2, 1) 是将 out_low 的维度顺序从 [B, C, L] 变为 [B, L, C]，这样做的目的是为了方便后续操作
        out_trend_list = [out_low.permute(0, 2, 1)]

        # 迭代总次数为 len(trend_list_reverse) - 1，比如有 4 层 trend，就会循环 3 次
        for i in range(len(trend_list_reverse) - 1):
            # out_low 是上一次迭代的输出（初始为 trend_8）
            # up_sampling_layers是上面写的处理模块，第一轮中Linear(8 → 16) → GELU → Linear(16 → 16)
            out_high_res = self.up_sampling_layers[i](out_low)
            # 把当前层 trend (out_high) 和刚刚上采样的结果相加，用下层的信息增强上层趋势
            out_high = out_high + out_high_res
            # 更新 out_low，为下一次上采样做准备
            out_low = out_high
            # 提前获取下一层的原始 trend（如果还没结束）， 为下一轮准备好对应的 out_high（即 trend_32, trend_64...
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            # 把当前输出保存到 out_trend_list， 并permute 成 [B, L, C]，符合常用输入输出格式。
            out_trend_list.append(out_low.permute(0, 2, 1))
        # 因为我们是从底层开始融合的，输出也按这个顺序生成,所以最终要反转回来，顺序和输入 trend_list 对齐
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
    PDM模块内部每次处理时，都会把输入序列（不管来源于 pre_enc 还是上层输出的）
    再次动态地分解成两个部分：

                season （快速波动成分，小周期）

                trend （平滑变化成分，大趋势）

    可以理解成一个完整的处理模块， 包含上面趋势和季节的处理过程， 顺便新增一些东西
    对输入序列进行 趋势（trend）与季节（season）分解；
    分别对 trend 和 season 进行 多尺度融合处理；
    将融合结果加和，作为输出增强后的序列。
    """
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        # 存储输入、输出序列长度和下采样窗口（用于多尺度构建）,参数初始化
        self.seq_len = configs.seq_len # 输入序列长度
        self.pred_len = configs.pred_len # 预测未来的时间长度
        self.down_sampling_window = configs.down_sampling_window # 下采样窗口大小

        self.layer_norm = nn.LayerNorm(configs.d_model) # 对最后一个维度做归一化，保证每个样本的维度特征分布更稳定；
        self.dropout = nn.Dropout(configs.dropout) # 防止过拟合，在训练过程中以一定概率将某些神经元置为 0
        self.channel_independence = configs.channel_independence # 如果设为 0，就使用“通道间交互”方式（shared channel interaction）；如果设为 1，就使用“每个通道独立处理”方式（channel independence）；对应下面 cross_layer 是否生效。

        # 对处理后序列的 趋势/季节 分解方式的选择
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        # 通道交互模块（仅在非独立通道模式下使用）
        # 这是一个 MLP 子网络，用于对每个 time step 上的通道维进行深度转换
        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        # 多尺度季节成分融合模块从细到粗
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        # 混合趋势， 从粗到细
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)


        # 最终输出前的一个残差增强网络；
        # 如果启用了 channel_independence = 1，该层用于进一步建模 trend + season 之后的特征；        # 理解上类似 transformer 的 residual block 或 Autoformer 中的 decoder FFN。
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list): # x_list是个多尺度的输入序列列表， 其中每个元素的形状【B，T，C】
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = [] # 初始化季节列表用于存储分解后的季节性分量
        trend_list = [] # 初始化趋势列表用于存储分解后的趋势性分量
        # 对每个输入序列进行趋势和季节性分解
        for x in x_list:
            season, trend = self.decompsition(x) # decomposition方法取决于上面的配置
            # 若开启“通道交互模式”，则对 trend 与 season 分别送入全连接层增强表达
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            # 把 shape 从 [B, T, C] 变为 [B, C, T]再存储到对应的列表，适配后续 MultiScale...Mixing 中对时间序列的处理
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        if len(season_list) % 50 == 0:
            print(f"[Debug] season_list length: {len(season_list)}")  # 👈
        # 进行季节性分量的混合
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        # 进行趋势性分量的混合
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        # 初始化out_list，用于存储最终的输出
        out_list = []

        # 这个 for 循环会同时遍历：
        # x_list：原始的输入序列（shape 通常为 [B, T, C]）；
        # out_season_list：多尺度混合后的季节性成分（shape 一般为 [B, C, T]）；
        # out_trend_list：多尺度混合后的趋势成分（shape 同样为 [B, C, T]）；
        # length_list：每个输入序列的原始时间长度 T，用于裁剪；
        # zip(...) 会把四个列表里的元素打包成元组，一起迭代。
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend # 将季节性和趋势部分加起来，构成最终融合的时间特征， 注意这俩形状相同所以可以直接相加
            # 如果 channel_independence == 1，也就是说每个通道互不独立，我们要加入额外的建模
            # 使用 self.out_cross_layer（一个包含两层 Linear + GELU 的网络）对 out 做非线性变换； 然后再加上原始输入 ori（残差连接）；
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
                # 因为多尺度上采样或变化后，时间维 T 可能大于原始值；
                # 所以我们使用 length 进行裁剪，确保时间维一致；
                # out[:, :length, :]：
                # 取 batch 维度全部；
                # 时间维度取前 length 个；
                # 通道维度全部；
                # 最终把裁剪后的 out 加入 out_list 列表中
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs): # configs 是一个配置对象，包含了模型的超参数和其他设置
        super(Model, self).__init__() # 调用父类nn.Module的的初始化方法， 确保模型的继承的功能被正确初始化
        self.configs = configs # 传入的配置对象 configs 存储为类的属性，供其他方法访问

        # 从configs中提取任务名称、序列长度、标签长度、预测长度等参数
        self.task_name = configs.task_name # 从配置中提取任务名称， 用于决定模型的行为
        self.seq_len = configs.seq_len  # 输入序列长度
        self.label_len = configs.label_len # 标签序列长度
        self.pred_len = configs.pred_len # 预测序列长度
        self.down_sampling_window = configs.down_sampling_window # 下采样窗口大小
        self.channel_independence = configs.channel_independence # 通道独立性标志，决定是否使用通道间交互
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)    # 初始化多个 PastDecomposableMixing 模块，用于多尺度特征处理
                                         for _ in range(configs.e_layers)]) # 构建多个 Past Decomposable Mixing 编码块，用于多层时间序列特征提取，每层一个模块，共 e_layers 层。

        self.preprocess = series_decomp(configs.moving_avg) # 初始化序列分解模块，常用于时间序列去噪或分离趋势，移动平均窗口大小由 configs.moving_avg 指定
        self.enc_in = configs.enc_in # enc_in: 输入序列的通道数量（特征维度）
        self.use_future_temporal_feature = configs.use_future_temporal_feature # 指定是否使用未来时间特征，影响模型的预测行为

        # 构建数据嵌入模块
        # DataEmbedding_wo_pos：去除位置编码的嵌入层（仅使用数值和时间特征）
        # 若通道独立，则每个通道单独处理；否则一起处理
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers # 设置模型的层数，保存编码器的总层数，表示 PastDecomposableMixing 模块的堆叠次数。

        # 构建归一化层，对每个下采样尺度分别归一化，保证多尺度输入在同一数值范围。
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        #长期预测或者是短期预测
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 每个下采样尺度对应一个预测层，用于把当前序列长度预测为未来 pred_len 长度
            # self.predict_layers:

            # 初始化一个 ModuleList，用于存储多个线性层，每个线性层对应一个下采样尺度。
            # 每个线性层的作用是将当前序列长度（configs.seq_len // (configs.down_sampling_window ** i)）映射到预测序列长度（configs.pred_len）
            self.predict_layers = torch.nn.ModuleList(
                [
                    # 定义线性层，输入维度为当前下采样后的序列长度，输出维度为预测序列长度。
                    # 作用：将每一尺度的下采样时间序列（长度变短了）映射为固定的预测长度 pred_len
                    # configs.seq_len // (configs.down_sampling_window ** i)：表示当前下采样后的序列长度，i 是下采样层的索引。
                    # configs.pred_len：表示预测序列的长度。
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),

                        configs.pred_len,
                    )
                    # 遍历所有下采样层（包括原始序列和下采样后的序列），共 configs.down_sampling_layers + 1 层。
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            #  构建输出 projection 层（预测值通道映射），将嵌入维度映射为实际输出维度（每通道或整体）
            if self.channel_independence == 1: # 如果通道间相互独立
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:# 如果通道间不相互独立
                self.projection_layer = nn.Linear( # projection 层（预测值通道映射），将嵌入维度映射为实际输出维度
                    configs.d_model, configs.c_out, bias=True)

                # 构建残差模块
                # 在多尺度建模中，模型往往会提取主特征（主干输出），但残留的低频趋势或非线性部分可能没有建模好。
                # 因此，引入 残差建模模块，对这些“遗留信号”单独学习，并作为辅助分支修正主预测。
                # out_res_layers 的作用：用于学习残差信号（比如原始趋势部分，输入输出维度相同，相当于学一个变换（非降维）
                # 假设：
                #
                # seq_len = 96
                #
                # down_sampling_window = 2
                #
                # down_sampling_layers = 2
                #
                # 则 out_res_layers 构建的 3 个线性层分别是：
                #
                # Linear(96, 96)
                #
                # Linear(48, 48)
                #
                # Linear(24, 24)
                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                # 构建回归层（将残差变为最终预测）
                # 将残差信号从输入长度 → 映射为目标预测长度 pred_len
                # 结合 projection_layer 之后用于修正主输出
                # 将残差信号转换为实际的预测序列，长度为 pred_len
                # 继续上面例子，pred_len = 48，则：
                #
                # regression_layers 构建的线性层分别是：
                #
                # Linear(96, 48)
                #
                # Linear(48, 48)
                #
                # Linear(24, 48)
                '''
                    它们在 Model.out_projection() 函数中被调用：
                    原始残差信号 out_res → out_res_layers[i] 做变换, 
                    然后 regression_layers[i] 映射到预测长度, 最终 dec_out + out_res，实现主输出 + 残差补偿
                '''
                self.regression_layers = torch.nn.ModuleList(
                    [
                        # 构建一个残差变换层列表，每个下采样层都有一个对应的线性层。
                        # 每层结构：输入长度 → 同样长度
                        # 意味着这个线性层不是做降维或升维，而是“学习一个变换关系
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        # 注意现在仍然在__init__阶段
        # 缺失值填补 / 异常检测任务
        # 不同任务的最终输出需要什么，决定了输出层的维度，在时间序列预测任务中， 输出的形状应该和输入的序列数据形状相匹配
        # 比如：如果输入是 [B, L, C]（B=batch_size, L=序列长度, C=通道数），那么最终预测也应该是 [B, L, C]
        # 模式	输出需求	输出层设置
        # self.channel_independence == 1	每个通道自己单独预测，不考虑其他通道	每个通道只预测一个数（输出维度1）
        # self.channel_independence == 0	多个通道一起预测，有跨通道建模	预测全部通道的值（输出维度 = 通道数 configs.c_out）
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1: # 通道独立性处理，此时每个通道独立处理，输出层的维度为 1
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

        # 分类任务
        if self.task_name == 'classification':# 此时模型的最终目标是输出分类概率/得分， 而非生成时间序列
            self.act = F.gelu # 激活函数，使用 GELU
            self.dropout = nn.Dropout(configs.dropout) # Dropout 正则化，防止过拟合，随机丢弃部分特征
            # 从序列整体抽取特征后进行分类
            # 将模型编码后的整个时间序列展开（flatten）为一个长向量，再映射到分类类别数
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def out_projection(self, dec_out, i, out_res):
        """编码器输出 dec_out → projection_layer 映射
                 +
        残差信号 out_res → out_res_layer 学习变换 → regression_layer 变成预测量
                ↓
        主分支预测 + 残差预测
                ↓
        最终输出 dec_out
        """
        # 预测阶段残差修正， 在预测输出 dec_out 的基础上，利用残差信号 out_res 进行修正， 让最终预测结果更准确
        # 把编码器的输出 dec_out（形状是 [B, T, d_model]）， 投影到目标输出维度、
        # 使用上面定义的projection_layer（）
        dec_out = self.projection_layer(dec_out)
        # 把 out_res 的维度从 [B, T, C] 调换成 [B, C, T]
        # 目的是为了让下面的 Linear 层（对最后一维进行操作）能正确作用在时间步维度上。
        out_res = out_res.permute(0, 2, 1)
        # 使用第 i 个尺度对应的 out_res_layer 进行变换。
        # 注意这里输入输出长度不变。
        # 这是学习一种对残差信号的变换，比如补充模型无法直接捕获的趋势、偏移等信息。
        out_res = self.out_res_layers[i](out_res)
        # 使用第 i 个尺度对应的 regression_layer 把残差信号的时间长度，映射成预测长度 pred_len。
        # 然后再 permute 回 [B, pred_len, C]，匹配 dec_out 的形状
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        # 把主分支预测输出 dec_out 和残差修正项 out_res 相加
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list): # 在正式进入编码器前，对输入做序列分解预处理, 把每个输入序列分成趋势部分和残差（季节）部分
        if self.channel_independence == 1: # 如果每个通道是独立建模的（channel_independence == 1）, 那么不需要序列分解，直接返回原始输入 x_list
            return (x_list, None)
        else:
            # 存储分解后的主成分（out1）和趋势成分（out2）
            out1_list = []
            out2_list = []
            for x in x_list:
                # 对输入的每个尺度的序列 x，调用 self.preprocess 进行分解
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # 多尺度下采样预处理
        # 把原始输入 x_enc（以及时间戳信息 x_mark_enc），根据不同的下采样策略，生成多尺度版本，供后续网络使用
        if self.configs.down_sampling_method == 'max': # 如果配置的下采样方法是 max 池化， 进行最大池化处理
            # 创建一个 MaxPool1d 池化层。
            # 池化窗口大小 = configs.down_sampling_window
            # 不返回索引，只要最大值，取窗口内最大的特征值，强调突变特性
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)

        elif self.configs.down_sampling_method == 'avg': # 配置是 avg，走平均池化逻辑
            # 创建一个 AvgPool1d 层。
            # 同样窗口大小 = down_sampling_window。
            # 取局部区域平均，强调平滑特性。
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)

        elif self.configs.down_sampling_method == 'conv': # 配置是 conv，走卷积降采样逻辑
            padding = 1 if torch.__version__ >= '1.5.0' else 2 # 不同 PyTorch 版本卷积 padding 规则不同
            # 构建一个卷积层做下采样， 一维卷积
            # 输入输出通道都是 enc_in（不改变通道数）。
            # 核大小是3，步幅是 down_sampling_window。
            # 边界循环填充（padding_mode='circular'）。
            # 不使用偏置。
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        # 如果配置错误（不是 max/avg/conv），直接返回原数据，不处理
        else:
            return x_enc, x_mark_enc

        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        # 保存原始输入（或后续每次下采样后的新输入）
        x_enc_ori = x_enc
        # 保存原始时间戳特征
        # 时间戳timestamp： 描述每一个时间步处"发生在什么时候"的信息
        # 时间戳的格式是一个三维张量，形状为 [B, T, F]，其中：
        # B (Batch Size)：批量大小，即一次输入的样本数量。
        # T (Time Steps)：时间序列的长度。
        # F (Features)：时间戳的特征数量，例如小时、星期几、月份等。
        """ 
            在时间序列建模中，时间戳通常包括：
            小时 (hour)
            星期几 (weekday)
            月份 (month)
            季节 (season)
            是否节假日 (holiday indicator)
            
            比如你采集了一条温度变化的时间序列，一小时一条：

            时间	                温度	     时间戳特征（举例）
            2024-04-01 00:00	20.5°C	[0点, 星期一, 4月]
            2024-04-01 01:00	20.0°C	[1点, 星期一, 4月]
            2024-04-01 02:00	19.8°C	[2点, 星期一, 4月]
            ...	...	...
            这些 [小时数, 星期几, 月份]，就是时间戳特征，跟温度这个主序列配套提供的
        """
        x_mark_enc_mark_ori = x_mark_enc

        # 初始化两个空列表，用来存储多尺度下采样后的特征和时间戳。
        x_enc_sampling_list = []
        x_mark_sampling_list = []

        # 把原始输入 x_enc（再 permute回来 [B, T, C]格式）加入列表。
        # 这是第0层（原始层）
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        # 把原始时间戳 x_mark_enc也加入列表。
        # 保持时间戳和特征一一对应
        x_mark_sampling_list.append(x_mark_enc)

        # 开始多次下采样循环。
        # 总共执行 down_sampling_layers 次
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori) # 对 x_enc_ori 进行一次下采样，得到新的 x_enc_sampling

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1)) # 把新下采样后的特征也加到列表中。再次 permute回 [B, T, C]，方便后续统一处理
            x_enc_ori = x_enc_sampling # 更新 x_enc_ori，下一次循环继续下采样新的输入

            if x_mark_enc_mark_ori is not None: # 如果时间戳特征存在
                # 对时间戳特征做下采样（步长取样）
                # 保证时间步和特征同步减少
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                # 更新时间戳特征，供下一次使用
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list # 最后，把特征集合赋值给 x_enc

        # 把时间戳集合赋值给 x_mark_enc
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 其实这就是长短期预测的forward过程了
        # 输入参数：
        # N是特征维度, 即每个时间步包含的主特征数量。例如，时间序列的温度、湿度、风速等。
        # F是表示时间特征的维度，通常是与时间戳相关的辅助信息数量。例如，小时、星期几、月份等。
        # 前两者用于编码器
        # x_enc----【B，T，N】 历史输入数据，提供模型的的历史输入序列
        # x_mark_enc----【B，T，F】 历史时间特征，包含时间戳的相关辅助信息，如小时、星期几、月份等
        # 后两者用于解码器
        # x_dec----【B，T_dec，N】 未来时间序列的占位符(一般是0), 用于解码器的输入， T_dec 是预测序列的长度
        # x_mark_dec----【B，T_dec，F】 未来时间特征，包含时间戳的相关辅助信息，如小时、星期几、月份等
        # 未来时间特征处理
        # 如果模型启用未来时间特征（如日历信息、周期特征等），则首先对解码器的时间特征 x_mark_dec 进行嵌入处理
        if self.use_future_temporal_feature:
            if self.channel_independence == 1: # 如果各个通道独立
                B, T, N = x_enc.size() # B是样本数， T是时间步数， N是通道数（特征数）
                # 假设 B=2, T_dec=24, 特征维度=F, N=3，原始 x_mark_dec 为 [2,24,F]，重复后变成 [6,24,F]（即把未来时间特征复制给每个序列通道）
                # 沿着批次维度复制时间特征，
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec) # 将时间特征映射到隐藏维度
            else:
                # 如果每个通道不独立，直接将时间特征映射到隐藏维度
                # x_mark_dec是每个未来时间步的时间特征, 比如比如未来48小时里的 [小时数、星期几、是否节假日] 等
                # 告诉模型某天是周末还是工作日, 某个时刻是凌晨还是中午
                # 所有特征通道（比如温度、湿度、风速）一起建模。也就是说，未来时间特征 是所有通道 统一用的，不需要给每个通道单独一份
                # 直接拿未来时间特征 x_mark_dec，通过 enc_embedding 映射成隐藏向量 [B, T_dec, d_model]

                # 这里enc_embedding 是嵌入层，专门把输入数据映射到统一的隐藏空间 d_model 维度
                # 没有原始序列只有时间特征也能嵌入出高维表示
                # 每一条样本、每一个未来时间步，都有一个 d_model 维度的隐藏向量
                # B	Batch Size	一次喂入的样本数量
                # T_dec	Future Time Steps	要预测的未来步数，比如48步
                # d_model	Hidden Dimension	嵌入后的隐藏空间特征维度，比如64、128
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec) # [B*N, T_dec, d_model]

        # 多尺度输入处理, 就是刚开始输入的原始序列进行多轮下采样得到不同颗粒度的序列
        # 对输入序列进行多尺度下采样处理。如果未启用下采样方法则直接返回原始输入
        # 根据不同的下采样策略，生成多尺度版本
        # 有最大池化， 平均池化等
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        # 归一化与通道独立处理
        # 接下来对每个尺度的序列进行归一化处理，并根据通道独立情况调整张量形状。
        # 代码中创建了空列表 x_list, x_mark_list，然后遍历各尺度
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size() # x.size() 得到 [B, T, N]
                x = self.normalize_layers[i](x, 'norm') # 调用 self.normalize_layers[i](x, 'norm') 对每个尺度的历史序列做归一化处理（通常是RevIN等，去均值除方差），以稳定训练过程。

                # 如果通道独立，则将 x 从 [B, T, N] 转置到 [B, N, T]，再 reshape 成 [B*N, T, 1]
                # 即把所有序列通道展平到批量维。这时每个“样本”对应原来的一个序列通道。
                # 对应地，将该尺度的时间特征 x_mark 通过 repeat(N,1,1) 复制 N 份，与展平后的 x 对齐，
                # 例如 [2, 48, F]→[6, 48, F]
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)

                # 如果不通道独立，上述步骤跳过，直接将 [B, T, N] 保留为 [B, T, N] 传入下一步
                # 这样的设计允许模型在通道独立模式下对每条序列单独建模，而非混合多条序列的信息；
                # 对应地，x_list 和 x_mark_list 列表分别存储每个尺度处理后的输入张量。
                # 举例：如果 B=2, T=96, N=3，处理后 x_list 中的某尺度元素形状可能变为 [6, 96, 1]（平坦批量为6）或保持 [2,96,3]
                x_list.append(x)
                x_mark_list.append(x_mark)


        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        # 嵌入层
        # 嵌入层的作用是将原始序列和时间特征融合到统一的隐藏空间，便于后续的MLP块处理
        enc_out_list = []

        # 初次分解, 一个序列分解器, 调用的是autoformer的series_decomp
        # 把原始时间序列 x，分成：
        # trend-cyclical（长期趋势部分）
        # seasonal（短期季节波动）

        # 是 TimeMixer 在进入 PastDecomposableMixing (PDM) 编码器之前的一个预处理函数。
        # 对输入的每一个尺度的序列进行初步的“趋势+季节性”分解，
        # 让后续的 PDM 编码器能从一开始就处理"干净"、"清晰结构化"的特征。
        x_list = self.pre_enc(x_list) # 对每个尺度的数据做分解处理，通道独立时候返回原list

        # 根据是否有时间特征分别进行编码嵌入
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                #  self.enc_embedding 是一个数据嵌入模块（通常包括线性层和 dropout），将原始值和时间特征映射到隐藏表示
                # 若通道独立，则 x_list[0] 即原始列表，嵌入后的 enc_out 形状为 [B*N, T, d_model]
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                # 若不独立，则形状为 [B, T, d_model]
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)
        # 嵌入完成后，enc_out_list 是一个列表，包含每个尺度嵌入后的历史表示，各元素形状均为 [batch’, seq_len, d_model]

        # Past Decomposable Mixing
        # 过去可分解混合（PDM）编码
        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer): # 表示应用若干个 PDM 块来处理过去信息。每个 pdm_blocks[i] 接受多尺度的嵌入表示列表 enc_out_list
            # 分开对季节性和趋势成分进行混合
            # 并在尺度之间进行“自下而上”和“自上而下”的信息交互
            # 经过一个 PDM 块后，输出仍然是同样结构的列表，元素形状不变。
            # 层叠多个 PDM 块可以让模型递进地聚合细粒度的季节性信息和粗粒度的趋势信息
            enc_out_list = self.pdm_blocks[i](enc_out_list)


        # 未来多预测者混合（FMM）解码及输出
        # Future Multipredictor Mixing as decoder for future

        # self.future_multi_mixing 实现了 FMM 的解码过程，它对每个尺度的编码结果独立应用一个预测器（线性层）
        # 对于每个尺度的编码输出 enc_out，首先用 self.predict_layers[i] 将其在时间维度从序列长度映射到预测长度 h，并重排回 [B', h, d_model]。
        # 如果启用了未来时间特征，则在预测结果上加上前面嵌入得到的 self.x_mark_dec（形状匹配，参考步骤1），使预测能够利用未来的时间信息。
        # 再通过 projection_layer（若通道独立则为线性映射到标量，否则映射到多变量输出）生成最终预测。该输出在通道独立模式下形状为 [B*N, h, 1]，重塑并转置后得到 [B, h, N]；在非独立模式下输出直接为 [B, h, N]。
        # 每个尺度会生成一个预测张量，共有 len(x_list) 个预测器。之后代码 torch.stack(dec_out_list, dim=-1).sum(-1) 将这些尺度预测沿新维度堆叠并求和，实现对多尺度预测结果的融合
        # 最终得到的 dec_out 形状为 [B, h, N]。
        # 最后调用 self.normalize_layers[0](dec_out, 'denorm') 对输出做反归一化（对应之前的 norm），还原原始数据的尺度。返回的 dec_out 即为模型的预测值
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        # 该函数用来处理未来预测的多尺度混合
        # B：batch size（样本数）
        # enc_out_list：编码器输出的多尺度特征列表，每个元素形状大致是 [B, T, d_model]
        # x_list：
        # 如果是通道独立模式 channel_independence==1，是一个简单的 list
        # 如果通道共享模式，是一个 tuple，包含 (season_list, trend_list) 两部分（季节和趋势）
        # 根据不同尺度的编码器输出（enc_out_list）进行未来预测，融合各尺度信息，输出每一层预测结果 dec_out_list
        # 输入：多个尺度的编码后的序列表示
        # 输出：对应每个尺度独立预测的未来值（预测 pred_len 步）
        dec_out_list = [] # 初始化输出列表

        # 第一种情况, 各个通道相互独立
        if self.channel_independence == 1:
            x_list = x_list[0] # x_list 是一个 tuple (out1_list, out2_list)。取出 out1_list，即季节性成分

            # 遍历每个尺度，做未来预测
            # 遍历每个尺度对应的编码输出 enc_out 和尺度索引 i
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                # 使用预测器进行时间维变换
                # enc_out.permute(0,2,1)：把 [B, T, d_model] 变成 [B, d_model, T]
                # 原因：nn.Linear 通常默认最后一维是要做线性变化的
                # self.predict_layers[i]：是一个 Linear 层
                # 把原来的时间步数（如48、24等）映射成预测长度 pred_len！
                # 输入是 [B, d_model, T]
                # 输出是 [B, d_model, pred_len]
                # 再 permute(0,2,1)回来：[B, pred_len, d_model]
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                # 如果启用了未来时间特征，则将其加到预测结果上
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)

                # 由于通道独立模式下 Batch 可能展开过（B*N），这里reshape回正确的 [B, pred_len, N] 形状
                # 最终每条输出形状统一为 [B, pred_len, N]
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                # 把当前尺度预测的结果加入到列表里
                dec_out_list.append(dec_out)

        else:# 第二种情况, 各个通道间不独立
            # 遍历每个尺度，同时拿到 trend 成分
            # 遍历每个尺度，同时取：
            # enc_out ：编码输出    out_res ：趋势成分（在非通道独立模式下需要一起建模）
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                # 线性映射预测未来同样是：
                # 先 permute
                # 过线性层 predict_layers[i]
                # permute回来
                # 输出形状是 [B, pred_len, d_model]
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension

                # 调用 out_projection() 方法
                # 把预测结果 dec_out 和 trend 残差 out_res 结合起来
                # 进一步修正预测结果（补充长期趋势信息）
                dec_out = self.out_projection(dec_out, i, out_res)
                # 收集这一尺度的最终预测输出
                dec_out_list.append(dec_out)
        # 返回所有尺度的预测列表，每个元素形状统一是[B, pred_len, N]
        return dec_out_list

    def classification(self, x_enc, x_mark_enc):

        """
            输入 x_enc [B, T, N]
                ↓
            多尺度下采样 → [x0, x1, x2]  每个 [B, T_i, N]
                ↓
            embedding → enc_out_list[i]: [B, T_i, d_model]
                ↓
            PDM编码器（多层） → 更新后的 enc_out_list
                ↓
            取 enc_out_list[0] 作为主尺度
                ↓
            激活 + dropout → 保留有效时间步（mask）
                ↓
            拉平 → [B, T*d_model]
                ↓
            projection → [B, num_classes]
        """
        # 分类任务 的前向过程, 如预测未来行为属于哪一类、异常属于哪种类型
        # 调用多尺度处理模块，对原始输入 x_enc 进行多尺度下采样（MaxPool / AvgPool / Conv），生成一组不同时间粒度的序列。
        # x_enc: [B, T, N]
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        # 嵌入层
        # 将每个尺度的输入 x 通过嵌入模块 self.enc_embedding，编码成高维特征表示：
        # x: [B, T_i, N] → enc_out: [B, T_i, d_model]
        # 数值特征映射到统一的 d_model 维空间，便于后续处理
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        # 经过多层 PDM 块处理
        # 对 enc_out_list 做动态分解 + 多尺度趋势/季节性融合。
        # 每一层都更新 enc_out_list 中每个尺度的 [B, T_i, d_model]
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 只取第一个尺度(通常是原始尺度)作为最终用于分类的特征序列
        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity

        # 激活 + Dropout
        # 激活函数 GELU，引入非线性，提高表达能力
        # Dropout 防止过拟合，增加模型泛化性
        output = self.act(enc_out)
        output = self.dropout(output)

        # 去除 padding 影响, x_mark_enc 是一个形状为 [B, T] 的 mask 张量，表示哪些时间步是有效的
        # 通过乘法将 padding 时间步的嵌入设为 0，防止干扰最终分类
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        # 拉平成一维向量, [B, T, d_model] 拉平成 [B, T * d_model]
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        """x_enc [B, T, N]
              ↓
            多尺度处理 → [x0, x1, ...] 每个 [B, T_i, N]
              ↓
            标准化 + reshape（视通道独立性）
              ↓
            嵌入编码 → [B, T, d_model]
              ↓
            PDM 编码器（多层）提取长期+短期模式
              ↓
            projection → 预测序列 [B, T, N]
              ↓
            反标准化（恢复真实数值）
              ↓
            → 用于异常检测（与原序列对比）
"""
        # 异常检测（Anomaly Detection）任务 的前向传播逻辑
        # 该函数用于检测输入时间序列中的异常点（outliers）。
        # 核心思想是：通过多尺度建模和趋势-季节解耦，重建输入序列，然后比较原始序列与重建序列的误差，以此判断哪些点是异常。

        # 提取输入序列x_enc的基本维度
        B, T, N = x_enc.size()
        # 调用多尺度下采样模块，对输入进行多粒度处理,_代表不用时间戳
        # 返回 x_enc 是一个多尺度列表：[x0, x1, x2, ...], 每个 xi 的形状是 [B, Ti, N]
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        # 初始化列表，存储处理后的每个尺度输入
        x_list = []


        '''
            对每个尺度的输入：
            
            做标准化（norm）：
            
            归一化每个尺度的输入，消除均值/方差差异
            
            有助于后续建模更稳定，尤其是异常检测非常依赖数值大小
            
            reshape（仅在通道独立时）：
            
            若 channel_independence == 1：
            
            把形状从 [B, T, N] → [B * N, T, 1]
            
            即：把每个通道当作一个独立序列处理
            最终 x_list 是标准化后的输入列表，形状统一为 [B', T, 1] 或 [B, T, N]'''
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        # 嵌入层
        # 将每个尺度的输入送入嵌入模块：[B, T, N] → [B, T, d_model]
        # 也可以理解为：将数值特征升维为语义空间中的特征向量
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        # 多层 PDM 编码器处理
        # 将多个尺度的嵌入特征输入多层 Past Decomposable Mixing 模块中：
        # 每一层：动态趋势-季节性分解   多尺度交叉混合（Criss-Cross Attention 替代）
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 对主尺度（通常是最细粒度）输出 enc_out_list[0]：
        # 用 projection_layer 做最后一层映射：
        # [B, T, d_model] → [B, T, N]（或 [B*N, T, 1]）
        dec_out = self.projection_layer(enc_out_list[0])
        # 把预测输出 dec_out 变回 [B, T, N] 形式，便于和原始输入对比
        # .reshape(B, N, T) → .permute(0, T, N) → [B, T, N]
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        # 把预测序列反标准化，恢复真实数值空间
        # 以便于和原始输入进行直接比较（如做残差分析）
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
