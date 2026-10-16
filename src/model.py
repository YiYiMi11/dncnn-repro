"""DnCNN 网络结构。

严格对应论文 Fig.1 / Sec.III-B：
    Conv(3x3, 3->64) + ReLU
    [Conv(3x3, 64->64) + BN + ReLU] x (D-2)
    Conv(3x3, 64->3)
    输出 = 输入 - 网络输出        （残差学习：网络学的是噪声 v，不是干净图 x）

要点（都是论文里明确强调的）：
  * 卷积前 **zero padding**，不使用 reflect/symmetric padding，也不裁图像边界。
    论文 Sec.III-B2："we directly pad zeros before convolution ... we find that
    the simple zero padding strategy does not result in any boundary artifacts."
  * 只有中间层有 BN，最后一层没有 BN；第一层没有 BN。
  * 残差学习：整个网络只预测噪声 v，估计结果 x = y - R(y)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    """DnCNN。init 可选：
        'orthogonal' : 正交初始化 gain=0.2 —— MatConvNet(cnn_init) 的默认值，
                       也是作者 PyTorch 版 KAIR 里 dncnn 用的
                       （options/train_dncnn.json: init_type=orthogonal, init_gain=0.2）。
                       权重范数比 He 小约 7 倍，因此论文的 SGD lr=1e-1 不会发散。
        'he'         : He 正交初始化（论文正文说的 "initialize by the method in [27]"）。
                       权重大约 7 倍，配 lr=1e-1 会直接发散，需配合更小的 lr。
    """

    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True,
                 kernel_size=3, init="orthogonal", init_gain=0.2):
        super().__init__()
        if depth < 2:
            raise ValueError("depth 至少为 2")
        pad = kernel_size // 2

        layers = [
            nn.Conv2d(image_channels, n_channels, kernel_size, padding=pad, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(n_channels, n_channels, kernel_size, padding=pad, bias=False)
            ]
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=pad, bias=False))

        self.layers = nn.Sequential(*layers)
        self.depth = depth
        self.init_type = init
        self.init_gain = init_gain
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_type == "orthogonal":
                    # 与 MatConvNet cnn_init 一致：权重矩阵正交化后整体缩放到 gain
                    w = torch.empty_like(m.weight)
                    nn.init.orthogonal_(w.view(m.weight.size(0), -1))
                    with torch.no_grad():
                        m.weight.copy_(w.view_as(m.weight) * self.init_gain)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """输入含噪图 y（float, [0,1] 附近），返回去噪图 x_hat。"""
        noise = self.layers(x)
        return x - noise          # 残差学习

    @torch.no_grad()
    def merge_bnorm(self):
        """把 Conv+BN 合并成单个 Conv（测试期加速，数值上应当等价）。

        设 Conv 输出 z = W·x + b_conv，BN 推理时
            y = gamma·(z − mu)/sqrt(var+eps) + beta
                = (W·g)·x + [g·(b_conv − mu) + beta],   g = gamma/sqrt(var+eps)
        所以   W' = W·g      b' = g·(b_conv − mu) + beta

        ⚠️ 常见错误是把偏置写成 (beta − mu)·g + b_conv，
        那样漏掉了 conv 自身 bias 也要被 g 缩放。Conv 没 bias 时
        b_conv = 0，此时两种写法才碰巧一致——这正是本 bug 藏了很久的原因：
        第一层 Conv 有 bias（不为 None），中间层 bias=False，
        而只有第一层会暴露问题。

        实现上**从后往前遍历**：把 layers[i+1] 换成 Identity 时，
        更小的索引不受影响，避免索引错位导致漏合并。
        """
        dev = self.layers[0].weight.device
        for i in range(len(self.layers) - 2, -1, -1):
            m, nxt = self.layers[i], self.layers[i + 1]
            if not (isinstance(m, nn.Conv2d) and isinstance(nxt, nn.BatchNorm2d)):
                continue
            w = m.weight.data
            if m.bias is None:
                # bias=False 的中间层：等价于 b_conv = 0
                m.bias = nn.Parameter(torch.zeros(w.size(0), dtype=w.dtype, device=dev))
            b_conv = m.bias.data
            g = nxt.weight / torch.sqrt(nxt.running_var + nxt.eps)
            m.weight.data = w * g.view(-1, 1, 1, 1)
            m.bias.data = g * (b_conv - nxt.running_mean) + nxt.bias
            self.layers[i + 1] = nn.Identity()
        return self


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_dncnn(path, map_location="cpu"):
    """按 checkpoint 里存的训练配置重建网络并载入权重。

    权重文件和网络的结构参数必须成对使用，不能硬编码 depth：
      * DnCNN-S（已知噪声级）用 depth=17；
      * DnCNN-B（盲去噪）用 depth=20 —— 论文 Sec. IV-A 就是这么分的。
    写死 depth=17 去载盲模型会直接 shape mismatch，而反过来（拿 S 的结构
    配 depth=17 但权重其实是 B 的）更糟，会在某处静默出错。
    checkpoint 里除了 `model`，还带一个 `args`（结构 + 训练超参），这里以它为准。
    """
    ck = torch.load(path, map_location=map_location, weights_only=False)
    a = ck.get("args", {})
    color = bool(a.get("color", False))
    model = DnCNN(
        depth=a.get("depth", 17),
        n_channels=a.get("width", 64),
        image_channels=3 if color else 1,
        init=a.get("init", "orthogonal"),
        init_gain=a.get("init_gain", 0.2),
    )
    model.load_state_dict(ck["model"])
    return model.eval(), a


if __name__ == "__main__":
    for d, c in [(17, 1), (17, 3), (20, 3)]:
        m = DnCNN(depth=d, image_channels=c)
        print(f"depth={d} in_ch={c} -> {count_params(m):,} params, "
              f"receptive field {2*d+1}x{2*d+1}")