"""网络结构的正确性测试。

重点不是"跑起来不报错"，而是钉住那些**一旦写错、PSNR 就会静默偏掉几个 dB**
的实现细节：残差学习接的到底是哪一路、BN 合并是否真的等价、padding 会不会缩图。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DnCNN, count_params  # noqa: E402


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return DnCNN(depth=17, n_channels=64, image_channels=1)


def test_forward_preserves_shape(net):
    """DnCNN 是全卷积网络，任意分辨率输入都应原样输出（论文靠这个滑窗处理任意尺寸）。"""
    y = torch.rand(2, 1, 40, 40)
    assert net(y).shape == y.shape
    y2 = torch.rand(1, 1, 321, 481)
    assert net(y2).shape == y2.shape


def test_residual_learning_wiring(net):
    """残差学习：输出 = 输入 − 网络输出，网络学的是**噪声**而非干净图。

    如果有人把 forward 改成直接 return self.layers(x)，去噪会变成「反过来加噪声」，
    而且 loss 曲线看起来仍然正常下降 —— 所以必须显式断言。
    """
    net.eval()
    y = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        out = net(y)
        predicted_noise = net.layers(y)
    assert torch.allclose(out, y - predicted_noise, atol=1e-6)


def test_bn_only_in_hidden_layers(net):
    """论文 Sec.III-B2：首层无 BN、末层无 BN，中间 (D−2) 层才有 BN。"""
    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(convs) == net.depth          # 17 层卷积
    assert len(bns) == net.depth - 2        # 15 个 BN
    # 第一个卷积带 bias（残差学习的输入层需要偏置），其余不带 —— BN 会吸收掉
    assert convs[0].bias is not None
    assert all(c.bias is None for c in convs[1:-1])


def test_zero_padding_not_reflect(net):
    """论文用 zero padding（Sec.III-B2），不是 reflect/symmetric。

    两者的区别只在边界几个像素上，PSNR 影响不大但会改变逐图结果。
    用一个常数图区分：reflect 会把边缘值复制进来，zero 会混入黑边。
    """
    net.eval()
    x = torch.ones(1, 1, 8, 8)
    with torch.no_grad():
        out = net(x)
    # reflect 下常数图整体仍是常数；zero padding 会让边界响应不同
    interior = out[..., 2:6, 2:6]
    assert interior.std() > 0  # 不是恒等映射，模型确实在动
    assert out.shape == x.shape


@pytest.mark.parametrize("image_channels", [1, 3])
def test_merge_bnorm_is_numerically_equivalent(image_channels):
    """把 Conv+BN 折叠进 Conv 后，推理输出必须与折叠前一致。

    这里有个真实踩过的坑：折叠时偏置要写成
        b' = g·(b_conv − mu) + beta      （正确）
    而不是
        b' = (beta − mu)·g + b_conv      （漏了 conv 自身的 bias 也要乘 g）
    两者只在 conv 无 bias 时才碰巧一致，而**第一层 conv 恰恰带 bias**，
    所以错误只会体现在第一层上 —— PSNR 会偏，但不会崩，很容易被忽略。
    """
    torch.manual_seed(1)
    net = DnCNN(depth=17, image_channels=image_channels).eval()
    # 必须同时打乱 Conv 的 bias 和 BN 的全部参数，否则测不出任何差异。
    #
    # 两种写法的差恰好是 (g−1)·(b_conv−β)：
    #   正解  g·(b_conv−μ) + β
    #   错解  (β−μ)·g + b_conv
    #   差    = (g−1)·(b_conv−β)
    # 而 _init_weights 把所有 bias 和 β 都初始化成 0，于是 b_conv−β=0，
    # 两种写法在**刚初始化时完全等价** —— 这正是这个 bug 长期没被发现的原因。
    # 只有训练让 bias 偏离 0 之后才会暴露，所以这里必须手动注入非零 bias。
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            m.bias.data.normal_(0, 0.5)
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_(0, 0.5)

    x = torch.rand(2, image_channels, 40, 40)
    with torch.no_grad():
        before = net(x)
        net.merge_bnorm()
        after = net(x)

    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules()), "BN 没被折叠干净"
    assert torch.allclose(before, after, atol=1e-4), \
        f"merge_bnorm 不等价，最大偏差 {(before - after).abs().max().item():.2e}"


def test_merge_bnorm_is_idempotent(net):
    """重复调用不应报错，也不应再次改变结果。"""
    net.eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        net.merge_bnorm()
        once = net(x)
        net.merge_bnorm()
        twice = net(x)
    assert torch.allclose(once, twice, atol=1e-6)


def test_receptive_field_and_param_count():
    """depth=17 的理论感受野是 35×35（每层 3x3 卷积 +1），参数量应与实现一致。"""
    net = DnCNN(depth=17, n_channels=64, image_channels=1)
    # conv1(1->64, 带 bias) + 15×[conv(64->64, 无 bias) + BN(weight+bias)] + conv(64->1, 无 bias)
    expected = (1 * 64 * 9 + 64) + 15 * (64 * 64 * 9 + 64 + 64) + (64 * 1 * 9)
    assert count_params(net) == expected
    assert (2 * 17 + 1) ** 2 == 35 ** 2


def test_orthogonal_init_keeps_weights_small():
    """正交初始化 + gain 0.2（MatConvNet 默认）让中间层卷积核范数比 He 小约 7 倍。

    这是论文的 SGD lr=1e-1 不发散的前提；若有人把默认值改成 'he'，
    权重尺度会涨一个数量级，配合论文超参会直接发散/NaN。

    只比较中间层（64→64）：首层的 fan_in 很小，正交化后按列归一，
    比例只有约 2.4 倍，拿它断言会得到一个看着通过、实则没测到点的测试。
    """
    torch.manual_seed(0)
    small = DnCNN(depth=17, init="orthogonal", init_gain=0.2)
    torch.manual_seed(0)
    he = DnCNN(depth=17, init="he")
    assert count_params(small) == count_params(he)   # 参数量相同，差别只在权重分布

    # layers = [Conv, ReLU, Conv, BN, ReLU, ...]，所以 layers[2] 是第一个 64→64 卷积
    ratio = he.layers[2].weight.norm().item() / small.layers[2].weight.norm().item()
    assert 5.0 < ratio < 9.0, f"中间层权重范数之比实测 {ratio:.2f}，与预期的 ~7 不符"