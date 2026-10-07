"""显式的教学顺序：Notebook 的叙事、源码与实验都在这里声明。"""

SETUP_FIRST = '''from pathlib import Path
import random
import sys

root = Path.cwd()
if not (root / "slm").is_dir():
    root = next((parent for parent in root.parents if (parent / "slm").is_dir()), None)
if root is None:
    raise RuntimeError("请从仓库根目录或 notebooks 目录运行")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)'''

SETUP_LATER = '''from pathlib import Path
import sys

root = Path.cwd()
if not (root / "slm").is_dir():
    root = next((parent for parent in root.parents if (parent / "slm").is_dir()), None)
if root is None:
    raise RuntimeError("请从仓库根目录或 notebooks 目录运行")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from slm import set_seed, show

torch.set_num_threads(1)
set_seed(0)
show("Python", sys.version.split()[0])
show("PyTorch", torch.__version__)
show("device", "cpu")'''

PPO_LOOP = '''set_seed(0)
env = CartPole(max_steps=100)
agent = ActorCritic(4, 2)
optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
state, episode, history = env.reset(), 0.0, []
for update in range(8):
    buffer = {name: [] for name in ('obs','actions','log_probs','values','rewards','dones')}
    for _ in range(128):
        obs = torch.tensor(state)
        with torch.no_grad():
            dist, value = agent(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        state, reward, terminated, truncated = env.step(action.item())
        done = terminated or truncated
        for name, value in zip(buffer, (obs,action,log_prob,value,reward,float(done))):
            buffer[name].append(value)
        episode += reward
        if done:
            history.append(episode)
            state, episode = env.reset(), 0.0
    obs, actions, old_probs, values = [torch.stack(buffer[name]) for name in ('obs','actions','log_probs','values')]
    rewards, dones = torch.tensor(buffer['rewards']), torch.tensor(buffer['dones'])
    with torch.no_grad():
        last = agent(torch.tensor(state))[1]
    advantages, returns = compute_gae(rewards, values, dones, last, 0.99, 0.95)
    for _ in range(2):
        for indices in torch.randperm(128).split(64):
            loss, info = ppo_loss(agent,obs[indices],actions[indices],old_probs[indices],advantages[indices],returns[indices])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
    show(f'update {update+1}', f'loss {loss.item():.3f} | clip {info["clip_frac"]:.3f}')
plt.plot(range(1,len(history)+1), history)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Short PPO training: mechanics, not a solved policy')
plt.show()'''

RLHF_LOOP = '''set_seed(0)
policy = TinyLM(8, hidden=32)
reference = copy.deepcopy(policy).eval().requires_grad_(False)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
records = []
for iteration in range(12):
    tokens = policy.generate(32, 4)
    with torch.no_grad():
        logits, values = policy(tokens)
        old_probs = token_log_probs(logits, tokens)
        ref_probs = token_log_probs(reference(tokens)[0], tokens)
    scores = (tokens == 3).float().mean(-1)
    kl = old_probs - ref_probs
    rewards = -0.1 * kl
    rewards[:, -1] += scores
    dones = torch.tensor([0.,0.,0.,1.])
    pairs = [compute_gae(rewards[i],values[i],dones,torch.tensor(0.),1.,0.95) for i in range(32)]
    advantages, returns = [torch.stack(items) for items in zip(*pairs)]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for _ in range(2):
        logits, new_values = policy(tokens)
        ratio = (token_log_probs(logits,tokens)-old_probs).exp()
        surrogate = torch.minimum(ratio*advantages, ratio.clamp(0.8,1.2)*advantages)
        loss = -surrogate.mean() + 0.5*F.mse_loss(new_values,returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    records.append((scores.mean().item(),kl.sum(-1).mean().item()))
    show(f'RLHF iteration {iteration+1}', f'score {records[-1][0]:.3f} | sampled KL {records[-1][1]:.3f}')
fig, axes = plt.subplots(1,2,figsize=(9,3))
for axis, index, label in zip(axes,(0,1),('Reward','Sampled sequence KL')):
    axis.plot(range(1,len(records)+1), [row[index] for row in records])
    axis.set(xlabel='Iteration',ylabel=label)
plt.tight_layout()
plt.show()'''

CHAPTERS = [
    {
        'slug': '00_transformer',
        'title': 'EP.0 Transformer',
        'cells': [
            ('markdown', '''# EP.0 Transformer：从张量形状到完整 Encoder–Decoder

配套文章：[Transformer 的数学表示与代码实现](https://momoyeyu.github.io/posts/llm-transformer/)。文章负责完整数学推导；本 Notebook 负责把每个公式落成可执行代码，并逐步核对形状、数值与边界。这里首次出现的实现会直接展开真实源码，而不是隐藏在导入中。

学习路线：缩放点积注意力 → 多头 → FFN → mask → embedding/位置 → Encoder/Decoder → 完整 Transformer。下一章见 [01_moe.ipynb](01_moe.ipynb)。'''),
            ('code', SETUP_FIRST),
            ('markdown', '## 可复用实验工具\n\n先展开随机种子与统一输出函数。后续章节会把它们当作已经学过的公共工具导入。'),
            ('source', ('slm/common.py', ('set_seed', 'show'))),
            ('code', '''set_seed(0)
show("Python", sys.version.split()[0])
show("PyTorch", torch.__version__)
show("device", "cpu")'''),
            ('markdown', '''## 1. 缩放点积注意力：为什么除以 $\\sqrt{D_k}$？

给定 $Q\\in\\mathbb{R}^{B\\times N\\times D_k}$、$K\\in\\mathbb{R}^{B\\times M\\times D_k}$、$V\\in\\mathbb{R}^{B\\times M\\times D_v}$：

$$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(QK^\\top/\\sqrt{D_k})V.$$

softmax 必须沿最后一个 key 轴 $M$ 做；输出形状为 $[B,N,D_v]$。缩放用于控制随机点积随维度增长的量级。'''),
            ('source', ('slm/transformer.py', ('attention',))),
            ('demo', 'scripts/transformer/01_attention.py'),
            ('markdown', '''输出应显示：未缩放 logits 的标准差随 $\\sqrt{D_k}$ 增长，缩放后接近同一量级；每行权重和为 1。它是初始化分布下的量级检查，不是对所有训练状态的界。'''),
            ('markdown', '''## 2. 多头注意力：分头不等于增加总投影矩阵

每个头把 $D_{model}$ 投影到 $D_k=D_v=D_{model}/H$，拼接后再经输出投影。合起来仍对应 Q/K/V/O 四个 $D_{model}\\times D_{model}$ 矩阵，因此在这个实现里改变头数不会改变这些矩阵的总参数量或主矩阵乘 MAC；这不意味着头数不影响表达或准确率。'''),
            ('source', ('slm/transformer.py', ('MHA',))),
            ('demo', 'scripts/transformer/02_multi_head.py'),
            ('markdown', '演示把每个头单独计算后拼接，结果与 `MHA.forward` 一致；输出形状恢复为 `[B,N,D_model]`。'),
            ('markdown', '''## 3. FFN：逐 token 的非线性通道变换

注意力在 token 轴混合信息；FFN 对每个 token 独立复用同一 MLP，在通道轴做 $D_{model}\\to D_{ff}\\to D_{model}$。默认 ReLU 保持本章 Transformer 行为，后续 MoE 会复用同一类并换成 GELU。'''),
            ('source', ('slm/transformer.py', ('FFN',))),
            ('code', '''x = torch.randn(2, 5, 16)
show('FFN shape', tuple(FFN(16, 32)(x).shape))'''),
            ('markdown', '输入输出都是 `[2,5,16]`：FFN 不改变序列长度，也不在不同 token 间通信；非线性使它不退化为单个线性层。'),
            ('markdown', '''## 4. Mask：把不可见位置变成 $-\\infty$

causal mask 阻止位置 $i$ 读取未来 $j>i$；padding mask 阻止读取补齐 key。加到 logits 后，softmax 把这些位置变成 0。实现假设每个参与评估的 query 至少有一个可见 key；整行全为 $-\\infty$ 会使 softmax 无定义。'''),
            ('source', ('slm/transformer.py', ('_NEG_INF', 'causal_mask', 'padding_mask'))),
            ('demo', 'scripts/transformer/03_masked_attention.py'),
            ('markdown', '因果矩阵第一行只能看自己；修改最后一个 key/value 不会影响更早 query。padding key 的最大权重为 0。'),
            ('markdown', '''## 5. Token、位置与分类头

`Embedding` 把 id `[B,N]` 变为 `[B,N,D]` 并乘 $\\sqrt D$。正弦位置编码交替使用 $\\sin$/$\\cos$，与 token embedding 相加。`LinearClassifier` 把每个位置映射回词表；传入 embedding 权重即可 weight tying。'''),
            ('source', ('slm/transformer.py', ('Embedding', 'PositionEncoding', 'LinearClassifier'))),
            ('code', '''pe = PositionEncoding(16, max_len=8)
show('PE first 3 rows', pe(torch.zeros(1, 3, 16))[0])
emb = Embedding(20, 16)
head = LinearClassifier(16, 20, emb.w_e.weight)
show('tied weight identity', head.classifier.weight is emb.w_e.weight)'''),
            ('markdown', '前三行直接显示位置 0/1/2 的 sin/cos 模式。缓存之外仍可按公式计算位置，但“能计算”不等于模型在未训练长度上必然泛化。'),
            ('markdown', '''## 6. Encoder：self-attention + FFN

每个子层采用残差后 LayerNorm：$x\\leftarrow\\mathrm{LN}(x+F(x))$。这里把 `nn.LayerNorm` 当作 PyTorch 基元；其内部统计量会在 [02_normalization.ipynb](02_normalization.ipynb) 从零实现。'''),
            ('source', ('slm/transformer.py', ('EncoderBlock', 'Encoder'))),
            ('markdown', 'Encoder 的 self-attention 使用同一序列作 Q/K/V；每层保持 `[B,src_len,D]`，mask 可广播到每个 query。'),
            ('markdown', '''## 7. Decoder：masked self-attention + cross-attention

Decoder 先在目标序列上做 causal self-attention，再做 cross-attention：Q 来自目标，K/V 来自 Encoder。因而 `tgt_len` 与 `src_len` 可以不同，cross logits 形状是 `[B,tgt_len,src_len]`。'''),
            ('source', ('slm/transformer.py', ('DecoderBlock', 'Decoder'))),
            ('markdown', '三个残差子层依次是目标自注意力、源目标交叉注意力、逐 token FFN；每个阶段都回到目标形状。'),
            ('markdown', '''## 8. 完整 Transformer 与 teacher forcing

完整模型组合 embedding、位置、Encoder、Decoder 和 tied 分类头。训练时目标序列整体输入，但 causal mask 保证位置 $t$ 只能使用不晚于 $t$ 的输入；`logits[:,t]` 对齐下一个目标 `tgt[:,t+1]`。padding 标签必须从 loss 中忽略。'''),
            ('source', ('slm/transformer.py', ('Transformer',))),
            ('demo', 'scripts/transformer/04_transformer.py'),
            ('markdown', '演示只对随机初始化 logits 计算一次 teacher-forcing loss，并不代表模型已经学会翻译。输出同时验证 tied 权重身份、logits 形状与 padding-aware 标签移位。'),
        ],
    },
    {
        'slug': '01_moe',
        'title': 'EP.1 MoE',
        'cells': [
            ('markdown', '''# EP.1 Mixture of Experts：路由、容量与共享专家

配套文章：[MoE 技术原理](https://momoyeyu.github.io/posts/llm-moe/)。上一章：[00_transformer.ipynb](00_transformer.ipynb)；下一章：[02_normalization.ipynb](02_normalization.ipynb)。本章复用上一章首次展开的 `FFN`，以及其中定义的 `set_seed` / `show` 实验工具，把 token-wise FFN 扩展成稀疏专家池。核心问题不是“多放几个 MLP”，而是：每个 token 选谁、如何平衡、溢出如何处理。'''),
            ('code', SETUP_LATER + '\nfrom slm import FFN'),
            ('markdown', '## 1. Dense 与专家 FFN 的矩阵乘成本\n\n先只数两个线性层的乘加：$2D_{model}D_{ff}$。'),
            ('source', ('slm/moe.py', ('ffn_macs',))),
            ('demo', 'scripts/moe/01_dense_vs_moe.py'),
            ('markdown', '这些数只覆盖专家内部 dense matmul MAC，不含 router、bias、dispatch、通信或 padding。所谓“等参拆分”也不是整个模型参数量严格相等。'),
            ('markdown', '''## 2. Router：概率、Top-K 索引与门权重

Router 先产生所有专家概率 `[T,N]`，再取每个 token 的 Top-K 索引 `[T,K]`，并在选中项内重新归一化权重。'''),
            ('source', ('slm/moe.py', ('Router',))),
            ('demo', 'scripts/moe/02_router.py'),
            ('markdown', '每行概率和与选中权重和都为 1。特别地 K=1 时，重归一化后的主分支门权重恒为 1，主损失不能通过这个标量训练选择；辅助负载损失因而很重要。'),
            ('markdown', '''## 3. 负载均衡损失

$f_i$ 是被 Top-K 选中专家 $i$ 的 token 比例，$P_i$ 是其平均路由概率：

$$L_{aux}=\\alpha N\\sum_i f_iP_i.$$

每个 token 选择 K 个专家，所以 $\\sum_i f_i=K$；均衡时损失是 $\\alpha K$，不是 $\\alpha$。'''),
            ('source', ('slm/moe.py', ('load_balancing_loss',))),
            ('demo', 'scripts/moe/03_load_balance.py'),
            ('markdown', 'Top-K 的离散选择本身不可微；对概率元素的梯度是 $\\partial L/\\partial p_{t,i}=\\alpha N f_i/T$。这里人为固定 Top-K 索引以隔离梯度路径，索引并非由这组随机概率选出；不要把两个损失数值解读为均衡状态的全局最小值。空载专家在这个固定 dispatch mask 下梯度为零。'),
            ('markdown', '''## 4. Expert Capacity 与一次统一 dispatch

容量 $C=\\lceil TK/N\\cdot CF\\rceil$ 是每个专家跨所有 Top-K 槽位共享的总预算。dispatch 按 slot、expert、token 顺序接收，因而较早槽位与较早 token 有优先级；丢弃后不重新归一化剩余权重。'''),
            ('source', ('slm/moe.py', ('expert_capacity', 'dispatch', 'MoELayer'))),
            ('demo', 'scripts/moe/04_capacity.py'),
            ('markdown', '输出展示每个专家总调用数不超过容量，以及完全溢出的 token。MoE 分支为零不等于整层输出必须为零：真实 Transformer 外部通常还有 residual。'),
            ('markdown', '''## 5. Shared + Routed：教学版 DeepSeekMoE

共享专家处理全部 token，路由专家仍受 Top-K 与容量控制。这里把总宽度按 routed+shared 专家数拆分，单专家宽度为 `d_ff/(routed+shared)`。'''),
            ('source', ('slm/moe.py', ('DeepSeekMoELayer',))),
            ('demo', 'scripts/moe/05_deepseek_moe.py'),
            ('markdown', 'hook 计数验证 shared 专家始终看见全部 token，并在使用后显式移除。这个小模型只讲机制，不是生产 DeepSeek 架构的完整复刻。'),
        ],
    },
    {
        'slug': '02_normalization',
        'title': 'EP.2 Normalization',
        'cells': [
            ('markdown', '# EP.2 归一化：统计轴、残差位置与 QK 尺度\n\n配套文章：[一文搞懂归一化技术](https://momoyeyu.github.io/posts/llm-normalization/)。上一章：[01_moe.ipynb](01_moe.ipynb)；下一章：[03_position.ipynb](03_position.ipynb)。本章沿用 [EP.0](00_transformer.ipynb) 定义的 `set_seed` / `show`，从统计量的“沿哪些轴计算”出发，实现 BatchNorm、LayerNorm、RMSNorm，并比较归一化放在残差前后时的结构差异。'),
            ('code', SETUP_LATER + '\nimport time'),
            ('markdown', '## 1. 残差流为何会漂移？\n\n若每层叠加独立、零均值、方差 0.01 的扰动，则 $\\mathrm{Var}(x_L)=1+0.01L$，标准差是其平方根。'),
            ('demo', 'scripts/normalization/01_residual_drift.py'),
            ('markdown', '观测值应接近独立噪声公式；真实 learned residual 往往相关且非零均值，所以这不是深网残差尺度的普遍定理。'),
            ('markdown', '## 2. BatchNorm：跨 N、L 的通道统计\n\n对 `[N,L,C]`，每个通道在 `(N,L)` 上统计。训练方差用于当前输出时取 biased 估计；running variance 按 PyTorch 惯例更新 unbiased 估计。'),
            ('source', ('slm/norm.py', ('BatchNorm',))),
            ('demo', 'scripts/normalization/02_batchnorm.py'),
            ('markdown', 'train/eval 使用不同统计量，padding 零值会污染 batch 统计。本教学实现只处理 `[N,L,C]`，训练时 unbiased running variance 至少需要两个观测。'),
            ('markdown', '## 3. LayerNorm：每个 token 自己归一化\n\nLayerNorm 只沿最后通道 C 统计，因此其他 token 是否 padding 不改变当前 token。'),
            ('source', ('slm/norm.py', ('LayerNorm',))),
            ('demo', 'scripts/normalization/03_layernorm.py'),
            ('markdown', '平移会同时平移均值，因此理想算术下平移不变；正比例缩放只在忽略 `eps` 时严格不变。负比例会翻转标准化部分的符号。'),
            ('markdown', '## 4. RMSNorm：只保留均方根缩放\n\nRMSNorm 不减均值，也没有 beta，只学习逐通道 gamma；统计量用 fp32 计算后转回输入精度。'),
            ('source', ('slm/norm.py', ('RMSNorm',))),
            ('demo', 'scripts/normalization/04_rmsnorm.py'),
            ('markdown', '对拍验证公式。计时仅是当前 CPU、eager、forward 的局部测量；不能据此声称 RMSNorm 在所有设备、编译器或训练负载上更快。'),
            ('markdown', '## 5. Pre-Norm 与 Post-Norm\n\nPost：`norm(x + F(x))`；Pre：`x + F(norm(x))`。复用刚实现的 RMSNorm，避免把核心机制藏在原生层中。'),
            ('source', ('slm/norm.py', ('PostNormBlock', 'PreNormBlock'))),
            ('demo', 'scripts/normalization/05_pre_post_norm.py'),
            ('markdown', 'EP.0 的 Transformer 使用 LayerNorm，而此实验使用 RMSNorm；归一化“放在哪里”的思想与具体 norm 类型独立。真实 Pre-Norm 堆栈通常还需最终 norm。单个 seed、深度和学习率不能推出哪种结构总是更稳定。'),
            ('markdown', '## 6. QK-Norm：直接控制 attention logits\n\n在 Q/K 投影后逐 head 归一化，再做点积。这里不提前依赖下一章 RoPE。'),
            ('demo', 'scripts/normalization/06_qk_norm.py'),
            ('markdown', 'gamma 初始为 1 时，每个向量模长约为 $\\sqrt D$，缩放点积绝对值上界为 $\\sqrt D$。gamma 学习后会改变这个界。下一章会把 QK-Norm 与 RoPE 真正串起来。'),
        ],
    },
    {
        'slug': '03_position',
        'title': 'EP.3 Position',
        'cells': [
            ('markdown', '# EP.3 位置编码：从等变性到 RoPE 与长度缩放\n\n配套文章：[位置编码技术的演进](https://momoyeyu.github.io/posts/llm-position-encoding/)。上一章：[02_normalization.ipynb](02_normalization.ipynb)；下一章：[04_ppo.ipynb](04_ppo.ipynb)。本章复用 [EP.0](00_transformer.ipynb) 首次展开的 `MHA`、`PositionEncoding`、`attention`、`causal_mask` 与 `set_seed` / `show`，逐步加入绝对、相对、线性 bias 与旋转位置。重点区分“公式可计算”“结构有位置线索”和“模型能泛化”三件事。'),
            ('code', SETUP_LATER + '\nfrom types import SimpleNamespace\nfrom slm import MHA, PositionEncoding, attention, causal_mask'),
            ('markdown', '## 1. 没有位置时的 permutation equivariance\n\n重排输入 token 后，输出按相同方式重排，这叫等变，不是输出完全不变的 invariance。'),
            ('demo', 'scripts/position/01_permutation_equivariance.py'),
            ('markdown', '实验只验证无 mask MHA 的结构性质；加入位置向量后，同一内容被分配到不同位置向量，因此不再满足同样等式。'),
            ('markdown', '## 2. 可学习绝对位置与正弦位置\n\nLearnedPE 查表，超过 `max_len` 会越界；正弦位置可在缓存外继续按公式计算。'),
            ('source', ('slm/position.py', ('LearnedPE',))),
            ('code', '''learned = LearnedPE(8, 16)
zeros = torch.zeros(1, 8, 16)
show('learned table matches', torch.allclose(learned(zeros)[0], learned.pos_emb(torch.arange(8))))
sinusoidal = PositionEncoding(16, 8)
show('sinusoidal shape', tuple(sinusoidal(zeros).shape))'''),
            ('markdown', 'LearnedPE 的硬长度限制清晰可见；sin/cos 在更长位置仍能算出数值，但这不自动带来长度外推质量。'),
            ('markdown', '## 3. 相对距离：Shaw 向量与 T5 标量 bias\n\n先把有符号距离 `j-i` 截断并平移成 embedding 下标。'),
            ('source', ('slm/position.py', ('rel_index', 'ShawRelBias', 'T5RelBias'))),
            ('demo', 'scripts/position/02_relative_bias.py'),
            ('markdown', '这是简化的 clipped signed-distance 教学版本，不是真实 T5 的对数分桶，也不是 Shaw 同时含 value-relative 项的完整模型。配套文章讨论了这些形式的重参数化关系。'),
            ('markdown', '## 4. ALiBi：把距离先验直接加到 logits\n\n每个头使用不同斜率，bias 为 $-m_h|i-j|$。'),
            ('source', ('slm/position.py', ('alibi_slopes', 'alibi_bias'))),
            ('demo', 'scripts/position/03_alibi.py'),
            ('markdown', '这里的几何斜率日程对应 canonical power-of-two 头数；原论文对非 2 次幂头数另有构造。距离 bias 只是先验，内容 logits 仍可能让远 token 获得更高权重。'),
            ('markdown', '## 5. RoPE：旋转后让点积依赖相对位置\n\n实现采用 split-half 配对：前半维与后半维成二维旋转对，因此 D 必须为偶数。'),
            ('source', ('slm/position.py', ('rope_matrix', 'precompute_cos_sin', 'apply_rope'))),
            ('demo', 'scripts/position/04_rope.py'),
            ('markdown', '正交旋转保持模长；在 q、k 内容固定时，位置 m/n 的点积可写成相对旋转。若内容随位置变化，不能把所有差异都归因于相对距离。'),
            ('markdown', '### QK-Norm 与 RoPE 的组合\n\n复用 [EP.2](02_normalization.ipynb) 首次展开的 `RMSNorm`：先归一化 Q/K，再旋转；正交旋转不改变已经控制好的向量模长。'),
            ('code', '''from slm import RMSNorm
set_seed(0)
q, k = torch.randn(2, 4, 8, 16) * 20, torch.randn(2, 4, 8, 16) * 20
cos, sin = precompute_cos_sin(8, 16)
qn, kn = RMSNorm(16)(q), RMSNorm(16)(k)
qr, kr = apply_rope(qn, cos, sin), apply_rope(kn, cos, sin)
show('Q norm preserved by RoPE', torch.allclose(qr.norm(dim=-1), qn.norm(dim=-1), atol=1e-5))
show('normalized rotated logits', (qr @ kr.transpose(-1,-2) / 4).abs().max().item())'''),
            ('markdown', '第一项应为 True；第二项是当前随机样本的测量，不把它写成固定预期数值。learned gamma 会进一步改变尺度。'),
            ('markdown', '## 6. PI、NTK scaling 与滑窗 PPL\n\nPI 把位置除以 s，因此所有频率角度都缩小；NTK base scaling 保持最高频不变，并让最低频约缩小到 1/s。公式要求 `d_head > 2`。'),
            ('source', ('slm/position.py', ('ntk_base', 'eval_ppl'))),
            ('demo', 'scripts/position/05_extrapolation.py'),
            ('markdown', '滑窗评估让每行 target 索引 1..T-1 恰好计分一次；普通 causal LM 的 logits[t] 预测 token[t+1]，这不同于本项目 TinyLM 内部显式 BOS shift 的约定。固定 bigram 在不同 context 得到相同 PPL 只验证记账，不是长上下文能力测量。YaRN、decoupled RoPE、多维 PE 在文章中介绍，但本核心未实现，不能据此声称覆盖全部位置技术。'),
            ('markdown', '## 7. NoPE：mask 可见数量是否就是位置？\n\n因果 mask 使第 t 行可见 t+1 个 key，因此结构中可能携带顺序线索。'),
            ('demo', 'scripts/position/06_nope.py'),
            ('code', '''q = k = torch.zeros(1, 8, 4)
v = torch.ones(1, 8, 4)
y = attention(q, k, v, causal_mask(8))
show('identical values stay equal', torch.allclose(y, v))'''),
            ('markdown', '反例中虽然可见数量不同，完全相同的 value 平均后仍相同。因此 mask visibility 可以携带顺序，但它本身不是“网络一定恢复绝对位置”的证明。'),
        ],
    },
    {
        'slug': '04_ppo',
        'title': 'EP.4 PPO 与 RLHF',
        'cells': [
            ('markdown', '# EP.4 PPO：从 MDP、GAE 到 token-level RLHF\n\n配套文章：[从零理解 PPO](https://momoyeyu.github.io/posts/llm-ppo/)。上一章：[03_position.ipynb](03_position.ipynb)。本章沿用 [EP.0](00_transformer.ipynb) 定义的 `set_seed` / `show`，不调用隐藏训练器：环境物理、损失、采样 buffer、更新循环全部在眼前。短运行用于检查采样与更新流程，不用于证明收敛；完整训练见 `trainer` CLI。'),
            ('code', SETUP_LATER + '\nimport copy\nimport random\nimport matplotlib.pyplot as plt\nfrom torch.distributions import Categorical'),
            ('markdown', '## 1. MDP、轨迹与折扣回报\n\nCartPole 的四维状态和物理更新完整展开。有限轨迹回报为 $G_t=\\sum_k\\gamma^k r_{t+k}$。'),
            ('source', ('slm/envs.py', ('CartPole',))),
            ('source', ('slm/rl.py', ('discounted_returns', 'rollout'))),
            ('demo', 'scripts/ppo/01_mdp.py'),
            ('markdown', '随机与恒向右策略只作基线。gamma 控制远期奖励权重；这里 episode 在 terminated 或 max_steps truncated 时结束，是明确的有限时域约定。'),
            ('markdown', '## 2. 价值函数：Bellman 精确解与 Monte Carlo 样本\n\n五状态随机游走把边界设为终止状态，从中间 C 出发。'),
            ('source', ('slm/envs.py', ('N_STATES', 'START', 'walk_step', 'is_terminal'))),
            ('demo', 'scripts/ppo/02_value.py'),
            ('markdown', 'Bellman 迭代给出这个小 MDP 的数值解；Monte Carlo 是有限样本估计，因此只应接近而非逐位相等。'),
            ('markdown', '## 3. REINFORCE 与 baseline\n\n策略梯度使用 score function $\\nabla\\log\\pi(a|s)G$。与动作无关的 baseline 在期望上不改变梯度，却可改变采样方差。'),
            ('source', ('slm/rl.py', ('mlp', 'Policy'))),
            ('demo', 'scripts/ppo/03_baseline.py'),
            ('markdown', '老虎机实验比较同一批量下的梯度均值与方差。有限样本均值不会严格等于精确梯度；结论针对 action-independent baseline 的期望。'),
            ('markdown', '## 4. Actor–Critic 与 GAE\n\nTD residual 为 $\\delta_t=r_t+\\gamma V(s_{t+1})-V(s_t)$；GAE 递推 $A_t=\\delta_t+\\gamma\\lambda A_{t+1}$。'),
            ('source', ('slm/rl.py', ('layer_init', 'ActorCritic', 'compute_gae'))),
            ('demo', 'scripts/ppo/04_gae.py'),
            ('markdown', 'lambda=0 对应单步 TD，lambda=1 在 episode 边界对应 Monte Carlo；非终止末端用 last_value bootstrap。本项目把时间上限也当有限时域终止，不宣称等同 Gym 的无限时域 truncation 语义。'),
            ('markdown', '## 5. Importance sampling 与支持集\n\n用行为分布 q 的样本估计 p 下期望，权重为 p(x)/q(x)。'),
            ('source', ('slm/rl.py', ('importance_sampling',))),
            ('demo', 'scripts/ppo/05_importance_sampling.py'),
            ('markdown', 'q 必须在 p 非零处有完整支持，否则权重不可定义。q 与 p 相差大时，有限样本权重方差会增大。'),
            ('markdown', '## 6. PPO clipping：只截断“过度有利”的方向\n\n先看正负 advantage 下超界 ratio 的梯度，再展开训练真正调用的损失。'),
            ('demo', 'scripts/ppo/06_ppo_clip.py'),
            ('source', ('slm/rl.py', ('ppo_loss',))),
            ('markdown', 'PPO 对 `min(ratio*A, clip(ratio)*A)` 取负作 policy loss，并同时拟合 value、加入 entropy。clip fraction 与近似 KL 是诊断量，不是成功保证。'),
            ('markdown', '## 7. 一个完全展开的短 PPO 循环\n\n下面依次采样 128 步、算 GAE、做两轮 minibatch 更新并画 episode return。常数保持小规模，目的只是看清数据流。'),
            ('code', PPO_LOOP),
            ('markdown', '曲线是当前随机种子下的真实记录；8 次 update 很可能波动或尚未学会。短运行用于检查采样、GAE 与 minibatch 更新是否连通，不用于证明收敛；完整训练见 `trainer/train_ppo.py`。'),
            ('markdown', '## 8. TinyLM：把 token 当 action\n\nTinyLM 用 BOS + 前缀预测当前 token；GRU 主干由 policy logits 与 value head 共享。这个对齐约定不同于普通 causal LM 的 logits[t] 预测 t+1。'),
            ('source', ('slm/lm.py', ('TinyLM', 'token_log_probs'))),
            ('markdown', '生成时每一步从 Categorical 采样；`token_log_probs` 从完整词表 logits 中取出已采样 token 的 log probability。'),
            ('markdown', '## 9. 直接展开的 token-level PPO/RLHF\n\n冻结初始 policy 作 reference，用“token 是否等于 3”的比例作 score，并把 sampled log-ratio KL 惩罚分配到每一步。'),
            ('code', RLHF_LOOP),
            ('markdown', '图展示真实 score 与 sampled sequence KL。小 batch 下 sampled KL 可以为负；reference 是冻结的随机初始网络，并非严格均匀分布，因此若另行推导均匀 reference 的闭式解，也不能直接当作本实验精确目标。'),
        ],
    },
]
