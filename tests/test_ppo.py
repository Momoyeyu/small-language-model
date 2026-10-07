"""PPO 模块：验证文章「从零理解 PPO」与代码中的每一条结论。

快速测试默认运行（CPU 上约 30 秒）；完整训练测试较慢，需设置环境变量 SLM_SLOW=1。
设备由 SLM_DEVICE 指定（默认 cpu）。
"""
import importlib.util
import math
import os
import random

import pytest
import torch

from envs import CartPole
from model import ActorCritic
from trainer.train_ppo import ppo, ppo_loss
from trainer.train_reinforce import reinforce
from trainer.train_rlhf import closed_form, rlhf_ppo
from utils.common import ROOT, get_device, set_seed
from utils.rl_utils import compute_gae, discounted_returns, importance_sampling, rollout

DEVICE = get_device(os.environ.get("SLM_DEVICE", "cpu"))
slow = pytest.mark.skipif(os.environ.get("SLM_SLOW") != "1", reason="set SLM_SLOW=1 to run full training")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "ppo", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- MDP ----------

def test_discounted_returns_matches_definition():
    rewards, gamma = [1.0, 0.0, 2.0, 3.0], 0.9
    direct = [sum(gamma**k * rewards[t + k] for k in range(len(rewards) - t)) for t in range(len(rewards))]
    assert discounted_returns(rewards, gamma) == pytest.approx(direct)


def test_baseline_policies_fail_quickly():
    set_seed(0)
    env = CartPole()
    rand = [len(rollout(env, lambda s: random.randint(0, 1))[2]) for _ in range(1000)]
    right = [len(rollout(env, lambda s: 1)[2]) for _ in range(100)]
    assert 15 < sum(rand) / len(rand) < 30
    assert max(right) < 15


# ---------- 价值函数 ----------

def test_bellman_and_monte_carlo_on_random_walk():
    value = load_script("02_value")
    set_seed(0)
    V = value.policy_evaluation()
    assert V[1:6] == pytest.approx([i / 6 for i in range(1, 6)], abs=1e-8)
    V_mc = value.monte_carlo(10000)
    assert max(abs(V_mc[i] - i / 6) for i in range(1, 6)) < 0.02
    assert V[4] - V[3] == pytest.approx(1 / 6) and V[2] - V[3] == pytest.approx(-1 / 6)


# ---------- 策略梯度 ----------

def test_baseline_keeps_expectation_zero():
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    probs = torch.softmax(logits, -1)
    # E_a[∇log π(a) · b] = b · Σ_a ∇π(a) = 0
    term = sum(probs[a].detach() * torch.autograd.grad(torch.log(probs[a]), logits, retain_graph=True)[0]
               for a in range(5))
    assert torch.allclose(term, torch.zeros(5), atol=1e-6)


# ---------- GAE ----------

def test_gae_limits():
    set_seed(0)
    T, gamma = 10, 0.9
    rewards, values = torch.rand(T), torch.rand(T)
    dones = torch.zeros(T)
    dones[4] = dones[-1] = 1.0
    last_value = torch.tensor(0.7)

    adv, ret = compute_gae(rewards, values, dones, last_value, gamma, 1.0)
    G = torch.tensor(discounted_returns(rewards[:5].tolist(), gamma) + discounted_returns(rewards[5:].tolist(), gamma))
    assert torch.allclose(adv, G - values, atol=1e-6) and torch.allclose(ret, G, atol=1e-6)

    adv, _ = compute_gae(rewards, values, dones, last_value, gamma, 0.0)
    next_values = torch.cat([values[1:], last_value.view(1)])
    assert torch.allclose(adv, rewards + gamma * next_values * (1 - dones) - values, atol=1e-6)

    adv, _ = compute_gae(rewards, values, torch.zeros(T), last_value, gamma, 1.0)
    G = torch.tensor(discounted_returns(rewards.tolist() + [last_value.item()], gamma)[:T])
    assert torch.allclose(adv, G - values, atol=1e-5)


# ---------- 重要性采样 ----------

def test_importance_sampling_unbiased_but_variance_grows():
    set_seed(0)
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    f = torch.tensor([1.0, 2.0, 3.0, 4.0])
    near = importance_sampling(p, torch.tensor([0.15, 0.2, 0.3, 0.35]), f, 100_000)
    far = importance_sampling(p, torch.tensor([0.7, 0.2, 0.07, 0.03]), f, 100_000)
    assert near[0] == pytest.approx(3.0, abs=0.02)
    assert far[1] > 3 * near[1]


# ---------- PPO ----------

def test_ppo_loss_starts_unclipped():
    set_seed(0)
    model = ActorCritic(4, 2)
    obs, actions = torch.randn(32, 4), torch.randint(0, 2, (32,))
    with torch.no_grad():
        old = model(obs)[0].log_prob(actions)
    _, info = ppo_loss(model, obs, actions, old, torch.randn(32), torch.randn(32))
    assert info["clip_frac"] == 0 and abs(info["approx_kl"]) < 1e-7


@pytest.mark.parametrize("A, zero_grad", [(1.0, True), (-1.0, False)])
def test_clip_gradient(A, zero_grad):
    log_prob = torch.tensor([0.0], requires_grad=True)
    ratio = torch.exp(log_prob - torch.tensor([-0.5]))   # ρ ≈ 1.65
    adv = torch.tensor([A])
    torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).sum().backward()
    assert (log_prob.grad.item() == 0) == zero_grad


def test_ppo_smoke():
    set_seed(0)
    history, _ = ppo(CartPole(), total_steps=20_480, device=DEVICE)
    early = sum(r for _, r in history[:20]) / 20
    late = sum(r for _, r in history[-10:]) / 10
    assert late > 2 * early


# ---------- RLHF ----------

def test_closed_form_values():
    p, kl = closed_form(8, 8, 0.1)
    assert p == pytest.approx(0.3327, abs=1e-3) and kl == pytest.approx(1.159, abs=1e-2)


def test_rlhf_smoke():
    set_seed(0)
    history, _ = rlhf_ppo(iterations=30, beta=0.1, device=DEVICE)
    assert history[-1][0] > history[0][0] + 0.05


# ---------- 完整训练（慢） ----------

@slow
def test_reinforce_baseline_solves():
    set_seed(0)
    history, _ = reinforce(CartPole(), num_episodes=1000, use_baseline=True, device=DEVICE)
    assert sum(history[-100:]) / 100 > 400


@slow
def test_ppo_solves():
    set_seed(0)
    history, _ = ppo(CartPole(), total_steps=100_000, device=DEVICE)
    assert any(sum(r for _, r in history[i - 9:i + 1]) / 10 >= 475 for i in range(9, len(history)))


@slow
@pytest.mark.parametrize("beta", [0.0, 0.1])
def test_rlhf_matches_theory(beta):
    set_seed(0)
    history, _ = rlhf_ppo(beta=beta, device=DEVICE)
    score = sum(s for s, _ in history[-10:]) / 10
    kl = sum(k for _, k in history[-10:]) / 10
    if beta == 0:
        assert score > 0.98 and kl == pytest.approx(8 * math.log(8), abs=0.5)
    else:
        p, kl_star = closed_form(8, 8, beta)
        assert score == pytest.approx(p, abs=0.03) and kl == pytest.approx(kl_star, abs=0.15)
