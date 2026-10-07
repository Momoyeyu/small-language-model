"""价值函数：在随机游走上用贝尔曼方程与蒙特卡洛估计 V(s)，并计算优势函数。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from slm import N_STATES, START, discounted_returns, is_terminal, set_seed, show, walk_step


def policy_evaluation(gamma: float = 1.0, tol: float = 1e-10) -> list[float]:
    V = [0.0] * N_STATES
    while True:
        delta = 0.0
        for s in range(1, N_STATES - 1):
            v_new = 0.0
            for s_next in (s - 1, s + 1):
                r = 1.0 if s_next == N_STATES - 1 else 0.0
                v_new += 0.5 * (r + gamma * V[s_next])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < tol:
            return V


def monte_carlo(num_episodes: int, gamma: float = 1.0) -> list[float]:
    total, count = [0.0] * N_STATES, [0] * N_STATES
    for _ in range(num_episodes):
        s, states, rewards = START, [], []
        while not is_terminal(s):
            s_next, r = walk_step(s)
            states.append(s)
            rewards.append(r)
            s = s_next
        for s, G in zip(states, discounted_returns(rewards, gamma)):
            total[s] += G
            count[s] += 1
    return [t / c if c else 0.0 for t, c in zip(total, count)]


if __name__ == "__main__":
    set_seed(0)
    V = policy_evaluation()
    V_mc = monte_carlo(10000)
    show("true value", [round(i / 6, 4) for i in range(1, 6)])
    show("Bellman", [round(v, 4) for v in V[1:-1]])
    show("Monte Carlo", f"{[round(v, 4) for v in V_mc[1:-1]]} (max error {max(abs(V_mc[i] - i / 6) for i in range(1, 6)):.4f})")
    # 在 C 处：Q(C, 右) = 0 + V(D)，Q(C, 左) = 0 + V(B)
    show("advantages at C", f"right {V[4] - V[3]:+.4f}, left {V[2] - V[3]:+.4f}")
