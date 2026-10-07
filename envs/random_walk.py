import random

# Sutton & Barto 书中的 5 状态随机游走：0 与 6 为终止状态，1..5 对应 A..E。
# 从 C(3) 出发，每步等概率向左或向右，到达右端终止状态时奖励为 1，其余为 0。
N_STATES = 7
START = N_STATES // 2


def walk_step(s: int) -> tuple[int, float]:
    s_next = s + random.choice([-1, 1])
    return s_next, 1.0 if s_next == N_STATES - 1 else 0.0


def is_terminal(s: int) -> bool:
    return not 0 < s < N_STATES - 1
