import math
import random


class CartPole:
    """手写的 CartPole，物理参数与 Gymnasium 的 CartPole-v1 一致。

    动作 0 / 1 表示向左 / 向右推小车；每坚持一步奖励为 1；
    杆子倾角超过 12° 或小车出界时 terminated，达到 max_steps 时 truncated。
    """

    def __init__(self, max_steps: int = 500) -> None:
        self.gravity, self.tau, self.force_mag = 9.8, 0.02, 10.0
        self.mass_cart, self.mass_pole, self.length = 1.0, 0.1, 0.5
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length
        self.theta_limit = 12 * 2 * math.pi / 360
        self.x_limit = 2.4
        self.max_steps = max_steps
        self.obs_dim, self.n_actions = 4, 2

    def reset(self) -> list[float]:
        self.state = [random.uniform(-0.05, 0.05) for _ in range(4)]
        self.t = 0
        return list(self.state)

    def step(self, action: int) -> tuple[list[float], float, bool, bool]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos, sin = math.cos(theta), math.sin(theta)
        temp = (force + self.pole_mass_length * theta_dot**2 * sin) / self.total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4 / 3 - self.mass_pole * cos**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos / self.total_mass
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        theta, theta_dot = theta + self.tau * theta_dot, theta_dot + self.tau * theta_acc
        self.state = [x, x_dot, theta, theta_dot]
        self.t += 1

        terminated = abs(x) > self.x_limit or abs(theta) > self.theta_limit
        truncated = self.t >= self.max_steps
        return list(self.state), 1.0, terminated, truncated


# Sutton & Barto 书中的 5 状态随机游走：0 与 6 为终止状态，1..5 对应 A..E。
# 从 C(3) 出发，每步等概率向左或向右，到达右端终止状态时奖励为 1，其余为 0。
N_STATES = 7
START = N_STATES // 2


def walk_step(s: int) -> tuple[int, float]:
    s_next = s + random.choice([-1, 1])
    return s_next, 1.0 if s_next == N_STATES - 1 else 0.0


def is_terminal(s: int) -> bool:
    return not 0 < s < N_STATES - 1
