"""PPO 에이전트 구현"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
from collections import deque
import os

from .config import TrainConfig


class ActorCritic(nn.Module):
    """Actor-Critic 네트워크 (정책 공유)"""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()

        # 공유 레이어
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor (정책)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        # Critic (가치)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        shared_out = self.shared(x)
        action_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_logits, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동 선택"""
        action_logits, value = self.forward(x)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동 평가 (학습용)"""
        action_logits, value = self.forward(x)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """경험 저장 버퍼"""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, obs: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: bool):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """GAE를 사용한 어드밴티지 계산"""
        self.advantages = []
        self.returns = []

        gae = 0
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[step])
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - float(self.dones[step])

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + self.values[step])

    def get_batches(self, batch_size: int):
        """미니배치 생성"""
        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                'observations': np.array([self.observations[i] for i in batch_indices]),
                'actions': np.array([self.actions[i] for i in batch_indices]),
                'log_probs': np.array([self.log_probs[i] for i in batch_indices]),
                'advantages': np.array([self.advantages[i] for i in batch_indices]),
                'returns': np.array([self.returns[i] for i in batch_indices]),
            }

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()


class PPOAgent:
    """PPO 에이전트 (멀티에이전트 셀프플레이)"""

    def __init__(self, obs_size: int, action_size: int, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.obs_size = obs_size
        self.action_size = action_size

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 네트워크 (모든 에이전트가 공유)
        self.network = ActorCritic(
            obs_size, action_size, self.config.hidden_size
        ).to(self.device)

        # 옵티마이저
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )

        # 경험 버퍼 (에이전트별)
        self.buffers: Dict[str, RolloutBuffer] = {}

        # 학습 통계
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """단일 관찰에 대한 행동 선택"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic)

        return action.item(), log_prob.item(), value.item()

    def get_actions(self, observations: Dict[str, np.ndarray],
                    deterministic: bool = False) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """여러 에이전트의 행동 선택"""
        actions = {}
        log_probs = {}
        values = {}

        for agent_id, obs in observations.items():
            action, log_prob, value = self.get_action(obs, deterministic)
            actions[agent_id] = action
            log_probs[agent_id] = log_prob
            values[agent_id] = value

        return actions, log_probs, values

    def store_transition(self, agent_id: str, obs: np.ndarray, action: int,
                         reward: float, value: float, log_prob: float, done: bool):
        """경험 저장"""
        if agent_id not in self.buffers:
            self.buffers[agent_id] = RolloutBuffer()

        self.buffers[agent_id].add(obs, action, reward, value, log_prob, done)

    def learn(self) -> Dict[str, float]:
        """PPO 학습"""
        # 모든 에이전트의 경험을 하나로 합침
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []

        # 마지막 가치 계산 및 어드밴티지 계산
        for agent_id, buffer in self.buffers.items():
            if len(buffer.observations) == 0:
                continue

            # 마지막 관찰의 가치
            with torch.no_grad():
                last_obs = torch.FloatTensor(buffer.observations[-1]).unsqueeze(0).to(self.device)
                _, last_value = self.network(last_obs)
                last_value = last_value.item()

            # GAE 계산
            buffer.compute_returns_and_advantages(
                last_value, self.config.gamma, self.config.gae_lambda
            )

            all_obs.extend(buffer.observations)
            all_actions.extend(buffer.actions)
            all_log_probs.extend(buffer.log_probs)
            all_advantages.extend(buffer.advantages)
            all_returns.extend(buffer.returns)

        if len(all_obs) == 0:
            return {}

        # 어드밴티지 정규화
        advantages = np.array(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 학습 통계
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # 여러 에폭 학습
        for _ in range(self.config.num_epochs):
            # 미니배치 생성
            indices = np.random.permutation(len(all_obs))

            for start in range(0, len(all_obs), self.config.batch_size):
                end = min(start + self.config.batch_size, len(all_obs))
                batch_indices = indices[start:end]

                # 배치 데이터
                batch_obs = torch.FloatTensor(
                    np.array([all_obs[i] for i in batch_indices])
                ).to(self.device)
                batch_actions = torch.LongTensor(
                    [all_actions[i] for i in batch_indices]
                ).to(self.device)
                batch_old_log_probs = torch.FloatTensor(
                    [all_log_probs[i] for i in batch_indices]
                ).to(self.device)
                batch_advantages = torch.FloatTensor(
                    [advantages[i] for i in batch_indices]
                ).to(self.device)
                batch_returns = torch.FloatTensor(
                    [all_returns[i] for i in batch_indices]
                ).to(self.device)

                # 새로운 로그 확률과 가치 계산
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    batch_obs, batch_actions
                )

                # PPO 클리핑
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실
                value_loss = F.mse_loss(values, batch_returns)

                # 총 손실
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )

                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 버퍼 초기화
        for buffer in self.buffers.values():
            buffer.clear()

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def save(self, path: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """모델 로드 (obs_size 불일치 허용)"""
        checkpoint = torch.load(path, map_location=self.device)
        saved_state = checkpoint['network']
        current_state = self.network.state_dict()

        # obs_size 불일치 확인 (shared.0.weight의 input 차원)
        saved_input_size = saved_state['shared.0.weight'].shape[1]
        current_input_size = current_state['shared.0.weight'].shape[1]

        if saved_input_size != current_input_size:
            print(f"[Warning] obs_size mismatch: saved={saved_input_size}, current={current_input_size}")

            if saved_input_size < current_input_size:
                # 작은 모델을 큰 네트워크에 로드 (추가 차원은 0으로)
                new_state = {}
                for key, param in saved_state.items():
                    if key == 'shared.0.weight':
                        # 첫 레이어의 입력 차원 확장
                        new_weight = torch.zeros_like(current_state[key])
                        new_weight[:, :saved_input_size] = param
                        new_state[key] = new_weight
                    else:
                        new_state[key] = param
                self.network.load_state_dict(new_state)
                print(f"Model loaded with padding (zeros for extra {current_input_size - saved_input_size} dims)")
            else:
                # 큰 모델을 작은 네트워크에 로드 (지원 안함)
                raise RuntimeError(f"Cannot load larger model ({saved_input_size}) into smaller network ({current_input_size})")
        else:
            self.network.load_state_dict(saved_state)

        # optimizer는 구조가 다르면 스킵
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            print("[Warning] Optimizer state skipped due to size mismatch")

        self.total_steps = checkpoint.get('total_steps', 0)
        print(f"Model loaded from {path}")