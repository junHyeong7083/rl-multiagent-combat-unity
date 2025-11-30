# v7 vs v11 협동 학습 비교 분석

## 한눈에 보는 결과

| 항목 | v7 (실패) | v11 (성공) |
|------|-----------|------------|
| **최종 Reward** | +181 | +1,701 |
| **승률** | 61% | 46% |
| **tank_dist** | 추적 안 함 | 8.87 → 3.05칸 |

---

## 1. 탱커 정책 차이 (가장 중요!)

### v7: 모델 100%
```python
# train_coop_v7.py (line 141-142)
obs_tank = {tank_id: obs[tank_id]}
actions_tank, _, _ = agent_tank.get_actions(obs_tank)
```

| 정책 | 비율 |
|------|------|
| v11 모델 | **100%** |

→ 탱커가 **항상 적에게 돌진**

---

### v11: Mixed Policy (혼합)
```python
# train_coop_v11.py (line 401-402)
tank_action = tank_policy.get_action(env, tank_id)
```

| 정책 | 비율 | 행동 |
|------|------|------|
| **GoalTankPolicy** | 50% | 랜덤 목표 지점으로 이동 |
| **ConditionalTankPolicy** | 35% | 적 멀면 랜덤, 가까우면 공격 |
| **ModelTankPolicy** | 15% | v11 모델 사용 |

→ 탱커가 **50%는 랜덤하게 움직임**

---

## 2. 각 정책 상세 설명

### GoalTankPolicy (50%)
```python
class GoalTankPolicy:
    """Type A: 목표 기반 이동 - 순수 따라가기 학습용"""

    def reset_episode(self, env):
        # 에피소드 시작 시 랜덤 목표 설정
        self.goal_x = random(1, map_width-1)
        self.goal_y = random(1, map_height-1)

    def get_action(self, env, tank_id):
        # 목표 도달하면 새 목표 설정
        # 목표 방향으로 이동
```
- **목적**: NPC가 "적 위치"가 아닌 "탱커 위치"를 따라가도록 강제
- **행동**: 맵 내 랜덤 위치로 이동

---

### ConditionalTankPolicy (35%)
```python
class ConditionalTankPolicy:
    """Type C: 조건부 - 적 멀면 랜덤 이동, 가까우면 공격"""

    def get_action(self, env, tank_id):
        nearest_enemy = find_nearest_enemy()

        if distance > engage_range:  # 4칸 초과
            return random_move()     # 랜덤 이동
        else:
            return attack_enemy()    # 공격
```
- **목적**: 현실적인 탱커 행동 시뮬레이션
- **행동**: 적이 멀면 랜덤, 가까우면 공격

---

### ModelTankPolicy (15%)
```python
class ModelTankPolicy:
    """Type B: v11 모델 사용"""

    def get_action(self, env, tank_id):
        obs = env.get_observation(tank_id)
        action, _, _ = self.agent.get_actions({tank_id: obs})
        return action
```
- **목적**: 기존 v11 모델의 공격적 행동 유지
- **행동**: 적에게 접근 + 공격

---

## 3. 보상 구조 차이

### v7 보상
```python
# train_coop_v7.py (line 93-117)
reward_per_tile = 1.5        # 근접 보상 (1칸당)
reward_leave_per_tile = -2.0 # 이탈 패널티
proximity_threshold = 2      # 2칸 기준

reward_combat = 5.0          # 공격 보상
combat_range = 3             # 3칸 이내 공격 시
reward_behind_tank = 0.5     # 탱커 뒤 위치

reward_kill = 10.0           # 처치
reward_damage = 0.5          # 데미지
reward_win = 20.0            # 승리
reward_lose = -10.0          # 패배
reward_time_penalty = -0.1   # 시간 패널티
```

---

### v11 보상
```python
# train_coop_v11.py (line 281-303)
reward_per_tile = 2.5        # 근접 보상 (↑)
reward_leave_per_tile = -1.5 # 이탈 패널티 (완화)
proximity_threshold = 2      # 동일

reward_combat = 15.0         # 공격 보상 (3배!)
combat_range = 4             # 4칸으로 확대
reward_far_attack = -1.0     # 먼 거리 공격 패널티 (완화)
reward_behind_tank = 0.2     # 탱커 뒤 위치 (↓)

reward_kill = 15.0           # 처치 (↑)
reward_damage = 0.5          # 동일
reward_win = 25.0            # 승리 (↑)
reward_lose = -15.0          # 패배
reward_draw = -10.0          # 무승부
reward_time_penalty = -0.02  # 시간 패널티 (완화)
```

---

### 보상 비교표

| 보상 항목 | v7 | v11 | 변화 |
|----------|-----|-----|------|
| 근접 보상 (칸당) | 1.5 | 2.5 | +67% |
| 이탈 패널티 (칸당) | -2.0 | -1.5 | 완화 |
| **공격 보상** | 5.0 | **15.0** | **+200%** |
| 공격 범위 | 3칸 | 4칸 | 확대 |
| 처치 보상 | 10.0 | 15.0 | +50% |
| 승리 보상 | 20.0 | 25.0 | +25% |
| 시간 패널티 | -0.1 | -0.02 | 완화 |

**핵심**: v11은 **공격 보상을 3배**로 올려서 "탱커 따라가면서 공격"을 유도

---

## 4. Observation 차원 차이

### v7
```
기본 obs: 217차원 (v11 환경)
tank_info: 6차원
─────────────────
NPC obs: 223차원
```

### v11
```
기본 obs: 229차원 (v11 환경 업데이트)
tank_info: 6차원
─────────────────
NPC obs: 235차원
```

---

### tank_info 구조 (동일)
```python
tank_info = np.array([
    (tank.x - npc.x) / map_width,   # [0] 상대 x
    (tank.y - npc.y) / map_height,  # [1] 상대 y
    distance / 20.0,                 # [2] 거리 (정규화)
    float(tank.is_alive),            # [3] 생존 여부
    tank.hp / tank.max_hp,           # [4] HP 비율
    1.0                              # [5] 존재 플래그
])
```

---

## 5. 왜 v7이 실패했는가?

### 핵심: v7에도 tank_info를 넣었지만, NPC가 "안 써도 됐다"

**v7 obs 구조 (223차원):**
```
기존 obs (217차원)     +     tank_info (6차원)
   ↓                              ↓
적 위치 정보 포함            탱커 상대좌표/거리
```

### Spurious Correlation (허위 상관관계)

```
┌─────────────────────────────────────────────────────┐
│                    v7 환경                          │
│                                                     │
│   [NPC] ───────→ [탱커] ───────→ [적]              │
│                                                     │
│   탱커가 100% 모델 = 항상 적에게 감                 │
│   결과: 탱커 위치 ≈ 적 위치                        │
│                                                     │
│   NPC obs (223차원):                                │
│   ├─ 기존 217차원: 적 위치 정보 있음               │
│   └─ tank_info 6차원: 탱커 위치 정보 있음          │
│                                                     │
│   BUT! 탱커 ≈ 적 위치이므로:                       │
│   - 적 위치 따라가기 → 탱커 근처 도착 → 보상 ✓    │
│   - tank_info 따라가기 → 탱커 근처 도착 → 보상 ✓  │
│                                                     │
│   → 둘 다 같은 결과! NPC가 tank_info 무시하고      │
│     더 익숙한 "적 따라가기"만 학습함 (shortcut)    │
└─────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────┐
│                    v11 환경                         │
│                                                     │
│   [적]        [탱커] ←── 50% 랜덤 이동             │
│     ↖           ↑                                  │
│       \       [NPC]                                │
│         \                                          │
│   탱커가 랜덤하게 움직임                            │
│   결과: 탱커 위치 ≠ 적 위치                        │
│                                                     │
│   NPC obs (235차원):                                │
│   ├─ 기존 229차원: 적 위치 정보 있음               │
│   └─ tank_info 6차원: 탱커 위치 정보 있음          │
│                                                     │
│   탱커 ≠ 적 위치이므로:                            │
│   - 적 위치 따라가기 → 탱커와 멀어짐 → 패널티 ✗   │
│   - tank_info 따라가기 → 탱커 근처 유지 → 보상 ✓  │
│                                                     │
│   → tank_info를 써야만 보상 획득!                  │
│     NPC가 진짜 "탱커 따라가기" 학습함              │
└─────────────────────────────────────────────────────┘
```

---

## 6. 학습 지표 비교

### v7 CSV 컬럼
```
episode, steps, avg_reward, avg_length,
win_rate_a, wins_a, wins_b, draws, fps
```
→ **tank_dist 없음** (검증 불가)

### v11 CSV 컬럼
```
episode, steps, avg_reward, avg_length,
win_rate_a, wins_a, wins_b, draws, fps,
avg_tank_dist, tank_policy_type          ← 추가!
```
→ **tank_dist로 따라가기 행동 검증 가능**

---

## 7. 학습 결과

### v7 최종 (ep 6510)
```
Reward: +181
Win Rate: 61%
tank_dist: 측정 안 함
```

### v11 최종 (ep 12180)
```
Reward: +1,701
Win Rate: 46%
tank_dist: 3.05칸 (초기 8.87칸에서 감소)
```

> **승률이 낮은 이유**: 탱커가 50% 랜덤 이동하므로 전투력 감소
> **하지만**: NPC가 탱커를 따라가는 행동은 성공적으로 학습 (tank_dist 감소)

---

## 8. 결론

### v7 실패 원인
1. **tank_info는 넣었음** (223차원 = 217 + 6)
2. 탱커 = v11 모델 100% → 항상 적에게 감
3. 탱커 위치 ≈ 적 위치 → **tank_info 안 써도 보상 획득 가능**
4. NPC가 tank_info를 무시하고 기존 적 위치 정보만 사용 (shortcut)
5. 결과: "탱커 따라가기" 대신 "적 따라가기" 학습

### v11 성공 요인
1. **tank_info 동일하게 넣음** (235차원 = 229 + 6)
2. Mixed Policy → 탱커가 50% 랜덤 이동
3. 탱커 위치 ≠ 적 위치 → **tank_info 써야만 보상 획득**
4. NPC가 tank_info를 사용하도록 강제됨
5. tank_dist 지표로 실제 행동 검증

### 핵심 교훈
> **"학습 환경에서 shortcut이 존재하면, 모델은 의도한 행동 대신 shortcut을 학습한다"**

---

## 9. 파일 참조

| 파일 | 설명 |
|------|------|
| `train_coop_v7.py` | v7 학습 스크립트 |
| `train_coop_v11.py` | v11 학습 스크립트 |
| `training_data_v7_new.csv` | v7 학습 로그 |
| `training_data_v11.csv` | v11 학습 로그 |
| `models_coop_v11/model_npc_final.pt` | v11 협동 NPC 모델 |
| `models_v11_10k_episodes/model_final.pt` | v11 일반 모델 |
