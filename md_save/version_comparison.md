# RL_Game_NPC 버전 비교 문서

## 1. 프로젝트 개요

| 항목 | 원본 (RL_Game_NPC) | 수정본 (rl-multiagent-combat-unity/RL_Game_NPC) |
|------|-------------------|------------------------------------------------|
| 위치 | `C:\Users\user\Git\RL_Game_NPC` | `C:\Users\user\Git\rl-multiagent-combat-unity\RL_Game_NPC` |
| 특징 | 팀원 버전 (Self-Play 학습) | Unity 연동 + Player Cooperation 학습 |
| 학습 스크립트 | `train.py` | `train_coop.py` |

---

## 2. 파일 구조 비교

### 2.1 공통 파일
```
main.py             # PyCharm 기본 템플릿 (동일)
play.py             # 플레이 스크립트
train.py            # Self-Play 학습 (원본과 동일)
train_asymmetric.py # 비대칭 학습
test_agent.py       # 에이전트 테스트
test_env.py         # 환경 테스트

src/
├── __init__.py
├── agent.py        # PPO 에이전트
├── config.py       # 설정 클래스 (수정됨)
├── env.py          # 환경 클래스 (수정됨)
├── env_backup.py   # 환경 백업
├── game_map.py     # 게임 맵
├── unit.py         # 유닛 클래스
└── visualizer.py   # 시각화
```

### 2.2 원본에만 있는 파일
```
generate_paper_images.py   # 논문 이미지 생성
play_asymmetric.py         # 비대칭 플레이
train_pipeline.py          # 학습 파이프라인
visualize_training.py      # 학습 시각화
```

### 2.3 수정본에만 있는 파일 (신규 추가)
```
train_coop.py              # 플레이어 협동 모드 학습 (핵심!)
unity_streamer.py          # Unity UDP 스트리밍
player_mode_streamer.py    # 플레이어 모드 스트리밍
test_coop_progress.py      # 협동 학습 진행 테스트

training_data_v2.csv       # 학습 데이터 로그
training_data_v3.csv
training_data_v4.csv       # 최신 학습 결과
```

---

## 3. 핵심 코드 변경 사항

### 3.1 config.py 변경 내용

#### 행동 공간 확장
```python
# 원본
num_actions: int = 9

# 수정본
num_actions: int = 12  # v11 모델 호환
```

#### 보상 함수 강화 (존버 방지)
| 보상 항목 | 원본 | 수정본 | 변경 목적 |
|----------|------|-------|----------|
| `reward_stay` | -0.2 | -0.5 | 제자리 패널티 강화 |
| `reward_approach` | 0.1 | 0.3 | 접근 보상 증가 |
| `reward_time_penalty` | -0.02 | -0.05 | 빠른 교전 유도 |
| `reward_draw` | -5.0 | -8.0 | 무승부 패널티 강화 |
| `reward_in_combat` | 0.05 | 0.15 | 전투 참여 보상 증가 |

#### 역할별 사망 패널티 추가 (신규)
```python
reward_tank_death: float = -2.0           # 탱커 사망 추가 패널티
reward_tank_absorb_bonus: float = 0.02    # 탱커 흡수 데미지당 감소
reward_dealer_death: float = -1.0         # 딜러 사망 (화력 손실)
reward_healer_death: float = -1.5         # 힐러 사망 (지속력 손실)
```

#### 팀별 스탯 배율 추가 (신규)
```python
team_a_stat_multiplier: float = 1.0  # A팀 스탯 배율
team_b_stat_multiplier: float = 1.0  # B팀 스탯 배율
```

#### 플레이어 협동 모드 설정 (신규)
```python
# 플레이어 유닛 인덱스 (-1: 없음, 0~4: 해당 역할)
player_idx: int = -1      # A팀 플레이어
player_idx_b: int = -1    # B팀 플레이어 (학습용)

# 협동 보상 설정 (대폭 강화)
reward_protect_player: float = 1.5      # 탱커가 플레이어 앞에서 보호
reward_support_player: float = 1.0      # 힐러가 플레이어 근처에서 힐
reward_follow_tank: float = 0.8         # 딜러/힐러가 탱커 뒤에 위치
reward_near_player: float = 0.5         # 플레이어 근처 (거리 3 이내)
reward_approach_player: float = 0.2     # 플레이어에게 접근
coop_distance_threshold: int = 5        # 협동 거리 임계값
```

#### 엔트로피 계수 변경
```python
# 원본
entropy_coef: float = 0.01

# 수정본 (탐색 증가)
entropy_coef: float = 0.05
```

---

### 3.2 env.py 변경 내용

#### 관찰 공간 확장
```python
# 원본: 223차원
self_state_size = 10
ally_state_size = 4 * 10      # 40
enemy_state_size = 5 * 10     # 50
terrain_size = 121            # (2*5+1)^2
global_size = 2

# 수정본: 229차원 (+6 플레이어 정보)
player_info_size = 6          # one-hot [없음, Tank, Dealer, Healer, Ranger, Support]
```

#### 유닛 생성 시 스탯 배율 적용
```python
# 원본
unit = Unit(unit_id=i, team_id=0, role=role, x=x, y=y)

# 수정본
unit = Unit(unit_id=i, team_id=0, role=role, x=x, y=y,
           stat_multiplier=self.config.team_a_stat_multiplier)
```

#### 존버 방지 패널티 추가 (보상 함수)
```python
# 적과 거리 8 초과 시 패널티 (양팀 공통)
if enemies:
    min_enemy_dist = min(unit.distance_to(e) for e in enemies)
    if min_enemy_dist > 8:
        reward -= 0.1 * (min_enemy_dist - 8)
```

#### 플레이어 협동 보상 시스템 (신규, ~200줄 추가)

**A팀 협동 보상:**
1. 거리 기반 연속 보상 - 플레이어에게 가까울수록 보상
2. 플레이어 근처 보상 (거리 3 이내)
3. 탱커가 플레이어(딜러/힐러/레인저) 보호 시 보상
4. 힐러가 플레이어 근처에서 힐 시 추가 보상
5. 힐러가 플레이어 HP 낮을 때 접근 유도
6. 딜러/힐러/레인저가 탱커 뒤에 위치 시 보상
7. 플레이어가 탱커일 때 뒤에서 따라오기 보상

**B팀 협동 보상:**
- A팀과 동일한 구조 (player_idx_b >= 0일 때)
- B팀 기본 협동 보상 (player_idx_b < 0일 때 - AI끼리 협동)

---

### 3.3 train_coop.py (신규 학습 스크립트)

#### 학습 모드 시스템
```python
class TrainingMode:
    AI_VS_AI = 'ai_vs_ai'          # 순수 5v5 AI 학습
    PLAYER_COOP = 'player_coop'    # 플레이어 협동 모드

AI_VS_AI_RATIO = 0.4  # AI vs AI 40% / Player Coop 60%
```

#### 플레이 스타일 시스템 (6가지)
```python
class PlayStyle:
    AGGRESSIVE = 'aggressive'      # 공격적: 적에게 돌진
    DEFENSIVE = 'defensive'        # 수비적: HP 관리, 후퇴
    FOLLOW_HEALER = 'follow_healer'  # 힐러 따라가기
    KITING = 'kiting'              # 카이팅: 공격 후 후퇴
    BEHIND_TANK = 'behind_tank'    # 탱커 뒤에서 공격
    FOCUS_FIRE = 'focus_fire'      # 집중 공격
```

#### Stochastic 행동 선택
```python
def stochastic_choose(actions_with_weights):
    """확률적 행동 선택 - 다양한 상황 학습 유도"""
    # 예: [(ActionType.ATTACK_NEAREST, 0.80), (ActionType.ATTACK_LOWEST, 0.15), ...]
```

#### 휴리스틱 플레이어 시뮬레이션
- 5% 확률 랜덤 행동 (예측 불가능성)
- HP 30% 미만 시 50% 확률로 수비적 전환
- 스타일별 상세 행동 로직 (~300줄)

#### 양팀 동시 학습
```python
# Team A: 학습 에이전트 (플레이어 제외)
agent_a = PPOAgent(obs_size, action_size, train_config)

# Team B: 함께 학습하는 에이전트
agent_b = PPOAgent(obs_size, action_size, train_config)
```

#### CSV 로깅 (논문용 그래프)
```python
# training_data_v4.csv 컬럼
episode, steps, avg_reward, avg_length,
win_rate_a, win_rate_b, draw_rate,
wins_a, wins_b, draws,
policy_loss, value_loss, entropy, fps,
ai_mode_count, coop_mode_count, ai_mode_ratio, coop_mode_ratio
```

---

### 3.4 Unity 연동 스크립트 (신규)

#### unity_streamer.py
- UDP로 게임 상태 전송 (포트 5005)
- FrameDTO 형식: 맵, 유닛 상태, 승패 정보
- 단일 모델 / 팀별 다른 모델 지원

#### player_mode_streamer.py
- UDP 송신 (게임 상태) + TCP 수신 (플레이어 입력)
- 플레이어 역할 선택 수신
- A팀: 협동 AI, B팀: 적 AI

---

## 4. Unity C# 스크립트

### 4.1 UdpReceiver.cs
- Python에서 UDP로 전송된 프레임 데이터 수신
- 싱글톤 패턴, 스레드 안전한 큐 사용
- FrameData로 JSON 파싱

### 4.2 PlayerInputSender.cs
- TCP로 플레이어 입력 전송 (포트 5006)
- 키 매핑: WASD 이동, Space 공격, Q/E/R 스킬
- 3D 쿼터뷰 시점 키 매핑 지원
- 플레이어 사망 시 입력 비활성화

### 4.3 PythonManager.cs
- Python 프로세스 생명주기 관리
- ESC 키로 메인 메뉴 복귀
- DontDestroyOnLoad 싱글톤

---

## 5. 학습 결과 분석 (training_data_v4.csv 기준)

### 5.1 학습 진행 상황
| 구간 | Episodes | 평균 보상 | Win Rate A | Win Rate B |
|------|----------|----------|------------|------------|
| 초기 | 10~100 | -139 ~ -18 | 30% ~ 40% | 57% ~ 70% |
| 중기 | 100~1000 | -18 ~ +64 | 40% ~ 54% | 44% ~ 57% |
| 후기 | 1000~3030 | +55 ~ +76 | 48% ~ 54% | 46% ~ 52% |

### 5.2 학습 모드 분포
- AI vs AI: ~38%
- Player Coop: ~62%

### 5.3 손실 함수 특성
- **Policy Loss**: -0.09 ~ +0.09 범위, 정상적 변동
- **Value Loss**: 스파이크 발생 (604, 409, 389 등)
  - 원인: Bootstrap target 불안정성, 보상 분포 변화
  - 학습에 심각한 영향 없음 (보상/승률 지속 개선)
- **Entropy**: 2.0 ~ 2.4 유지 (적절한 탐색)

---

## 6. 주요 개선 사항 요약

| 항목 | 설명 |
|------|------|
| 플레이어 협동 모드 | AI가 플레이어와 협동하도록 학습 |
| 양팀 동시 학습 | A팀, B팀 별도 모델로 경쟁 학습 |
| 휴리스틱 플레이어 | 6가지 스타일 + Stochastic 행동 |
| 존버 방지 | 패널티/보상 조정으로 적극적 교전 유도 |
| 역할별 사망 패널티 | 탱커/딜러/힐러 중요도에 따른 차별화 |
| Unity 연동 | UDP/TCP로 실시간 게임 시각화 및 조작 |
| CSV 로깅 | 논문용 학습 곡선 데이터 저장 |

---

## 7. 사용 방법

### 7.1 협동 모드 학습 실행
```bash
cd RL_Game_NPC
python train_coop.py --total-steps 5000000 --save-dir models_coop_v1 \
    --csv-file training_data.csv --random-role
```

### 7.2 Unity 스트리밍 (관전 모드)
```bash
python unity_streamer.py --mode asymmetric \
    --model models_coop_v1/model_a_latest.pt \
    --model-b models_coop_v1/model_b_latest.pt
```

### 7.3 플레이어 모드
```bash
python player_mode_streamer.py \
    --model models_coop_v1/model_a_latest.pt \
    --model-b models_coop_v1/model_b_latest.pt
```

---

## 8. 파일 경로 참조

### Python 핵심 파일
| 파일 | 경로 | 설명 |
|------|------|------|
| train_coop.py | `RL_Game_NPC/train_coop.py` | 협동 학습 메인 |
| config.py | `RL_Game_NPC/src/config.py` | 설정 (보상, 협동) |
| env.py | `RL_Game_NPC/src/env.py` | 환경 (협동 보상) |
| unity_streamer.py | `RL_Game_NPC/unity_streamer.py` | Unity 스트리밍 |
| player_mode_streamer.py | `RL_Game_NPC/player_mode_streamer.py` | 플레이어 모드 |

### Unity 핵심 파일
| 파일 | 경로 | 설명 |
|------|------|------|
| UdpReceiver.cs | `Assets/01_Script/UdpReceiver.cs` | UDP 수신 |
| PlayerInputSender.cs | `Assets/01_Script/PlayerInputSender.cs` | 플레이어 입력 |
| PythonManager.cs | `Assets/01_Script/PythonManager.cs` | Python 관리 |
| GameViewer.cs | `Assets/01_Script/GameViewer.cs` | 게임 시각화 |

---

*문서 작성일: 2025-11-29*
*프로젝트: rl-multiagent-combat-unity*
