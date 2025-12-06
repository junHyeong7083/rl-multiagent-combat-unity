"""학습 결과 그래프 생성 스크립트"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def parse_selfplay_log(filepath):
    """셀프플레이 로그 파일 파싱"""
    episodes = []
    steps = []
    avg_rewards = []
    win_rate_a = []
    win_rate_b = []
    draws = []
    fps_list = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 정규식으로 에피소드 데이터 추출
    pattern = r'\[Episode (\d+)\] Steps: ([\d,]+)\s+Avg Reward: ([\d.-]+)\s+Avg Length: [\d.]+\s+Win Rate A: \d+/\d+ \(([\d.]+)%\)\s+Win Rate B: \d+/\d+ \(([\d.]+)%\)\s+Draws: \d+ \(([\d.]+)%\)\s+FPS: (\d+)'

    matches = re.findall(pattern, content)

    for match in matches:
        episodes.append(int(match[0]))
        steps.append(int(match[1].replace(',', '')))
        avg_rewards.append(float(match[2]))
        win_rate_a.append(float(match[3]))
        win_rate_b.append(float(match[4]))
        draws.append(float(match[5]))
        fps_list.append(int(match[6]))

    return pd.DataFrame({
        'episode': episodes,
        'steps': steps,
        'avg_reward': avg_rewards,
        'win_rate_a': win_rate_a,
        'win_rate_b': win_rate_b,
        'draws': draws,
        'fps': fps_list
    })

def plot_selfplay_results(df, save_path='selfplay_results.png'):
    """셀프플레이 결과 그래프 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1단계: 셀프플레이 학습 결과 (Self-Play Training)', fontsize=16, fontweight='bold')

    # 1. 승률 변화 (Win Rate Curve)
    ax1 = axes[0, 0]
    ax1.plot(df['episode'], df['win_rate_a'], 'b-', label='Team A', linewidth=2)
    ax1.plot(df['episode'], df['win_rate_b'], 'r-', label='Team B', linewidth=2)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='균형점 (50%)')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('승률 변화 추이 (Win Rate Curve)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. 평균 보상 (Avg Reward)
    ax2 = axes[0, 1]
    ax2.plot(df['episode'], df['avg_reward'], 'g-', linewidth=2)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('평균 에피소드 보상 (Avg Reward)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 3. 학습 스텝 진행
    ax3 = axes[1, 0]
    ax3.plot(df['episode'], df['steps'] / 1_000_000, 'purple', linewidth=2)
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Total Steps (M)')
    ax3.set_title('학습 스텝 진행')
    ax3.grid(True, alpha=0.3)

    # 4. FPS
    ax4 = axes[1, 1]
    ax4.plot(df['episode'], df['fps'], 'orange', linewidth=2)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('FPS')
    ax4.set_title('학습 FPS')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장됨: {save_path}")

def plot_coop_results(csv_path, save_path='coop_v11_results.png'):
    """협동 학습 결과 그래프 생성"""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('2단계: 협동 학습 결과 (Cooperative Training v11)', fontsize=16, fontweight='bold')

    # 1. 평균 보상 (Reward Curve)
    ax1 = axes[0, 0]
    ax1.plot(df['episode'], df['avg_reward'], 'g-', linewidth=2)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('평균 에피소드 보상 (Reward Curve)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 시작점과 끝점 표시
    ax1.scatter([df['episode'].iloc[0]], [df['avg_reward'].iloc[0]], color='red', s=100, zorder=5, label=f'시작: {df["avg_reward"].iloc[0]:.0f}')
    ax1.scatter([df['episode'].iloc[-1]], [df['avg_reward'].iloc[-1]], color='blue', s=100, zorder=5, label=f'종료: {df["avg_reward"].iloc[-1]:.0f}')
    ax1.legend()

    # 2. 평균 탱커 거리 (핵심 지표!)
    ax2 = axes[0, 1]
    ax2.plot(df['episode'], df['avg_tank_dist'], 'r-', linewidth=2)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Avg Tank Distance')
    ax2.set_title('평균 탱커 거리 (협동 지표)')
    ax2.grid(True, alpha=0.3)

    # 시작점과 끝점 표시
    ax2.scatter([df['episode'].iloc[0]], [df['avg_tank_dist'].iloc[0]], color='red', s=100, zorder=5, label=f'시작: {df["avg_tank_dist"].iloc[0]:.1f}')
    ax2.scatter([df['episode'].iloc[-1]], [df['avg_tank_dist'].iloc[-1]], color='blue', s=100, zorder=5, label=f'종료: {df["avg_tank_dist"].iloc[-1]:.1f}')
    ax2.legend()

    # 3. 승률 변화
    ax3 = axes[1, 0]
    ax3.plot(df['episode'], df['win_rate_a'], 'b-', linewidth=2, label='Team A (NPC)')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='균형점 (50%)')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Win Rate A (%)')
    ax3.set_title('Team A 승률 변화')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 60)

    # 4. 에피소드 길이
    ax4 = axes[1, 1]
    ax4.plot(df['episode'], df['avg_length'], 'purple', linewidth=2)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Avg Episode Length')
    ax4.set_title('평균 에피소드 길이')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장됨: {save_path}")

def plot_selfplay_winrate_only(df, save_path='selfplay_winrate.png'):
    """셀프플레이 승률 그래프만 (PPT용)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['episode'], df['win_rate_a'], 'b-', label='Team A', linewidth=2.5)
    ax.plot(df['episode'], df['win_rate_b'], 'r-', label='Team B', linewidth=2.5)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('승률 변화 추이 (Win Rate Curve)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장됨: {save_path}")

def plot_coop_reward_only(csv_path, save_path='coop_reward.png'):
    """협동 학습 보상 그래프만 (PPT용)"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['episode'], df['avg_reward'], 'g-', linewidth=2.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 시작점과 끝점 표시
    ax.scatter([df['episode'].iloc[0]], [df['avg_reward'].iloc[0]], color='red', s=100, zorder=5)
    ax.scatter([df['episode'].iloc[-1]], [df['avg_reward'].iloc[-1]], color='blue', s=100, zorder=5)

    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('평균 에피소드 보상 (Reward Curve)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 주석 추가
    ax.annotate(f'시작: {df["avg_reward"].iloc[0]:.0f}',
                xy=(df['episode'].iloc[0], df['avg_reward'].iloc[0]),
                xytext=(1000, df['avg_reward'].iloc[0] + 300),
                fontsize=10, color='red')
    ax.annotate(f'종료: {df["avg_reward"].iloc[-1]:.0f}',
                xy=(df['episode'].iloc[-1], df['avg_reward'].iloc[-1]),
                xytext=(df['episode'].iloc[-1] - 2000, df['avg_reward'].iloc[-1] + 300),
                fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장됨: {save_path}")

if __name__ == '__main__':
    import os

    # 경로 설정
    selfplay_log = r"C:\Users\user\Git\RL_Game_NPC\training_log.txt"
    coop_csv = r"C:\Users\user\Git\rl-multiagent-combat-unity\RL_Game_NPC\training_data_v11.csv"
    output_dir = r"C:\Users\user\Git\rl-multiagent-combat-unity\RL_Game_NPC\graphs"

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 1. 셀프플레이 그래프 생성
    print("=" * 50)
    print("1단계: 셀프플레이 그래프 생성")
    print("=" * 50)

    if os.path.exists(selfplay_log):
        df_selfplay = parse_selfplay_log(selfplay_log)
        print(f"파싱 완료: {len(df_selfplay)} 에피소드")

        # 전체 결과 그래프
        plot_selfplay_results(df_selfplay, os.path.join(output_dir, 'selfplay_results.png'))

        # 승률만 (PPT용)
        plot_selfplay_winrate_only(df_selfplay, os.path.join(output_dir, 'selfplay_winrate.png'))
    else:
        print(f"파일 없음: {selfplay_log}")

    # 2. 협동 학습 그래프 생성
    print("\n" + "=" * 50)
    print("2단계: 협동 학습 그래프 생성")
    print("=" * 50)

    if os.path.exists(coop_csv):
        # 전체 결과 그래프
        plot_coop_results(coop_csv, os.path.join(output_dir, 'coop_v11_results.png'))

        # 보상만 (PPT용)
        plot_coop_reward_only(coop_csv, os.path.join(output_dir, 'coop_reward.png'))
    else:
        print(f"파일 없음: {coop_csv}")

    print("\n완료!")
