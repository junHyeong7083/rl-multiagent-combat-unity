"""Training Progress Visualization Script

Reads training_data_v11.csv and generates training progress charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

def load_training_data(csv_path):
    """Load CSV file"""
    df = pd.read_csv(csv_path)
    return df

def plot_main_chart(df, save_path=None):
    """Main chart: Normalized Cooperation + Reward"""

    fig, ax = plt.subplots(figsize=(12, 5))

    window = 30
    episodes = df['episode']

    # === Raw data ===
    cooperation = 15 - df['avg_tank_dist']
    reward = df['avg_reward']

    # === Normalization (Min-Max to 0-1) ===
    coop_min, coop_max = cooperation.min(), cooperation.max()
    reward_min, reward_max = reward.min(), reward.max()

    coop_norm = (cooperation - coop_min) / (coop_max - coop_min)
    reward_norm = (reward - reward_min) / (reward_max - reward_min)

    # Smoothing
    coop_smooth = coop_norm.rolling(window=window, center=True).mean()
    reward_smooth = reward_norm.rolling(window=window, center=True).mean()

    # === Plot ===
    ax.plot(episodes, coop_smooth, color='red', linewidth=2, label='NPC-User Distance')
    ax.plot(episodes, reward_smooth, color='blue', linewidth=2, label='Avg Reward')

    # === Reward=0 baseline (normalized position) ===
    reward_zero_norm = (0 - reward_min) / (reward_max - reward_min)
    ax.axhline(y=reward_zero_norm, color='black', linestyle='--', linewidth=1, alpha=0.7, label=f'Reward=0')

    # === Training stage separators ===
    stages = [
        (1000, 'Early'),
        (5000, 'Mid'),
        (10000, 'Final'),
    ]
    for ep, label in stages:
        if ep <= episodes.max():
            ax.axvline(x=ep, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(ep, 1.03, label, ha='center', fontsize=10, color='gray', fontweight='bold',
                   transform=ax.get_xaxis_transform())

    ax.set_xlabel('Episode (Training Progress)', fontsize=11)
    ax.set_ylabel('Normalized Value (0-1)', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved: {save_path}")

    # Print normalization info
    print(f"\n=== Normalization Info ===")
    print(f"Cooperation: min={coop_min:.2f}, max={coop_max:.2f}")
    print(f"Reward: min={reward_min:.2f}, max={reward_max:.2f}")
    print(f"Reward=0 normalized position: {reward_zero_norm:.3f}")

    plt.show()


def plot_summary_card(df, save_path=None):
    """요약 카드 형태의 차트"""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('학습 성과 요약 (초기 vs 최종)', fontsize=14, fontweight='bold')

    # 데이터 준비
    early = df[df['episode'] <= 100]
    final = df[df['episode'] >= df['episode'].max() - 100]

    metrics = [
        ('협동력 (탱커 근접도)',
         15 - early['avg_tank_dist'].mean(),
         15 - final['avg_tank_dist'].mean(),
         '#e74c3c', '점'),
        ('평균 보상',
         early['avg_reward'].mean(),
         final['avg_reward'].mean(),
         '#3498db', ''),
        ('A팀 승률',
         early['win_rate_a'].mean(),
         final['win_rate_a'].mean(),
         '#27ae60', '%'),
    ]

    for ax, (title, early_val, final_val, color, unit) in zip(axes, metrics):
        # 막대 그래프
        bars = ax.bar(['초기\n(Ep 1-100)', f'최종\n(Ep {int(df["episode"].max()-100)}-{int(df["episode"].max())})'],
                      [early_val, final_val],
                      color=[color+'50', color],
                      edgecolor=color, linewidth=2)

        # 값 표시
        for bar, val in zip(bars, [early_val, final_val]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{val:.1f}{unit}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

        # 변화량 화살표
        change = final_val - early_val
        arrow = '▲' if change > 0 else '▼'
        change_color = '#27ae60' if change > 0 else '#e74c3c'
        ax.text(0.5, max(early_val, final_val) * 1.3,
               f'{arrow} {abs(change):.1f}{unit}',
               ha='center', fontsize=14, fontweight='bold', color=change_color,
               transform=ax.get_xaxis_transform())

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, max(early_val, final_val) * 1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"차트 저장됨: {save_path}")

    plt.show()


def plot_combined(df, save_path=None):
    """단일 차트에 모든 지표 (정규화)"""

    fig, ax = plt.subplots(figsize=(14, 6))

    window = 30
    episodes = df['episode']

    # 정규화 (0-1 범위)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    # 협동력 (탱커 거리 반전)
    coop = normalize(15 - df['avg_tank_dist']).rolling(window=window, center=True).mean()
    reward = normalize(df['avg_reward']).rolling(window=window, center=True).mean()
    winrate = normalize(df['win_rate_a']).rolling(window=window, center=True).mean()

    ax.plot(episodes, coop, color='#e74c3c', linewidth=2.5, label='협동력 (탱커 근접도)')
    ax.plot(episodes, reward, color='#3498db', linewidth=2.5, label='평균 보상')
    ax.plot(episodes, winrate, color='#27ae60', linewidth=2.5, label='A팀 승률')

    ax.fill_between(episodes, 0, coop, alpha=0.1, color='#e74c3c')
    ax.fill_between(episodes, 0, reward, alpha=0.1, color='#3498db')
    ax.fill_between(episodes, 0, winrate, alpha=0.1, color='#27ae60')

    ax.set_xlabel('에피소드 (Episode)', fontsize=12, fontweight='bold')
    ax.set_ylabel('정규화된 성능 (0 = 최저, 1 = 최고)', fontsize=12, fontweight='bold')
    ax.set_title('학습 진행에 따른 성능 지표 변화 (정규화)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 학습 단계 배경색
    ax.axvspan(0, 500, alpha=0.1, color='red', label='_초기')
    ax.axvspan(500, 3000, alpha=0.1, color='orange', label='_적응')
    ax.axvspan(3000, 8000, alpha=0.1, color='yellow', label='_성숙')
    ax.axvspan(8000, episodes.max(), alpha=0.1, color='green', label='_완성')

    # 단계 텍스트
    ax.text(250, 1.05, '초기', ha='center', fontsize=10, color='red')
    ax.text(1750, 1.05, '적응기', ha='center', fontsize=10, color='orange')
    ax.text(5500, 1.05, '성숙기', ha='center', fontsize=10, color='#9a7b00')
    ax.text(10000, 1.05, '완성', ha='center', fontsize=10, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"차트 저장됨: {save_path}")

    plt.show()


def print_summary(df):
    """학습 결과 요약 출력"""

    print("\n" + "="*60)
    print("📊 학습 결과 요약")
    print("="*60)

    early = df[df['episode'] <= 100]
    final = df[df['episode'] >= df['episode'].max() - 100]

    print(f"\n{'지표':<20} {'초기':>12} {'최종':>12} {'변화':>12}")
    print("-"*60)

    # 탱커 거리 (협동력)
    e_dist = early['avg_tank_dist'].mean()
    f_dist = final['avg_tank_dist'].mean()
    print(f"{'탱커 거리':<20} {e_dist:>12.2f} {f_dist:>12.2f} {'▼ '+str(round(e_dist-f_dist,2)):>12}")

    # 협동력 (변환값)
    e_coop = 15 - e_dist
    f_coop = 15 - f_dist
    print(f"{'협동력 (15-거리)':<20} {e_coop:>12.2f} {f_coop:>12.2f} {'▲ '+str(round(f_coop-e_coop,2)):>12}")

    # 평균 보상
    e_reward = early['avg_reward'].mean()
    f_reward = final['avg_reward'].mean()
    print(f"{'평균 보상':<20} {e_reward:>12.2f} {f_reward:>12.2f} {'▲ '+str(round(f_reward-e_reward,2)):>12}")

    # 승률
    e_win = early['win_rate_a'].mean()
    f_win = final['win_rate_a'].mean()
    print(f"{'A팀 승률 (%)':<20} {e_win:>12.2f} {f_win:>12.2f} {'▲ '+str(round(f_win-e_win,2))+'%p':>12}")

    print("="*60)


if __name__ == "__main__":
    csv_path = "training_data_v11.csv"

    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        exit(1)

    print(f"📂 데이터 로드 중: {csv_path}")
    df = load_training_data(csv_path)
    print(f"✅ 총 {len(df)}개 에피소드 데이터 로드됨\n")

    # 요약 출력
    print_summary(df)

    # 차트 생성
    print("\n📈 차트 생성 중...")

    # 1. 메인 차트 (3개 서브플롯)
    plot_main_chart(df, save_path="chart_main.png")

    # 2. 요약 카드
    plot_summary_card(df, save_path="chart_summary.png")

    # 3. 정규화 통합 차트
    plot_combined(df, save_path="chart_combined.png")

    print("\n✅ 완료!")
