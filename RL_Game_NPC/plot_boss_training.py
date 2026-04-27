"""Boss Raid 학습 그래프 생성기

사용법:
    python plot_boss_training.py                      # boss/v1 기본 경로
    python plot_boss_training.py --run-name v2
    python plot_boss_training.py --dir boss/v1        # 직접 지정
    python plot_boss_training.py --smooth 100         # 이동평균 윈도우 크기

출력 PNG:
    boss/<run-name>/plots/
        main.png         (보상 + 승률 + 스텝 + 보스HP잔량)
        by_role.png      (역할별 보상·딜·힐 추이)
        mechanics.png    (기믹 성공률 · 스태거 · 페이즈 도달)
        combined.png     (한 장 요약)
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(y, window):
    """Simple moving average. 양 끝은 window 축소로 계산."""
    if window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def plot_main(df, out_path, smooth_w):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ep = df["episode"]

    # 1. 평균 보상
    ax = axes[0, 0]
    ax.plot(ep, df["avg_reward_all"], alpha=0.25, color="gray", label="raw")
    ax.plot(ep, smooth(df["avg_reward_all"], smooth_w), color="#1f77b4", label=f"ma({smooth_w})")
    ax.set_title("Average NPC Reward per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.grid(alpha=0.3); ax.legend()

    # 2. 결과별 누적 비율 (Victory / Wipe / Timeout)
    ax = axes[0, 1]
    for result, color in [("VICTORY", "#2ca02c"), ("WIPE", "#d62728"), ("TIMEOUT", "#ff7f0e")]:
        is_r = (df["result"] == result).astype(int)
        cum_rate = smooth(is_r, smooth_w) * 100
        ax.plot(ep, cum_rate, label=result, color=color)
    ax.set_title(f"Result Rate (moving avg {smooth_w}ep)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Rate (%)")
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3); ax.legend()

    # 3. 스텝 수 (생존 시간)
    ax = axes[1, 0]
    ax.plot(ep, df["steps"], alpha=0.2, color="gray")
    ax.plot(ep, smooth(df["steps"], smooth_w), color="#9467bd")
    ax.set_title("Episode Length (steps)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps"); ax.grid(alpha=0.3)

    # 4. 보스 HP 잔량 비율 (낮을수록 딜을 많이 넣음)
    ax = axes[1, 1]
    ax.plot(ep, df["boss_hp_ratio"] * 100, alpha=0.2, color="gray")
    ax.plot(ep, smooth(df["boss_hp_ratio"] * 100, smooth_w), color="#e377c2")
    ax.set_title("Boss HP Remaining at Episode End")
    ax.set_xlabel("Episode"); ax.set_ylabel("HP Remaining (%)")
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_by_role(df, out_path, smooth_w):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ep = df["episode"]

    # 1. 역할별 보상 추이
    ax = axes[0, 0]
    for col, color, label in [
        ("reward_dealer", "#ffcc00", "Dealer"),
        ("reward_tank", "#1f77b4", "Tank"),
        ("reward_healer", "#2ca02c", "Healer"),
        ("reward_support", "#ff69b4", "Support"),
    ]:
        ax.plot(ep, smooth(df[col], smooth_w), label=label, color=color)
    ax.set_title(f"Reward by Role (ma {smooth_w})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.grid(alpha=0.3); ax.legend()

    # 2. 역할별 딜량 (Tank/Healer/Support)
    ax = axes[0, 1]
    for col, color, label in [
        ("damage_tank", "#1f77b4", "Tank"),
        ("damage_healer", "#2ca02c", "Healer"),
        ("damage_support", "#ff69b4", "Support"),
    ]:
        ax.plot(ep, smooth(df[col], smooth_w), label=label, color=color)
    ax.set_title(f"Damage Dealt per Role (ma {smooth_w})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Damage")
    ax.grid(alpha=0.3); ax.legend()

    # 3. 힐러 총 힐량
    ax = axes[1, 0]
    ax.plot(ep, smooth(df["heal_done_healer"], smooth_w), color="#2ca02c")
    ax.set_title("Healer: Total Heal per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("HP Healed")
    ax.grid(alpha=0.3)

    # 4. 서포터 버프 횟수
    ax = axes[1, 1]
    ax.plot(ep, smooth(df["buffs_support"], smooth_w), color="#ff69b4")
    ax.set_title("Support: Buffs Applied per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Buffs")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mechanics(df, out_path, smooth_w):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ep = df["episode"]

    # 1. 기믹 성공/실패
    ax = axes[0, 0]
    ax.plot(ep, smooth(df["mechanic_success"], smooth_w), color="#2ca02c", label="Success")
    ax.plot(ep, smooth(df["mechanic_fail"], smooth_w), color="#d62728", label="Fail")
    ax.set_title(f"Mechanic (Mark) Success/Fail Count (ma {smooth_w})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Count")
    ax.grid(alpha=0.3); ax.legend()

    # 2. 기믹 성공률 (%) — 시도 중에서
    ax = axes[0, 1]
    total = df["mechanic_success"] + df["mechanic_fail"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(total > 0, df["mechanic_success"] / total * 100, np.nan)
    ax.plot(ep, smooth(np.nan_to_num(rate, nan=0.0), smooth_w), color="#2ca02c")
    ax.set_title(f"Mechanic Success Rate (ma {smooth_w})")
    ax.set_xlabel("Episode"); ax.set_ylabel("%")
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3)

    # 3. 스태거 성공/실패
    ax = axes[1, 0]
    ax.plot(ep, smooth(df["stagger_success"], smooth_w), color="#2ca02c", label="Success")
    ax.plot(ep, smooth(df["stagger_fail"], smooth_w), color="#d62728", label="Fail")
    ax.set_title(f"Stagger Check Success/Fail (ma {smooth_w})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Count")
    ax.grid(alpha=0.3); ax.legend()

    # 4. 최대 도달 페이즈
    ax = axes[1, 1]
    ax.plot(ep, df["phase_reached"], alpha=0.15, color="gray")
    ax.plot(ep, smooth(df["phase_reached"].astype(float), smooth_w), color="#9467bd")
    ax.set_title(f"Max Phase Reached (ma {smooth_w})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Phase (0=P1, 1=P2, 2=P3)")
    ax.set_yticks([0, 1, 2], ["P1", "P2", "P3"])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_combined_summary(df, out_path, smooth_w):
    """논문 발표용 한 장 요약."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ep = df["episode"]

    axes[0, 0].plot(ep, smooth(df["avg_reward_all"], smooth_w), color="#1f77b4")
    axes[0, 0].set_title("Average NPC Reward"); axes[0, 0].grid(alpha=0.3)

    victory = (df["result"] == "VICTORY").astype(int)
    axes[0, 1].plot(ep, smooth(victory, smooth_w) * 100, color="#2ca02c")
    axes[0, 1].set_title("Victory Rate (%)"); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim(-5, 105)

    axes[0, 2].plot(ep, smooth(df["boss_hp_ratio"] * 100, smooth_w), color="#e377c2")
    axes[0, 2].set_title("Boss HP Remaining (%)"); axes[0, 2].grid(alpha=0.3)
    axes[0, 2].set_ylim(-5, 105)

    # 역할별 보상
    for col, color, label in [
        ("reward_tank", "#1f77b4", "Tank"),
        ("reward_healer", "#2ca02c", "Healer"),
        ("reward_support", "#ff69b4", "Support"),
    ]:
        axes[1, 0].plot(ep, smooth(df[col], smooth_w), label=label, color=color)
    axes[1, 0].set_title("Reward by Role (NPCs)"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    total_mech = df["mechanic_success"] + df["mechanic_fail"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mech_rate = np.where(total_mech > 0, df["mechanic_success"] / total_mech * 100, np.nan)
    axes[1, 1].plot(ep, smooth(np.nan_to_num(mech_rate, nan=0.0), smooth_w), color="#9467bd")
    axes[1, 1].set_title("Mark Mechanic Success Rate (%)"); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim(-5, 105)

    axes[1, 2].plot(ep, smooth(df["steps"], smooth_w), color="#ff7f0e")
    axes[1, 2].set_title("Episode Length (steps)"); axes[1, 2].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Episode")

    fig.suptitle(f"Boss Raid Training Summary (smoothed {smooth_w}ep)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="v1")
    ap.add_argument("--dir", type=str, default=None,
                    help="직접 경로 지정 (예: boss/v1). 기본은 boss/<run-name>")
    ap.add_argument("--smooth", type=int, default=50, help="이동평균 윈도우")
    args = ap.parse_args()

    base = args.dir if args.dir else os.path.join("boss", args.run_name)
    log_path = os.path.join(base, "training_log.csv")
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print(f"[ERR] 로그 파일 없음: {log_path}")
        return

    df = pd.read_csv(log_path)
    if len(df) < 2:
        print(f"[WARN] 에피소드 수 부족: {len(df)}")
        return

    print(f"Loaded {len(df)} episodes from {log_path}")
    w = args.smooth

    plot_main(df, os.path.join(plots_dir, "main.png"), w)
    plot_by_role(df, os.path.join(plots_dir, "by_role.png"), w)
    plot_mechanics(df, os.path.join(plots_dir, "mechanics.png"), w)
    plot_combined_summary(df, os.path.join(plots_dir, "combined.png"), w)

    print(f"Plots saved to {plots_dir}/")
    print("  - main.png       (보상 / 결과 / 스텝 / 보스HP)")
    print("  - by_role.png    (역할별 추이)")
    print("  - mechanics.png  (기믹 성공률)")
    print("  - combined.png   (한 장 요약)")


if __name__ == "__main__":
    main()
