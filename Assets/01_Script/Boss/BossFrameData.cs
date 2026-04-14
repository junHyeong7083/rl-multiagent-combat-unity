using System;
using System.Collections.Generic;

namespace BossRaid
{
    /// <summary>
    /// Python boss_streamer.py 의 get_snapshot() JSON과 1:1 매칭되는 DTO.
    /// JsonUtility는 Dictionary를 지원하지 않으므로 배열 기반으로 구성.
    /// </summary>
    [Serializable]
    public class BossSnapshot
    {
        public int step;
        public BossData boss;
        public UnitData[] units;
        public TelegraphData[] telegraphs;
        public bool done;
        public bool victory;
        public bool wipe;
    }

    [Serializable]
    public class BossData
    {
        public int x;
        public int y;
        public int hp;
        public int max_hp;
        public int phase;           // 0=P1, 1=P2, 2=P3
        public int invuln;
        public int grog;
        public bool stagger_active;
        public float stagger_gauge;
    }

    [Serializable]
    public class UnitData
    {
        public int uid;
        public int role;            // 0=Dealer, 1=Tank, 2=Healer, 3=Support
        public int x;
        public int y;
        public int hp;
        public int max_hp;
        public bool alive;
        public bool marked;
        public int chained_with;    // -1이면 없음 (Python에서 None→JSON null이지만 편의상 int 처리)
        public int buff_atk;
        public int buff_shield;
    }

    [Serializable]
    public class TelegraphData
    {
        public int pattern;             // PatternID
        public int turns_remaining;
        public int total_wind_up;
        public int[][] danger_tiles;    // [[x,y], ...]
        public int[] target_uids;
    }

    public enum BossPatternId
    {
        Slash = 0,
        Charge = 1,
        Eruption = 2,
        TailSwipe = 3,
        Mark = 4,
        Stagger = 5,
        CrossInferno = 6,
        CursedChain = 7,
    }

    public enum PartyRole
    {
        Dealer = 0,
        Tank = 1,
        Healer = 2,
        Support = 3,
    }

    public enum BossActionId
    {
        Stay = 0,
        MoveUp = 1,
        MoveDown = 2,
        MoveLeft = 3,
        MoveRight = 4,
        AttackBasic = 5,
        AttackSkill = 6,
        Taunt = 7,
        Guard = 8,
        Heal = 9,
        Cleanse = 10,
        BuffAtk = 11,
        BuffShield = 12,
    }

    [Serializable]
    public class PlayerInputMessage
    {
        public int action;
        public PlayerInputMessage(int a) { action = a; }
    }
}
