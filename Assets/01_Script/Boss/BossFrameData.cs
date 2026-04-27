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
        public EventData[] events;
        public bool done;
        public bool victory;
        public bool wipe;
    }

    [Serializable]
    public class EventData
    {
        public int uid;
        public string type;      // "damage", "heal", "taunt", "guard", "buff", "cleanse", "death", "damage_taken"
        public int amount;       // optional
        public int target;       // optional (heal/buff target)
        public bool skill;       // optional (damage의 skill 여부)
        public string kind;      // optional (buff의 "atk"/"shield")
    }

    [Serializable]
    public class BossData
    {
        public float x;                 // 유클리드 float 좌표
        public float y;
        public float vx;
        public float vy;
        public int hp;
        public int max_hp;
        public int phase;               // 0=P1, 1=P2, 2=P3
        public int invuln;
        public int grog;
        public bool stagger_active;
        public float stagger_gauge;
        public float radius;
    }

    [Serializable]
    public class UnitData
    {
        public int uid;
        public int role;                // 0=Dealer, 1=Tank, 2=Healer, 3=Support
        public float x;                 // 유클리드 float 좌표
        public float y;
        public float vx;
        public float vy;
        public int hp;
        public int max_hp;
        public bool alive;
        public bool marked;
        public int chained_with;        // -1이면 없음
        public int buff_atk;
        public int buff_shield;
        public float radius;
    }

    /// <summary>
    /// 패턴 위험 영역 기하 도형.
    /// kind에 따라 params의 어떤 키를 읽을지 결정.
    /// - "circle": cx, cy, r
    /// - "fan":    cx, cy, r, angle(rad), width(rad)
    /// - "line":   ax, ay, bx, by, hw
    /// - "cross":  cx, cy, hw, safe_mask (bit 0~3: 안전 사분면)
    /// </summary>
    [Serializable]
    public class ShapeData
    {
        public string kind;
        public float cx, cy, r;
        public float angle, width;
        public float ax, ay, bx, by;
        public float hw;
        public float safe_mask;
    }

    [Serializable]
    public class TelegraphData
    {
        public int pattern;
        public int turns_remaining;
        public int total_wind_up;
        public ShapeData[] shapes;      // 기하 도형 리스트
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
        SealBreak = 8,
    }

    public enum PartyRole
    {
        Dealer = 0,
        Tank = 1,
        Healer = 2,
        Support = 3,
    }

    // Python BossActionID 와 인덱스 일치 (8방향 이동 도입 후 ID 재정렬)
    public enum BossActionId
    {
        Stay = 0,
        MoveUp = 1,
        MoveDown = 2,
        MoveLeft = 3,
        MoveRight = 4,
        MoveUpLeft = 5,
        MoveUpRight = 6,
        MoveDownLeft = 7,
        MoveDownRight = 8,
        AttackBasic = 9,
        AttackSkill = 10,
        Taunt = 11,
        Guard = 12,
        Heal = 13,
        Cleanse = 14,
        BuffAtk = 15,
        BuffShield = 16,
    }

    [Serializable]
    public class PlayerInputMessage
    {
        public int action;
        public PlayerInputMessage(int a) { action = a; }
    }
}
