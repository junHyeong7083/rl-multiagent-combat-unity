using System;

/// <summary>
/// Python RL_Game_NPC 환경에서 전송하는 유닛 정보
/// </summary>
[Serializable]
public class UnitData
{
    public int x;
    public int y;
    public int hp;
    public int maxHp;
    public int mp;
    public int maxMp;
    public int role;      // 0=Tank, 1=Dealer, 2=Healer, 3=Ranger, 4=Support
    public bool alive;
    public bool isPlayer; // 플레이어가 조종하는 유닛 여부
}

/// <summary>
/// Python RL_Game_NPC 환경에서 전송하는 프레임 데이터
/// </summary>
[Serializable]
public class FrameData
{
    public int step;
    public int mapWidth;
    public int mapHeight;
    public int[] tiles;       // mapHeight * mapWidth 1D array (row-major)
    public UnitData[] teamA;  // 5 units
    public UnitData[] teamB;  // 5 units
    public bool done;
    public string winner;     // "A", "B", "draw", or ""
    public int playerIdx;     // 플레이어가 조종하는 유닛 인덱스 (-1이면 관전 모드)
}

/// <summary>
/// 타일 타입 (Python TileType과 동일)
/// </summary>
public enum TileType
{
    Empty = 0,
    Wall = 1,
    Danger = 2,
    BuffAtk = 3,
    BuffDef = 4,
    BuffHeal = 5
}

/// <summary>
/// 역할 타입 (Python RoleType과 동일)
/// </summary>
public enum RoleType
{
    Tank = 0,
    Dealer = 1,
    Healer = 2,
    Ranger = 3,
    Support = 4
}
