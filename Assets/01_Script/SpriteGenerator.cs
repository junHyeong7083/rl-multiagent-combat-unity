using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 런타임에 역할별 스프라이트를 생성하는 유틸리티
/// </summary>
public static class SpriteGenerator
{
    private static Dictionary<string, Sprite> _cachedSprites = new Dictionary<string, Sprite>();

    // 스프라이트 크기
    private const int SIZE = 64;
    private const int HALF = SIZE / 2;

    /// <summary>
    /// 역할별 스프라이트 생성
    /// </summary>
    public static Sprite GetRoleSprite(RoleType role, bool isTeamA)
    {
        string key = $"{role}_{(isTeamA ? "A" : "B")}";

        if (_cachedSprites.TryGetValue(key, out Sprite cached))
        {
            return cached;
        }

        // 팀 베이스 색상: A팀 = 파랑 계열, B팀 = 빨강 계열
        // 역할별 색조 변화
        Color teamColor = GetRoleColor(role, isTeamA);
        Color outlineColor = isTeamA
            ? new Color(teamColor.r * 0.5f, teamColor.g * 0.5f, teamColor.b * 0.7f)
            : new Color(teamColor.r * 0.7f, teamColor.g * 0.5f, teamColor.b * 0.5f);

        Texture2D tex = role switch
        {
            RoleType.Tank => CreateTankSprite(teamColor, outlineColor),
            RoleType.Dealer => CreateDealerSprite(teamColor, outlineColor),
            RoleType.Healer => CreateHealerSprite(teamColor, outlineColor),
            RoleType.Ranger => CreateRangerSprite(teamColor, outlineColor),
            RoleType.Support => CreateSupportSprite(teamColor, outlineColor),
            _ => CreateTankSprite(teamColor, outlineColor)
        };

        Sprite sprite = Sprite.Create(tex, new Rect(0, 0, SIZE, SIZE), new Vector2(0.5f, 0.5f), SIZE);
        _cachedSprites[key] = sprite;

        return sprite;
    }

    /// <summary>
    /// Tank: 방패 들고 있는 사각형 캐릭터
    /// </summary>
    private static Texture2D CreateTankSprite(Color main, Color outline)
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        ClearTexture(tex);

        // 몸체 (큰 사각형)
        FillRect(tex, 16, 8, 32, 48, main);
        DrawRectOutline(tex, 16, 8, 32, 48, outline, 2);

        // 방패 (왼쪽에 큰 방패)
        FillRect(tex, 4, 16, 14, 32, new Color(0.6f, 0.6f, 0.6f));
        DrawRectOutline(tex, 4, 16, 14, 32, new Color(0.4f, 0.4f, 0.4f), 2);
        // 방패 문양
        FillRect(tex, 8, 24, 6, 16, new Color(0.8f, 0.7f, 0.2f));

        // 헬멧 (머리 위)
        FillRect(tex, 20, 50, 24, 10, new Color(0.5f, 0.5f, 0.5f));
        DrawRectOutline(tex, 20, 50, 24, 10, new Color(0.3f, 0.3f, 0.3f), 1);

        // 눈
        FillRect(tex, 24, 40, 4, 4, Color.white);
        FillRect(tex, 36, 40, 4, 4, Color.white);
        FillRect(tex, 25, 41, 2, 2, Color.black);
        FillRect(tex, 37, 41, 2, 2, Color.black);

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Dealer: 검을 든 삼각형 캐릭터
    /// </summary>
    private static Texture2D CreateDealerSprite(Color main, Color outline)
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        ClearTexture(tex);

        // 몸체 (삼각형 느낌의 날렵한 형태)
        FillTriangle(tex, HALF, 56, 12, 8, 52, 8, main);
        DrawTriangleOutline(tex, HALF, 56, 12, 8, 52, 8, outline, 2);

        // 검 (오른쪽 위로 뻗은 검)
        // 검 날
        for (int i = 0; i < 20; i++)
        {
            int x = 44 + i;
            int y = 40 + i;
            if (x < SIZE && y < SIZE)
            {
                FillRect(tex, x, y, 3, 3, new Color(0.8f, 0.8f, 0.9f));
            }
        }
        // 검 손잡이
        FillRect(tex, 40, 36, 6, 6, new Color(0.5f, 0.3f, 0.1f));

        // 눈 (날카로운 눈)
        FillRect(tex, 24, 36, 6, 3, Color.white);
        FillRect(tex, 34, 36, 6, 3, Color.white);
        FillRect(tex, 27, 36, 3, 3, Color.black);
        FillRect(tex, 37, 36, 3, 3, Color.black);

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Healer: 십자가가 있는 원형 캐릭터
    /// </summary>
    private static Texture2D CreateHealerSprite(Color main, Color outline)
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        ClearTexture(tex);

        // 몸체 (원형)
        FillCircle(tex, HALF, HALF, 24, main);
        DrawCircleOutline(tex, HALF, HALF, 24, outline, 2);

        // 십자가 (힐러 상징)
        Color crossColor = new Color(1f, 1f, 1f);
        FillRect(tex, 28, 20, 8, 24, crossColor);  // 세로
        FillRect(tex, 20, 28, 24, 8, crossColor);  // 가로

        // 십자가 테두리
        Color crossOutline = new Color(0.2f, 0.8f, 0.2f);
        DrawRectOutline(tex, 28, 20, 8, 24, crossOutline, 1);
        DrawRectOutline(tex, 20, 28, 24, 8, crossOutline, 1);

        // 부드러운 눈
        FillCircle(tex, 22, 40, 4, Color.white);
        FillCircle(tex, 42, 40, 4, Color.white);
        FillCircle(tex, 22, 40, 2, Color.black);
        FillCircle(tex, 42, 40, 2, Color.black);

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Ranger: 활을 든 다이아몬드 캐릭터
    /// </summary>
    private static Texture2D CreateRangerSprite(Color main, Color outline)
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        ClearTexture(tex);

        // 몸체 (다이아몬드)
        FillDiamond(tex, HALF, HALF, 22, main);
        DrawDiamondOutline(tex, HALF, HALF, 22, outline, 2);

        // 활 (왼쪽에 곡선 활)
        Color bowColor = new Color(0.6f, 0.4f, 0.2f);
        for (int i = 0; i < 30; i++)
        {
            float angle = (i / 30f) * Mathf.PI - Mathf.PI / 2;
            int x = 10 + (int)(8 * Mathf.Cos(angle));
            int y = HALF + (int)(15 * Mathf.Sin(angle));
            if (x >= 0 && x < SIZE && y >= 0 && y < SIZE)
            {
                FillRect(tex, x, y, 3, 3, bowColor);
            }
        }

        // 화살 (오른쪽으로 향하는 화살)
        FillRect(tex, 20, 30, 30, 4, new Color(0.5f, 0.3f, 0.1f));  // 화살대
        // 화살촉
        FillTriangle(tex, 54, 32, 48, 26, 48, 38, new Color(0.7f, 0.7f, 0.7f));

        // 눈
        FillRect(tex, 26, 38, 4, 4, Color.white);
        FillRect(tex, 38, 38, 4, 4, Color.white);
        FillRect(tex, 27, 39, 2, 2, Color.black);
        FillRect(tex, 39, 39, 2, 2, Color.black);

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Support: 별 모양 캐릭터
    /// </summary>
    private static Texture2D CreateSupportSprite(Color main, Color outline)
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        ClearTexture(tex);

        // 몸체 (오각형/별 느낌)
        FillPentagon(tex, HALF, HALF, 24, main);
        DrawPentagonOutline(tex, HALF, HALF, 24, outline, 2);

        // 마법 지팡이 (오른쪽)
        Color staffColor = new Color(0.6f, 0.4f, 0.8f);
        FillRect(tex, 48, 10, 4, 40, staffColor);
        // 지팡이 끝 보석
        FillCircle(tex, 50, 52, 6, new Color(1f, 0.8f, 0.2f));
        DrawCircleOutline(tex, 50, 52, 6, new Color(0.8f, 0.6f, 0.1f), 1);

        // 버프 이펙트 (작은 별들)
        DrawStar(tex, 12, 50, 4, new Color(1f, 1f, 0.5f));
        DrawStar(tex, 20, 56, 3, new Color(0.5f, 1f, 1f));

        // 눈
        FillCircle(tex, 24, 38, 4, Color.white);
        FillCircle(tex, 40, 38, 4, Color.white);
        FillCircle(tex, 24, 38, 2, Color.black);
        FillCircle(tex, 40, 38, 2, Color.black);

        tex.Apply();
        return tex;
    }

    #region Role Colors

    /// <summary>
    /// 팀별 색상 반환
    /// A팀: 파랑, B팀: 빨강
    /// </summary>
    private static Color GetRoleColor(RoleType role, bool isTeamA)
    {
        if (isTeamA)
        {
            // A팀 - 파랑
            return new Color(0.2f, 0.4f, 1.0f);
        }
        else
        {
            // B팀 - 빨강
            return new Color(1.0f, 0.2f, 0.2f);
        }
    }

    /// <summary>
    /// 캐시 초기화 (에디터에서 색상 변경 시 호출)
    /// </summary>
    public static void ClearCache()
    {
        _cachedSprites.Clear();
    }

    #endregion

    #region Drawing Helpers

    private static void ClearTexture(Texture2D tex)
    {
        Color[] clear = new Color[tex.width * tex.height];
        for (int i = 0; i < clear.Length; i++)
            clear[i] = Color.clear;
        tex.SetPixels(clear);
    }

    private static void FillRect(Texture2D tex, int x, int y, int w, int h, Color color)
    {
        for (int py = y; py < y + h && py < tex.height; py++)
        {
            for (int px = x; px < x + w && px < tex.width; px++)
            {
                if (px >= 0 && py >= 0)
                    tex.SetPixel(px, py, color);
            }
        }
    }

    private static void DrawRectOutline(Texture2D tex, int x, int y, int w, int h, Color color, int thickness)
    {
        for (int t = 0; t < thickness; t++)
        {
            for (int px = x; px < x + w; px++)
            {
                if (px >= 0 && px < tex.width)
                {
                    if (y + t >= 0 && y + t < tex.height) tex.SetPixel(px, y + t, color);
                    if (y + h - 1 - t >= 0 && y + h - 1 - t < tex.height) tex.SetPixel(px, y + h - 1 - t, color);
                }
            }
            for (int py = y; py < y + h; py++)
            {
                if (py >= 0 && py < tex.height)
                {
                    if (x + t >= 0 && x + t < tex.width) tex.SetPixel(x + t, py, color);
                    if (x + w - 1 - t >= 0 && x + w - 1 - t < tex.width) tex.SetPixel(x + w - 1 - t, py, color);
                }
            }
        }
    }

    private static void FillCircle(Texture2D tex, int cx, int cy, int radius, Color color)
    {
        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                if (x * x + y * y <= radius * radius)
                {
                    int px = cx + x;
                    int py = cy + y;
                    if (px >= 0 && px < tex.width && py >= 0 && py < tex.height)
                        tex.SetPixel(px, py, color);
                }
            }
        }
    }

    private static void DrawCircleOutline(Texture2D tex, int cx, int cy, int radius, Color color, int thickness)
    {
        for (int y = -radius - thickness; y <= radius + thickness; y++)
        {
            for (int x = -radius - thickness; x <= radius + thickness; x++)
            {
                float dist = Mathf.Sqrt(x * x + y * y);
                if (dist >= radius - thickness && dist <= radius + thickness)
                {
                    int px = cx + x;
                    int py = cy + y;
                    if (px >= 0 && px < tex.width && py >= 0 && py < tex.height)
                        tex.SetPixel(px, py, color);
                }
            }
        }
    }

    private static void FillTriangle(Texture2D tex, int x1, int y1, int x2, int y2, int x3, int y3, Color color)
    {
        int minX = Mathf.Min(x1, Mathf.Min(x2, x3));
        int maxX = Mathf.Max(x1, Mathf.Max(x2, x3));
        int minY = Mathf.Min(y1, Mathf.Min(y2, y3));
        int maxY = Mathf.Max(y1, Mathf.Max(y2, y3));

        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                if (PointInTriangle(x, y, x1, y1, x2, y2, x3, y3))
                {
                    if (x >= 0 && x < tex.width && y >= 0 && y < tex.height)
                        tex.SetPixel(x, y, color);
                }
            }
        }
    }

    private static void DrawTriangleOutline(Texture2D tex, int x1, int y1, int x2, int y2, int x3, int y3, Color color, int thickness)
    {
        DrawLine(tex, x1, y1, x2, y2, color, thickness);
        DrawLine(tex, x2, y2, x3, y3, color, thickness);
        DrawLine(tex, x3, y3, x1, y1, color, thickness);
    }

    private static void DrawLine(Texture2D tex, int x1, int y1, int x2, int y2, Color color, int thickness)
    {
        int dx = Mathf.Abs(x2 - x1);
        int dy = Mathf.Abs(y2 - y1);
        int steps = Mathf.Max(dx, dy);

        for (int i = 0; i <= steps; i++)
        {
            float t = steps == 0 ? 0 : (float)i / steps;
            int x = Mathf.RoundToInt(Mathf.Lerp(x1, x2, t));
            int y = Mathf.RoundToInt(Mathf.Lerp(y1, y2, t));

            for (int ty = -thickness / 2; ty <= thickness / 2; ty++)
            {
                for (int tx = -thickness / 2; tx <= thickness / 2; tx++)
                {
                    int px = x + tx;
                    int py = y + ty;
                    if (px >= 0 && px < tex.width && py >= 0 && py < tex.height)
                        tex.SetPixel(px, py, color);
                }
            }
        }
    }

    private static bool PointInTriangle(int px, int py, int x1, int y1, int x2, int y2, int x3, int y3)
    {
        float d1 = Sign(px, py, x1, y1, x2, y2);
        float d2 = Sign(px, py, x2, y2, x3, y3);
        float d3 = Sign(px, py, x3, y3, x1, y1);

        bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        return !(hasNeg && hasPos);
    }

    private static float Sign(int px, int py, int x1, int y1, int x2, int y2)
    {
        return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
    }

    private static void FillDiamond(Texture2D tex, int cx, int cy, int size, Color color)
    {
        for (int y = -size; y <= size; y++)
        {
            for (int x = -size; x <= size; x++)
            {
                if (Mathf.Abs(x) + Mathf.Abs(y) <= size)
                {
                    int px = cx + x;
                    int py = cy + y;
                    if (px >= 0 && px < tex.width && py >= 0 && py < tex.height)
                        tex.SetPixel(px, py, color);
                }
            }
        }
    }

    private static void DrawDiamondOutline(Texture2D tex, int cx, int cy, int size, Color color, int thickness)
    {
        DrawLine(tex, cx, cy - size, cx + size, cy, color, thickness);
        DrawLine(tex, cx + size, cy, cx, cy + size, color, thickness);
        DrawLine(tex, cx, cy + size, cx - size, cy, color, thickness);
        DrawLine(tex, cx - size, cy, cx, cy - size, color, thickness);
    }

    private static void FillPentagon(Texture2D tex, int cx, int cy, int radius, Color color)
    {
        float angleOffset = Mathf.PI / 2;
        int[] xPoints = new int[5];
        int[] yPoints = new int[5];

        for (int i = 0; i < 5; i++)
        {
            float angle = angleOffset + i * 2 * Mathf.PI / 5;
            xPoints[i] = cx + (int)(radius * Mathf.Cos(angle));
            yPoints[i] = cy + (int)(radius * Mathf.Sin(angle));
        }

        // 삼각형들로 채우기
        for (int i = 1; i < 4; i++)
        {
            FillTriangle(tex, xPoints[0], yPoints[0], xPoints[i], yPoints[i], xPoints[i + 1], yPoints[i + 1], color);
        }
    }

    private static void DrawPentagonOutline(Texture2D tex, int cx, int cy, int radius, Color color, int thickness)
    {
        float angleOffset = Mathf.PI / 2;
        int[] xPoints = new int[5];
        int[] yPoints = new int[5];

        for (int i = 0; i < 5; i++)
        {
            float angle = angleOffset + i * 2 * Mathf.PI / 5;
            xPoints[i] = cx + (int)(radius * Mathf.Cos(angle));
            yPoints[i] = cy + (int)(radius * Mathf.Sin(angle));
        }

        for (int i = 0; i < 5; i++)
        {
            int next = (i + 1) % 5;
            DrawLine(tex, xPoints[i], yPoints[i], xPoints[next], yPoints[next], color, thickness);
        }
    }

    private static void DrawStar(Texture2D tex, int cx, int cy, int size, Color color)
    {
        // 작은 4방향 별
        FillRect(tex, cx - size, cy, size * 2 + 1, 1, color);
        FillRect(tex, cx, cy - size, 1, size * 2 + 1, color);
        // 대각선
        for (int i = -size / 2; i <= size / 2; i++)
        {
            int px1 = cx + i;
            int py1 = cy + i;
            int px2 = cx + i;
            int py2 = cy - i;
            if (px1 >= 0 && px1 < tex.width && py1 >= 0 && py1 < tex.height)
                tex.SetPixel(px1, py1, color);
            if (px2 >= 0 && px2 < tex.width && py2 >= 0 && py2 < tex.height)
                tex.SetPixel(px2, py2, color);
        }
    }

    #endregion

    #region Tile Sprites

    /// <summary>
    /// 타일 타입별 스프라이트 생성
    /// </summary>
    public static Sprite GetTileSprite(TileType type)
    {
        string key = $"tile_{type}";

        if (_cachedSprites.TryGetValue(key, out Sprite cached))
        {
            return cached;
        }

        Texture2D tex = type switch
        {
            TileType.Wall => CreateWallSprite(),
            TileType.Danger => CreateDangerSprite(),
            TileType.BuffAtk => CreateBuffSprite(new Color(1f, 0.6f, 0.2f), "ATK"),
            TileType.BuffDef => CreateBuffSprite(new Color(0.3f, 0.7f, 1f), "DEF"),
            TileType.BuffHeal => CreateBuffSprite(new Color(0.3f, 1f, 0.4f), "HP"),
            _ => CreateEmptySprite()
        };

        Sprite sprite = Sprite.Create(tex, new Rect(0, 0, SIZE, SIZE), new Vector2(0.5f, 0.5f), SIZE);
        _cachedSprites[key] = sprite;

        return sprite;
    }

    /// <summary>
    /// Wall: 검정 벽 (외벽용)
    /// </summary>
    private static Texture2D CreateWallSprite()
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        Color wallColor = new Color(0.1f, 0.1f, 0.12f);  // 거의 검정
        Color edgeColor = new Color(0.2f, 0.2f, 0.25f);  // 약간 밝은 테두리

        // 배경 (검정)
        for (int y = 0; y < SIZE; y++)
        {
            for (int x = 0; x < SIZE; x++)
            {
                tex.SetPixel(x, y, wallColor);
            }
        }

        // 테두리 효과 (살짝 밝게)
        for (int i = 0; i < 2; i++)
        {
            DrawRectOutline(tex, i, i, SIZE - i * 2, SIZE - i * 2, edgeColor, 1);
        }

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Danger: 경고 패턴 (빨간색 + 느낌표)
    /// </summary>
    private static Texture2D CreateDangerSprite()
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        Color bgColor = new Color(0.8f, 0.2f, 0.2f, 0.7f);
        Color stripeColor = new Color(0.9f, 0.4f, 0.1f, 0.8f);

        // 배경
        for (int y = 0; y < SIZE; y++)
        {
            for (int x = 0; x < SIZE; x++)
            {
                // 대각선 스트라이프 패턴
                if ((x + y) % 16 < 8)
                    tex.SetPixel(x, y, bgColor);
                else
                    tex.SetPixel(x, y, stripeColor);
            }
        }

        // 가운데 느낌표
        Color white = Color.white;
        // 느낌표 몸통
        FillRect(tex, 28, 20, 8, 28, white);
        // 느낌표 점
        FillRect(tex, 28, 10, 8, 6, white);

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Buff: 버프 타일 (색상 + 텍스트)
    /// </summary>
    private static Texture2D CreateBuffSprite(Color color, string label)
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        Color bgColor = new Color(color.r * 0.3f, color.g * 0.3f, color.b * 0.3f, 0.5f);

        // 배경
        for (int y = 0; y < SIZE; y++)
        {
            for (int x = 0; x < SIZE; x++)
            {
                tex.SetPixel(x, y, bgColor);
            }
        }

        // 테두리 (빛나는 효과)
        for (int i = 0; i < 4; i++)
        {
            Color borderColor = new Color(color.r, color.g, color.b, 1f - i * 0.2f);
            DrawRectOutline(tex, i, i, SIZE - i * 2, SIZE - i * 2, borderColor, 1);
        }

        // 중앙 원
        FillCircle(tex, HALF, HALF, 20, new Color(color.r, color.g, color.b, 0.6f));
        DrawCircleOutline(tex, HALF, HALF, 20, color, 2);

        // 화살표 또는 심볼
        if (label == "ATK")
        {
            // 위쪽 화살표 (공격력 증가)
            FillTriangle(tex, HALF, HALF + 12, HALF - 10, HALF - 4, HALF + 10, HALF - 4, Color.white);
            FillRect(tex, HALF - 4, HALF - 12, 8, 10, Color.white);
        }
        else if (label == "DEF")
        {
            // 방패 모양
            FillRect(tex, HALF - 10, HALF - 8, 20, 16, Color.white);
            FillTriangle(tex, HALF, HALF + 14, HALF - 10, HALF + 8, HALF + 10, HALF + 8, Color.white);
        }
        else if (label == "HP")
        {
            // 십자가 (힐)
            FillRect(tex, HALF - 3, HALF - 12, 6, 24, Color.white);
            FillRect(tex, HALF - 12, HALF - 3, 24, 6, Color.white);
        }

        tex.Apply();
        return tex;
    }

    /// <summary>
    /// Empty: 빈 타일 (바닥)
    /// </summary>
    private static Texture2D CreateEmptySprite()
    {
        Texture2D tex = new Texture2D(SIZE, SIZE);
        Color floorColor = new Color(0.6f, 0.6f, 0.55f);
        Color lineColor = new Color(0.5f, 0.5f, 0.45f);

        // 배경
        for (int y = 0; y < SIZE; y++)
        {
            for (int x = 0; x < SIZE; x++)
            {
                tex.SetPixel(x, y, floorColor);
            }
        }

        // 격자 라인
        for (int i = 0; i < SIZE; i++)
        {
            tex.SetPixel(i, 0, lineColor);
            tex.SetPixel(i, SIZE - 1, lineColor);
            tex.SetPixel(0, i, lineColor);
            tex.SetPixel(SIZE - 1, i, lineColor);
        }

        tex.Apply();
        return tex;
    }

    #endregion
}