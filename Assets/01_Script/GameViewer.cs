using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Python RL_Game_NPC 환경을 Unity에서 시각화
/// </summary>
public class GameViewer : MonoBehaviour
{
    [Header("Network")]
    public UdpReceiver udpReceiver;

    [Header("Grid Settings")]
    [Tooltip("그리드 1칸의 월드 크기")]
    public float cellSize = 1f;
    [Tooltip("그리드 (0,0)의 월드 기준점")]
    public Vector3 originOffset = Vector3.zero;

    [Header("Auto Generation")]
    [Tooltip("프리팹 없이 자동 생성 (권장)")]
    public bool autoGenerate = true;

    [Header("Smoothing")]
    [Tooltip("위치 보간 속도 (0=즉시 이동)")]
    public float moveSmooth = 0f;  // 즉시 이동으로 변경 (떨림 방지)

    [Header("Debug")]
    public bool showDebugInfo = true;

    [Header("Camera")]
    [Tooltip("카메라 자동 조정")]
    public bool autoAdjustCamera = true;
    [Tooltip("카메라 높이 배율")]
    public float cameraHeightMultiplier = 1.2f;

    [Header("Grid Lines")]
    [Tooltip("그리드 라인 표시")]
    public bool showGridLines = true;
    [Tooltip("그리드 라인 색상")]
    public Color gridLineColor = new Color(0.3f, 0.3f, 0.3f, 0.8f);
    [Tooltip("그리드 라인 두께")]
    public float gridLineWidth = 0.02f;

    // 내부 상태
    private bool _cameraAdjusted = false;
    private readonly List<GameObject> _teamAUnits = new List<GameObject>();
    private readonly List<GameObject> _teamBUnits = new List<GameObject>();
    private readonly List<GameObject> _tileObjects = new List<GameObject>();
    private readonly List<GameObject> _gridLines = new List<GameObject>();

    private int _currentMapWidth;
    private int _currentMapHeight;
    private int _lastStep = -1;
    private bool _tilesGenerated = false;

    // 플레이어 상태
    private PlayerInputSender _playerInputSender;
    private bool _playerAlive = true;

    // 로딩 오버레이
    [Header("Loading Overlay")]
    public Color loadingBgColor = new Color(0.15f, 0.15f, 0.2f, 1f);
    public Color loadingCircleColor = new Color(0.4f, 0.7f, 1f, 1f);
    private Texture2D _loadingBgTexture;
    private Texture2D _loadingCircleTexture;
    private float _loadingProgress = 0.9f;
    private float _loadingDisplayProgress = 0.9f;
    private float _loadingStartTime;

    private void Awake()
    {
        // 스프라이트 캐시 초기화 (색상 변경 적용)
        SpriteGenerator.ClearCache();

        // 로딩 오버레이 초기화
        InitializeLoadingOverlay();
        _loadingStartTime = Time.time;
    }

    private void Start()
    {
        // Start에서 UdpReceiver 연결 (Awake에서 중복 제거가 완료된 후)
        if (udpReceiver == null)
        {
            // 싱글톤 인스턴스 확인
            if (UdpReceiver.Instance != null)
            {
                udpReceiver = UdpReceiver.Instance;
            }
            else
            {
                udpReceiver = FindObjectOfType<UdpReceiver>();
                if (udpReceiver == null)
                {
                    GameObject go = new GameObject("UdpReceiver");
                    udpReceiver = go.AddComponent<UdpReceiver>();
                }
            }
        }

        // PlayerInputSender 찾기/생성 (플레이어 모드일 때만)
        if (SelectMenu.IsPlayerMode)
        {
            _playerInputSender = FindObjectOfType<PlayerInputSender>();
            if (_playerInputSender == null)
            {
                GameObject inputObj = new GameObject("PlayerInputSender");
                _playerInputSender = inputObj.AddComponent<PlayerInputSender>();
            }
        }
    }

    private void InitializeLoadingOverlay()
    {
        // 배경 텍스처 생성
        _loadingBgTexture = new Texture2D(1, 1);
        _loadingBgTexture.SetPixel(0, 0, loadingBgColor);
        _loadingBgTexture.Apply();

        // 원형 프로그레스 텍스처 생성
        int size = 256;
        _loadingCircleTexture = new Texture2D(size, size, TextureFormat.RGBA32, false);
        float center = size / 2f;
        float outerRadius = size / 2f - 4;
        float innerRadius = outerRadius - 16;

        Color[] pixels = new Color[size * size];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = Color.clear;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float dist = Mathf.Sqrt((x - center) * (x - center) + (y - center) * (y - center));
                if (dist >= innerRadius && dist <= outerRadius)
                {
                    pixels[y * size + x] = Color.white;
                }
            }
        }

        _loadingCircleTexture.SetPixels(pixels);
        _loadingCircleTexture.Apply();
    }

    private void LateUpdate()
    {
        // 매 프레임 싱글톤 인스턴스 확인 (씬 전환 후 참조가 깨질 수 있음)
        if (udpReceiver == null || !udpReceiver.gameObject.activeInHierarchy)
        {
            if (UdpReceiver.Instance != null)
            {
                udpReceiver = UdpReceiver.Instance;
            }
            else
            {
                return;
            }
        }

        // 최신 프레임만 처리 (중간 프레임은 버림)
        FrameData latestFrame = null;
        while (udpReceiver.TryDequeue(out FrameData frame))
        {
            latestFrame = frame;
        }

        if (latestFrame != null)
        {
            ProcessFrame(latestFrame);
        }
    }

    // 타일 캐시 (타일 업데이트 최적화용)
    private int[] _cachedTiles;

    private void ProcessFrame(FrameData frame)
    {
        if (frame == null) return;

        // 맵 크기가 바뀌면 타일 재생성
        if (frame.mapWidth != _currentMapWidth || frame.mapHeight != _currentMapHeight)
        {
            _currentMapWidth = frame.mapWidth;
            _currentMapHeight = frame.mapHeight;
            _tilesGenerated = false;
            _cachedTiles = null;
        }

        // 타일 생성 (최초 1회)
        if (!_tilesGenerated && frame.tiles != null && frame.tiles.Length > 0)
        {
            GenerateTiles(frame);
            _tilesGenerated = true;
            _cachedTiles = (int[])frame.tiles.Clone();
        }
        // 타일 변경 감지 및 업데이트 (위험 타일 등이 변할 수 있음)
        else if (_tilesGenerated && frame.tiles != null && TilesChanged(frame.tiles))
        {
            GenerateTiles(frame);
            _cachedTiles = (int[])frame.tiles.Clone();
        }

        // 유닛 수 확인 및 생성
        int needA = frame.teamA != null ? frame.teamA.Length : 0;
        int needB = frame.teamB != null ? frame.teamB.Length : 0;

        EnsureUnits(_teamAUnits, needA, frame.teamA, true);
        EnsureUnits(_teamBUnits, needB, frame.teamB, false);

        // 유닛 위치/상태 업데이트
        UpdateUnits(_teamAUnits, frame.teamA);
        UpdateUnits(_teamBUnits, frame.teamB);

        // 플레이어 사망 상태 확인 및 입력 비활성화
        if (SelectMenu.IsPlayerMode && frame.playerIdx >= 0 && frame.teamA != null)
        {
            bool playerAlive = frame.playerIdx < frame.teamA.Length && frame.teamA[frame.playerIdx].alive;
            if (_playerAlive && !playerAlive)
            {
                // 플레이어가 방금 죽음
            }
            _playerAlive = playerAlive;

            // PlayerInputSender에 상태 전달
            if (_playerInputSender != null)
            {
                _playerInputSender.SetPlayerAlive(_playerAlive);
            }
        }

        _lastStep = frame.step;

        // 카메라 자동 조정 (최초 1회)
        if (autoAdjustCamera && !_cameraAdjusted && _currentMapWidth > 0)
        {
            AdjustCamera();
            _cameraAdjusted = true;
        }

    }

    private void GenerateTiles(FrameData frame)
    {
        // 기존 타일 삭제
        foreach (var obj in _tileObjects)
        {
            if (obj != null) Destroy(obj);
        }
        _tileObjects.Clear();

        if (frame.tiles == null) return;

        // 타일 생성
        for (int y = 0; y < frame.mapHeight; y++)
        {
            for (int x = 0; x < frame.mapWidth; x++)
            {
                int idx = y * frame.mapWidth + x;
                if (idx >= frame.tiles.Length) continue;

                TileType tileType = (TileType)frame.tiles[idx];

                // Empty 타일은 생성하지 않음 (성능 최적화)
                if (tileType == TileType.Empty) continue;

                Vector3 pos = GridToWorld(x, y);
                GameObject tile = CreateAutoTile(tileType, x, y);
                tile.transform.position = pos;
                _tileObjects.Add(tile);
            }
        }

        // 그리드 라인 생성
        if (showGridLines)
        {
            CreateGridLines();
        }

    }

    private void CreateGridLines()
    {
        // 기존 그리드 라인 제거
        foreach (var line in _gridLines)
        {
            if (line != null) Destroy(line);
        }
        _gridLines.Clear();

        // 그리드 라인용 스프라이트 생성 (1x1 흰색)
        Texture2D lineTex = new Texture2D(1, 1);
        lineTex.SetPixel(0, 0, Color.white);
        lineTex.Apply();
        Sprite lineSprite = Sprite.Create(lineTex, new Rect(0, 0, 1, 1), new Vector2(0.5f, 0.5f), 1);

        // 수직선 (X 방향)
        for (int x = 0; x <= _currentMapWidth; x++)
        {
            GameObject line = new GameObject($"GridLine_V_{x}");
            line.transform.SetParent(transform);

            SpriteRenderer sr = line.AddComponent<SpriteRenderer>();
            sr.sprite = lineSprite;
            sr.color = gridLineColor;
            sr.sortingOrder = -1;  // 타일보다 뒤에

            float posX = originOffset.x + x * cellSize - cellSize / 2f;
            float posZ = originOffset.z + (_currentMapHeight * cellSize) / 2f - cellSize / 2f;

            line.transform.position = new Vector3(posX, 0.01f, posZ);
            line.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            line.transform.localScale = new Vector3(gridLineWidth, _currentMapHeight * cellSize, 1f);

            _gridLines.Add(line);
        }

        // 수평선 (Z 방향)
        for (int y = 0; y <= _currentMapHeight; y++)
        {
            GameObject line = new GameObject($"GridLine_H_{y}");
            line.transform.SetParent(transform);

            SpriteRenderer sr = line.AddComponent<SpriteRenderer>();
            sr.sprite = lineSprite;
            sr.color = gridLineColor;
            sr.sortingOrder = -1;  // 타일보다 뒤에

            float posX = originOffset.x + (_currentMapWidth * cellSize) / 2f - cellSize / 2f;
            float posZ = originOffset.z + y * cellSize - cellSize / 2f;

            line.transform.position = new Vector3(posX, 0.01f, posZ);
            line.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            line.transform.localScale = new Vector3(_currentMapWidth * cellSize, gridLineWidth, 1f);

            _gridLines.Add(line);
        }
    }

    private GameObject CreateAutoTile(TileType type, int x, int y)
    {
        GameObject tile = new GameObject($"Tile_{type}_{x}_{y}");
        tile.transform.SetParent(transform);

        // SpriteRenderer 추가
        SpriteRenderer sr = tile.AddComponent<SpriteRenderer>();
        sr.sprite = SpriteGenerator.GetTileSprite(type);
        sr.sortingOrder = 0;  // 타일은 가장 뒤에

        // 타일을 바닥에 눕히기 (XZ 평면)
        tile.transform.rotation = Quaternion.Euler(90, 0, 0);
        tile.transform.localScale = Vector3.one * (cellSize / 64f) * 64f;  // 스프라이트 크기에 맞춤

        return tile;
    }

    private void EnsureUnits(List<GameObject> units, int need, UnitData[] data, bool isTeamA)
    {
        // 부족하면 생성
        while (units.Count < need)
        {
            int idx = units.Count;
            RoleType role = (RoleType)(data != null && idx < data.Length ? data[idx].role : 0);

            GameObject unit = CreateAutoUnit(role, isTeamA, idx);
            unit.SetActive(false);
            units.Add(unit);
        }

        // 남는 유닛은 비활성화
        for (int i = need; i < units.Count; i++)
        {
            if (units[i] != null)
            {
                units[i].SetActive(false);
            }
        }
    }

    private GameObject CreateAutoUnit(RoleType role, bool isTeamA, int idx)
    {
        string teamName = isTeamA ? "A" : "B";
        GameObject unit = new GameObject($"Unit_{teamName}_{idx}_{role}");
        unit.transform.SetParent(transform);

        // SpriteUnit 컴포넌트 추가
        SpriteUnit spriteUnit = unit.AddComponent<SpriteUnit>();
        spriteUnit.role = role;
        spriteUnit.isTeamA = isTeamA;
        spriteUnit.Initialize();  // role, isTeamA 설정 후 초기화

        return unit;
    }

    private void UpdateUnits(List<GameObject> units, UnitData[] data)
    {
        if (data == null) return;

        int n = Mathf.Min(units.Count, data.Length);

        for (int i = 0; i < n; i++)
        {
            var unit = units[i];
            var unitData = data[i];

            if (unit == null) continue;

            // 활성화/비활성화
            unit.SetActive(unitData.alive);

            if (!unitData.alive) continue;

            // 위치 즉시 적용 (Python에서 받은 그리드 좌표 그대로 사용)
            Vector3 targetPos = GridToWorld(unitData.x, unitData.y);
            targetPos.y = 0.5f;
            unit.transform.position = targetPos;

            // HP 바 및 플레이어 표시 업데이트
            var spriteUnit = unit.GetComponent<SpriteUnit>();
            if (spriteUnit != null)
            {
                // 디버그: HP 데이터 확인
                if (i == 0 && _lastStep % 20 == 0)
                {
                   // Debug.Log($"[GameViewer] Unit0 HP: {unitData.hp}/{unitData.maxHp}");
                }
                spriteUnit.SetHp(unitData.hp, unitData.maxHp);
                spriteUnit.SetIsPlayer(unitData.isPlayer);
            }
        }
    }

    private Vector3 GridToWorld(int gx, int gy)
    {
        return originOffset + new Vector3(gx * cellSize, 0f, gy * cellSize);
    }

    private void AdjustCamera()
    {
        Camera cam = Camera.main;
        if (cam == null) return;

        // 맵 중앙 계산
        float centerX = originOffset.x + (_currentMapWidth * cellSize) / 2f - cellSize / 2f;
        float centerZ = originOffset.z + (_currentMapHeight * cellSize) / 2f - cellSize / 2f;

        // 직교 카메라로 전환하여 정확한 뷰 제공
        cam.orthographic = true;

        // 화면 비율에 맞춰 orthographic size 계산
        float mapWidth = _currentMapWidth * cellSize;
        float mapHeight = _currentMapHeight * cellSize;
        float screenRatio = (float)Screen.width / Screen.height;
        float mapRatio = mapWidth / mapHeight;

        // 맵이 화면에 딱 맞도록 orthographic size 설정
        if (screenRatio >= mapRatio)
        {
            // 화면이 더 넓음 - 높이 기준
            cam.orthographicSize = mapHeight / 2f;
        }
        else
        {
            // 화면이 더 좁음 - 너비 기준
            cam.orthographicSize = mapWidth / (2f * screenRatio);
        }

        // 카메라 위치 설정 (위에서 수직으로 내려다봄)
        cam.transform.position = new Vector3(centerX, 50f, centerZ);
        cam.transform.rotation = Quaternion.Euler(90f, 0f, 0f);

    }

    private void OnDisable()
    {
        // 정리
        ClearAll();
    }

    private void OnDestroy()
    {
        if (_loadingBgTexture != null) Destroy(_loadingBgTexture);
        if (_loadingCircleTexture != null) Destroy(_loadingCircleTexture);
    }

    private void ClearAll()
    {
        foreach (var unit in _teamAUnits)
        {
            if (unit != null) Destroy(unit);
        }
        _teamAUnits.Clear();

        foreach (var unit in _teamBUnits)
        {
            if (unit != null) Destroy(unit);
        }
        _teamBUnits.Clear();

        foreach (var tile in _tileObjects)
        {
            if (tile != null) Destroy(tile);
        }
        _tileObjects.Clear();

        foreach (var line in _gridLines)
        {
            if (line != null) Destroy(line);
        }
        _gridLines.Clear();

        _tilesGenerated = false;
    }

    private void OnGUI()
    {
        // 첫 프레임 도착 전에는 로딩 오버레이 표시
        if (_lastStep < 0)
        {
            DrawLoadingOverlay();
            return;
        }

        if (!showDebugInfo) return;

        GUILayout.BeginArea(new Rect(10, 10, 200, 100));
        GUILayout.Label($"Step: {_lastStep}");
        GUILayout.Label($"Map: {_currentMapWidth}x{_currentMapHeight}");
        GUILayout.Label($"Team A: {CountAlive(_teamAUnits)}");
        GUILayout.Label($"Team B: {CountAlive(_teamBUnits)}");
        GUILayout.EndArea();
    }

    private void DrawLoadingOverlay()
    {
        if (_loadingBgTexture == null) return;

        // 90%에서 시작해서 시간에 따라 천천히 증가 (최대 99%)
        float elapsed = Time.time - _loadingStartTime;
        _loadingProgress = Mathf.Min(0.9f + elapsed * 0.03f, 0.99f);
        _loadingDisplayProgress = Mathf.Lerp(_loadingDisplayProgress, _loadingProgress, Time.deltaTime * 5f);

        // 배경
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), _loadingBgTexture);

        float centerX = Screen.width / 2f;
        float centerY = Screen.height / 2f;

        // 원형 프로그레스 그리기
        if (_loadingCircleTexture != null)
        {
            float circleSize = 120f;
            Rect circleRect = new Rect(centerX - circleSize / 2f, centerY - 60f - circleSize / 2f, circleSize, circleSize);

            // 배경 원
            GUI.color = new Color(0.3f, 0.3f, 0.35f, 1f);
            GUI.DrawTexture(circleRect, _loadingCircleTexture);

            // 회전하는 프로그레스 원
            Matrix4x4 matrixBackup = GUI.matrix;
            float rotation = Time.time * 180f;
            GUIUtility.RotateAroundPivot(rotation, new Vector2(centerX, centerY - 60f));

            GUI.color = loadingCircleColor;
            GUI.DrawTexture(circleRect, _loadingCircleTexture);

            GUI.matrix = matrixBackup;
            GUI.color = Color.white;
        }

        // 퍼센트 텍스트
        GUIStyle percentStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 32,
            alignment = TextAnchor.MiddleCenter,
            fontStyle = FontStyle.Bold
        };
        percentStyle.normal.textColor = Color.white;

        string percentText = $"{Mathf.RoundToInt(_loadingDisplayProgress * 100)}%";
        GUI.Label(new Rect(centerX - 100, centerY - 80, 200, 50), percentText, percentStyle);

        // 상태 텍스트
        GUIStyle statusStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 18,
            alignment = TextAnchor.MiddleCenter
        };
        statusStyle.normal.textColor = new Color(0.8f, 0.8f, 0.8f);
        GUI.Label(new Rect(centerX - 200, centerY + 30, 400, 30), "Waiting for game data...", statusStyle);

        // 타이틀
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 28,
            alignment = TextAnchor.MiddleCenter,
            fontStyle = FontStyle.Bold
        };
        titleStyle.normal.textColor = Color.white;
        GUI.Label(new Rect(centerX - 200, centerY - 180, 400, 40), "Loading...", titleStyle);
    }

    private int CountAlive(List<GameObject> units)
    {
        int count = 0;
        foreach (var u in units)
        {
            if (u != null && u.activeSelf) count++;
        }
        return count;
    }

    /// <summary>
    /// 타일 배열이 변경되었는지 확인
    /// </summary>
    private bool TilesChanged(int[] newTiles)
    {
        if (_cachedTiles == null) return true;
        if (_cachedTiles.Length != newTiles.Length) return true;

        for (int i = 0; i < newTiles.Length; i++)
        {
            if (_cachedTiles[i] != newTiles[i]) return true;
        }
        return false;
    }
}