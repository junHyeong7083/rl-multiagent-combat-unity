using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Python RL_Game_NPC 환경을 Unity 3D로 시각화
/// 캐릭터: 캡슐, 벽: 큐브, 버프: 바닥 2D 이미지
/// </summary>
public class GameViewer3D : MonoBehaviour
{
    [Header("Network")]
    public UdpReceiver udpReceiver;

    [Header("Grid Settings")]
    [Tooltip("그리드 1칸의 월드 크기")]
    public float cellSize = 1f;
    [Tooltip("그리드 (0,0)의 월드 기준점")]
    public Vector3 originOffset = Vector3.zero;

    [Header("Unit Settings")]
    [Tooltip("캡슐 반지름")]
    public float capsuleRadius = 0.3f;
    [Tooltip("캡슐 높이")]
    public float capsuleHeight = 0.8f;

    [Header("Wall Settings")]
    [Tooltip("벽 높이")]
    public float wallHeight = 1.5f;

    [Header("Camera")]
    [Tooltip("카메라 자동 조정")]
    public bool autoAdjustCamera = true;
    [Tooltip("쿼터뷰 카메라 각도 (X 회전)")]
    public float cameraAngleX = 45f;
    [Tooltip("쿼터뷰 카메라 각도 (Y 회전)")]
    public float cameraAngleY = 45f;
    [Tooltip("카메라 거리 배율")]
    public float cameraDistanceMultiplier = 1.5f;

    [Header("Debug")]
    public bool showDebugInfo = true;

    // 내부 상태
    private bool _cameraAdjusted = false;
    private readonly List<GameObject> _teamAUnits = new List<GameObject>();
    private readonly List<GameObject> _teamBUnits = new List<GameObject>();
    private readonly List<GameObject> _wallObjects = new List<GameObject>();
    private readonly List<GameObject> _floorTiles = new List<GameObject>();

    private int _currentMapWidth;
    private int _currentMapHeight;
    private int _lastStep = -1;
    private bool _mapGenerated = false;

    // 재료 캐시
    private Material _teamAMaterial;
    private Material _teamBMaterial;
    private Material _wallMaterial;
    private Material _floorMaterial;
    private Material _gridLineMaterial;
    private Material _dangerMaterial;
    private Material _buffAtkMaterial;
    private Material _buffDefMaterial;
    private Material _buffHealMaterial;

    // 그리드 라인
    private readonly List<GameObject> _gridLines = new List<GameObject>();

    // 플레이어 모드 지원
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
        InitializeMaterials();

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
                Debug.Log("[GameViewer3D] Using UdpReceiver singleton instance");
            }
            else
            {
                udpReceiver = FindObjectOfType<UdpReceiver>();
                if (udpReceiver == null)
                {
                    Debug.LogWarning("[GameViewer3D] UdpReceiver not found. Creating one...");
                    GameObject go = new GameObject("UdpReceiver");
                    go.AddComponent<UdpReceiver>();
                    udpReceiver = go.GetComponent<UdpReceiver>();
                }
                else
                {
                    Debug.Log("[GameViewer3D] Found existing UdpReceiver");
                }
            }
        }
        Debug.Log($"[GameViewer3D] udpReceiver assigned: {udpReceiver != null}, Instance: {UdpReceiver.Instance != null}");

        // PlayerInputSender 찾기/생성 (플레이어 모드일 때만)
        if (SelectMenu.IsPlayerMode)
        {
            _playerInputSender = FindObjectOfType<PlayerInputSender>();
            if (_playerInputSender == null)
            {
                Debug.Log("[GameViewer3D] Creating PlayerInputSender for player mode");
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

    private void InitializeMaterials()
    {
        // URP/Built-in 호환 Shader 찾기
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        if (shader == null) shader = Shader.Find("Legacy Shaders/Diffuse");

        // 팀 A - 파랑
        _teamAMaterial = CreateMaterial(shader, new Color(0.2f, 0.4f, 1.0f));

        // 팀 B - 빨강
        _teamBMaterial = CreateMaterial(shader, new Color(1.0f, 0.2f, 0.2f));

        // 벽 - 검정
        _wallMaterial = CreateMaterial(shader, new Color(0.15f, 0.15f, 0.18f));

        // 바닥 - 회색
        _floorMaterial = CreateMaterial(shader, new Color(0.5f, 0.5f, 0.45f));

        // 그리드 라인 - 어두운 회색
        Shader unlitShader = Shader.Find("Universal Render Pipeline/Unlit");
        if (unlitShader == null) unlitShader = Shader.Find("Unlit/Color");
        if (unlitShader == null) unlitShader = shader;
        _gridLineMaterial = new Material(unlitShader);
        if (_gridLineMaterial.HasProperty("_BaseColor"))
            _gridLineMaterial.SetColor("_BaseColor", new Color(0.25f, 0.25f, 0.25f));
        if (_gridLineMaterial.HasProperty("_Color"))
            _gridLineMaterial.SetColor("_Color", new Color(0.25f, 0.25f, 0.25f));

        // 위험 타일 - 빨간색
        _dangerMaterial = CreateMaterial(shader, new Color(0.9f, 0.3f, 0.3f));

        // 버프 타일들
        _buffAtkMaterial = CreateMaterial(shader, new Color(1f, 0.6f, 0.2f));
        _buffDefMaterial = CreateMaterial(shader, new Color(0.3f, 0.7f, 1f));
        _buffHealMaterial = CreateMaterial(shader, new Color(0.3f, 1f, 0.4f));
    }

    private Material CreateMaterial(Shader shader, Color color)
    {
        Material mat = new Material(shader);

        // URP용 색상 설정
        if (mat.HasProperty("_BaseColor"))
        {
            mat.SetColor("_BaseColor", color);
        }
        // Standard/Legacy용 색상 설정
        if (mat.HasProperty("_Color"))
        {
            mat.SetColor("_Color", color);
        }

        return mat;
    }

    private void LateUpdate()
    {
        // 매 프레임 싱글톤 인스턴스 확인 (씬 전환 후 참조가 깨질 수 있음)
        if (udpReceiver == null || !udpReceiver.gameObject.activeInHierarchy)
        {
            if (UdpReceiver.Instance != null)
            {
                udpReceiver = UdpReceiver.Instance;
                Debug.Log("[GameViewer3D] LateUpdate: Re-acquired UdpReceiver instance");
            }
            else
            {
                return;
            }
        }

        // 최신 프레임만 처리
        FrameData latestFrame = null;
        int frameCount = 0;
        while (udpReceiver.TryDequeue(out FrameData frame))
        {
            latestFrame = frame;
            frameCount++;
        }

        if (latestFrame != null)
        {
            if (_lastStep < 0)
            {
                Debug.Log($"[GameViewer3D] First frame received! Map: {latestFrame.mapWidth}x{latestFrame.mapHeight}, TeamA: {latestFrame.teamA?.Length}, TeamB: {latestFrame.teamB?.Length}");
            }
            ProcessFrame(latestFrame);
        }
    }

    private void ProcessFrame(FrameData frame)
    {
        if (frame == null) return;

        // 맵 크기가 바뀌면 재생성
        if (frame.mapWidth != _currentMapWidth || frame.mapHeight != _currentMapHeight)
        {
            _currentMapWidth = frame.mapWidth;
            _currentMapHeight = frame.mapHeight;
            _mapGenerated = false;
        }

        // 맵 생성 (최초 1회)
        if (!_mapGenerated && frame.tiles != null && frame.tiles.Length > 0)
        {
            GenerateMap(frame);
            _mapGenerated = true;
        }

        // 유닛 수 확인 및 생성
        int needA = frame.teamA != null ? frame.teamA.Length : 0;
        int needB = frame.teamB != null ? frame.teamB.Length : 0;

        EnsureUnits(_teamAUnits, needA, true);
        EnsureUnits(_teamBUnits, needB, false);

        // 유닛 위치/상태 업데이트
        UpdateUnits(_teamAUnits, frame.teamA, true, frame.playerIdx);
        UpdateUnits(_teamBUnits, frame.teamB, false, -1);

        // 플레이어 사망 상태 확인 및 입력 비활성화
        if (SelectMenu.IsPlayerMode && frame.playerIdx >= 0 && frame.teamA != null)
        {
            bool playerAlive = frame.playerIdx < frame.teamA.Length && frame.teamA[frame.playerIdx].alive;
            if (_playerAlive && !playerAlive)
            {
                Debug.Log("[GameViewer3D] Player died! Input disabled.");
            }
            _playerAlive = playerAlive;

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

        // 게임 종료 처리
        if (frame.done && showDebugInfo)
        {
            string result = frame.winner switch
            {
                "A" => "Team A Wins!",
                "B" => "Team B Wins!",
                "draw" => "Draw!",
                _ => "Game Over"
            };
            Debug.Log($"[GameViewer3D] Step {frame.step}: {result}");
        }
    }

    private void GenerateMap(FrameData frame)
    {
        // 기존 오브젝트 삭제
        ClearMap();

        if (frame.tiles == null) return;

        // 바닥 평면 생성
        CreateFloor();

        // 타일 생성
        for (int y = 0; y < frame.mapHeight; y++)
        {
            for (int x = 0; x < frame.mapWidth; x++)
            {
                int idx = y * frame.mapWidth + x;
                if (idx >= frame.tiles.Length) continue;

                TileType tileType = (TileType)frame.tiles[idx];
                Vector3 pos = GridToWorld(x, y);

                if (tileType == TileType.Wall)
                {
                    // 벽은 3D 큐브
                    CreateWall(pos, x, y);
                }
                else if (tileType != TileType.Empty)
                {
                    // 버프/위험 타일은 바닥에 평면
                    CreateFloorTile(pos, tileType, x, y);
                }
            }
        }

        if (showDebugInfo)
        {
            Debug.Log($"[GameViewer3D] Generated {frame.mapWidth}x{frame.mapHeight} 3D map");
        }
    }

    private void CreateFloor()
    {
        // 전체 바닥 평면
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(transform);

        float width = _currentMapWidth * cellSize;
        float height = _currentMapHeight * cellSize;

        floor.transform.position = new Vector3(
            originOffset.x + width / 2f - cellSize / 2f,
            -0.01f,
            originOffset.z + height / 2f - cellSize / 2f
        );

        // Plane의 기본 크기는 10x10이므로 스케일 조정
        floor.transform.localScale = new Vector3(width / 10f, 1f, height / 10f);

        var renderer = floor.GetComponent<Renderer>();
        renderer.material = _floorMaterial;

        _floorTiles.Add(floor);

        // 그리드 라인 생성
        CreateGridLines();
    }

    private void CreateGridLines()
    {
        float lineThickness = 0.02f;
        float lineHeight = 0.005f;

        // 수직선 (X 방향으로 일정 간격)
        for (int x = 0; x <= _currentMapWidth; x++)
        {
            GameObject line = GameObject.CreatePrimitive(PrimitiveType.Cube);
            line.name = $"GridLine_V_{x}";
            line.transform.SetParent(transform);

            float posX = originOffset.x + x * cellSize - cellSize / 2f;
            float posZ = originOffset.z + (_currentMapHeight * cellSize) / 2f - cellSize / 2f;

            line.transform.position = new Vector3(posX, lineHeight, posZ);
            line.transform.localScale = new Vector3(lineThickness, lineHeight * 2f, _currentMapHeight * cellSize);

            var renderer = line.GetComponent<Renderer>();
            renderer.material = _gridLineMaterial;

            // 콜라이더 제거
            var collider = line.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            _gridLines.Add(line);
        }

        // 수평선 (Z 방향으로 일정 간격)
        for (int y = 0; y <= _currentMapHeight; y++)
        {
            GameObject line = GameObject.CreatePrimitive(PrimitiveType.Cube);
            line.name = $"GridLine_H_{y}";
            line.transform.SetParent(transform);

            float posX = originOffset.x + (_currentMapWidth * cellSize) / 2f - cellSize / 2f;
            float posZ = originOffset.z + y * cellSize - cellSize / 2f;

            line.transform.position = new Vector3(posX, lineHeight, posZ);
            line.transform.localScale = new Vector3(_currentMapWidth * cellSize, lineHeight * 2f, lineThickness);

            var renderer = line.GetComponent<Renderer>();
            renderer.material = _gridLineMaterial;

            // 콜라이더 제거
            var collider = line.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            _gridLines.Add(line);
        }
    }

    private void CreateWall(Vector3 pos, int x, int y)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = $"Wall_{x}_{y}";
        wall.transform.SetParent(transform);
        wall.transform.position = new Vector3(pos.x, wallHeight / 2f, pos.z);
        wall.transform.localScale = new Vector3(cellSize, wallHeight, cellSize);

        var renderer = wall.GetComponent<Renderer>();
        renderer.material = _wallMaterial;

        _wallObjects.Add(wall);
    }

    private void CreateFloorTile(Vector3 pos, TileType type, int x, int y)
    {
        GameObject tile = GameObject.CreatePrimitive(PrimitiveType.Quad);
        tile.name = $"Tile_{type}_{x}_{y}";
        tile.transform.SetParent(transform);
        tile.transform.position = new Vector3(pos.x, 0.01f, pos.z);
        tile.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
        tile.transform.localScale = new Vector3(cellSize * 0.9f, cellSize * 0.9f, 1f);

        var renderer = tile.GetComponent<Renderer>();
        renderer.material = type switch
        {
            TileType.Danger => _dangerMaterial,
            TileType.BuffAtk => _buffAtkMaterial,
            TileType.BuffDef => _buffDefMaterial,
            TileType.BuffHeal => _buffHealMaterial,
            _ => _floorMaterial
        };

        // Quad의 Collider 제거 (불필요)
        var collider = tile.GetComponent<Collider>();
        if (collider != null) Destroy(collider);

        _floorTiles.Add(tile);
    }

    private void ClearMap()
    {
        foreach (var obj in _wallObjects)
        {
            if (obj != null) Destroy(obj);
        }
        _wallObjects.Clear();

        foreach (var obj in _floorTiles)
        {
            if (obj != null) Destroy(obj);
        }
        _floorTiles.Clear();

        foreach (var obj in _gridLines)
        {
            if (obj != null) Destroy(obj);
        }
        _gridLines.Clear();
    }

    private void EnsureUnits(List<GameObject> units, int need, bool isTeamA)
    {
        // 부족하면 생성
        while (units.Count < need)
        {
            int idx = units.Count;
            GameObject unit = CreateCapsuleUnit(isTeamA, idx);
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

    private GameObject CreateCapsuleUnit(bool isTeamA, int idx)
    {
        string teamName = isTeamA ? "A" : "B";

        // 부모 오브젝트
        GameObject unit = new GameObject($"Unit_{teamName}_{idx}");
        unit.transform.SetParent(transform);

        // 캡슐 메시
        GameObject capsule = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        capsule.name = "Body";
        capsule.transform.SetParent(unit.transform);
        capsule.transform.localPosition = new Vector3(0f, capsuleHeight / 2f, 0f);
        capsule.transform.localScale = new Vector3(capsuleRadius * 2f, capsuleHeight / 2f, capsuleRadius * 2f);

        var renderer = capsule.GetComponent<Renderer>();
        renderer.material = isTeamA ? _teamAMaterial : _teamBMaterial;

        // Unit3D 컴포넌트 추가
        Unit3D unit3D = unit.AddComponent<Unit3D>();
        unit3D.isTeamA = isTeamA;
        unit3D.capsuleRenderer = renderer;

        return unit;
    }

    private void UpdateUnits(List<GameObject> units, UnitData[] data, bool isTeamA, int playerIdx)
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

            // 위치 즉시 적용
            Vector3 targetPos = GridToWorld(unitData.x, unitData.y);
            unit.transform.position = targetPos;

            // HP 바 및 플레이어 표시 업데이트
            var unit3D = unit.GetComponent<Unit3D>();
            if (unit3D != null)
            {
                unit3D.SetHp(unitData.hp, unitData.maxHp);
                // 플레이어 표시 (A팀이고 playerIdx와 일치할 때)
                bool isPlayer = isTeamA && (i == playerIdx);
                unit3D.SetIsPlayer(isPlayer);
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
        Vector3 mapCenter = new Vector3(centerX, 0f, centerZ);

        // 맵 크기에 따른 카메라 거리
        float mapSize = Mathf.Max(_currentMapWidth, _currentMapHeight) * cellSize;
        float distance = mapSize * cameraDistanceMultiplier;

        // 쿼터뷰 카메라 설정
        cam.orthographic = false; // 원근 카메라
        cam.fieldOfView = 60f;

        // 카메라 위치 계산 (쿼터뷰)
        Quaternion rotation = Quaternion.Euler(cameraAngleX, cameraAngleY, 0f);
        Vector3 offset = rotation * new Vector3(0f, 0f, -distance);
        cam.transform.position = mapCenter + offset;
        cam.transform.LookAt(mapCenter);

        if (showDebugInfo)
        {
            Debug.Log($"[GameViewer3D] Quarter-view camera set - center ({centerX}, {centerZ}), distance {distance}");
        }
    }

    private void OnDisable()
    {
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

        ClearMap();
        _mapGenerated = false;
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
        GUILayout.Label($"[3D Mode]");
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
}
