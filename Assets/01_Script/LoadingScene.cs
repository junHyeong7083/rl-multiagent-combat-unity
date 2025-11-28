using System.Collections;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// 로딩 씬 - Python 시작 및 첫 프레임 대기
/// 원형 프로그레스 표시
/// </summary>
public class LoadingScene : MonoBehaviour
{
    [Header("Background")]
    public Color backgroundColor = new Color(0.15f, 0.15f, 0.2f, 1f);

    [Header("Progress Circle")]
    public Color circleBackgroundColor = new Color(0.3f, 0.3f, 0.35f, 1f);
    public Color circleProgressColor = new Color(0.4f, 0.7f, 1f, 1f);
    public float circleRadius = 60f;
    public float circleThickness = 8f;

    [Header("Timing")]
    public float minLoadingTime = 1f;  // 최소 로딩 시간
    public float pythonTimeout = 30f;  // Python 응답 대기 최대 시간

    // Static으로 씬 간 데이터 전달
    public static string TargetScene = "";
    public static string PythonScript = "";
    public static string PythonArgs = "";
    public static string PythonPath = "";

    private Texture2D _bgTexture;
    private Texture2D _circleTexture;
    private float _progress = 0f;
    private float _displayProgress = 0f;
    private string _statusText = "Initializing...";
    private bool _pythonStarted = false;
    private bool _firstFrameReceived = false;
    private float _elapsedTime = 0f;
    private Process _pythonProcess;
    private UdpReceiver _udpReceiver;

    // GUIStyle 캐싱 (매 프레임 생성 방지)
    private GUIStyle _percentStyle;
    private GUIStyle _statusStyle;
    private GUIStyle _titleStyle;
    private bool _stylesInitialized = false;

    // 회전 각도 캐싱 (OnGUI 떨림 방지)
    private float _cachedRotation = 0f;

    private void Start()
    {
        // 배경 텍스처 생성
        _bgTexture = new Texture2D(1, 1);
        _bgTexture.SetPixel(0, 0, backgroundColor);
        _bgTexture.Apply();

        // 원형 프로그레스 텍스처 생성
        CreateCircleTexture();

        // UdpReceiver 생성 (싱글톤이 없을 때만)
        if (UdpReceiver.Instance == null)
        {
            GameObject udpObj = new GameObject("UdpReceiver");
            _udpReceiver = udpObj.AddComponent<UdpReceiver>();
            // DontDestroyOnLoad는 UdpReceiver.Awake()에서 처리
        }
        else
        {
            _udpReceiver = UdpReceiver.Instance;
        }

        // 로딩 시작
        StartCoroutine(LoadingSequence());
    }

    private void CreateCircleTexture()
    {
        int size = 256;
        _circleTexture = new Texture2D(size, size, TextureFormat.RGBA32, false);

        float center = size / 2f;
        float outerRadius = size / 2f - 4;
        float innerRadius = outerRadius - (circleThickness / circleRadius * size / 2f);

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

        _circleTexture.SetPixels(pixels);
        _circleTexture.Apply();
    }

    private IEnumerator LoadingSequence()
    {
        // Python 시작
        _statusText = "Starting...";
        _progress = 0.3f;

        bool success = StartPython();
        if (!success)
        {
            _statusText = "Failed to start Python!";
            yield return new WaitForSeconds(1f);
            SceneManager.LoadScene("TitleScene");
            yield break;
        }

        _progress = 0.6f;
        _statusText = "Loading...";
        yield return null;

        // PythonManager에 프로세스 등록
        EnsurePythonManager();
        if (PythonManager.Instance != null)
        {
            PythonManager.Instance.SetProcess(_pythonProcess);
        }

        _progress = 0.9f;
        yield return null;

        // 바로 게임 씬으로 전환 (게임 씬에서 로딩 오버레이 표시)
        if (!string.IsNullOrEmpty(TargetScene))
        {
            SceneManager.LoadScene(TargetScene);
        }
        else
        {
            SceneManager.LoadScene("TitleScene");
        }
    }

    private bool StartPython()
    {
        try
        {
            // 기존 Python 프로세스 종료
            KillExistingPythonProcesses();

            // 프로젝트 루트 경로
            string projectRoot = Path.GetDirectoryName(Application.dataPath);
            string scriptPath = Path.Combine(projectRoot, PythonScript);
            scriptPath = Path.GetFullPath(scriptPath);

            if (!File.Exists(scriptPath))
            {
                UnityEngine.Debug.LogError($"[LoadingScene] Python script not found: {scriptPath}");
                return false;
            }

            string workingDir = Path.GetDirectoryName(scriptPath);

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = PythonPath,
                Arguments = $"\"{scriptPath}\" {PythonArgs}",
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = false,
                RedirectStandardOutput = false,
                RedirectStandardError = false
            };

            UnityEngine.Debug.Log($"[LoadingScene] Starting Python: {PythonPath} \"{scriptPath}\" {PythonArgs}");

            _pythonProcess = Process.Start(startInfo);

            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                UnityEngine.Debug.Log($"[LoadingScene] Python started (PID: {_pythonProcess.Id})");
                return true;
            }

            return false;
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError($"[LoadingScene] Error starting Python: {e.Message}");
            return false;
        }
    }

    private void KillExistingPythonProcesses()
    {
        try
        {
            Process[] pythonProcesses = Process.GetProcessesByName("python");
            foreach (var proc in pythonProcesses)
            {
                try
                {
                    proc.Kill();
                    proc.WaitForExit(1000);
                    proc.Dispose();
                }
                catch { }
            }
        }
        catch { }
    }

    private void EnsurePythonManager()
    {
        if (PythonManager.Instance == null)
        {
            GameObject go = new GameObject("PythonManager");
            go.AddComponent<PythonManager>();
        }
    }

    private void StopPython()
    {
        if (_pythonProcess != null && !_pythonProcess.HasExited)
        {
            try
            {
                _pythonProcess.Kill();
                _pythonProcess.Dispose();
            }
            catch { }
        }
        _pythonProcess = null;
    }

    private void Update()
    {
        _elapsedTime += Time.deltaTime;

        // 부드러운 프로그레스 표시
        _displayProgress = Mathf.Lerp(_displayProgress, _progress, Time.deltaTime * 5f);

        // 회전 각도 캐싱 (OnGUI에서 Time.time 사용 방지)
        _cachedRotation = _elapsedTime * 180f;
    }

    private void InitializeStyles()
    {
        if (_stylesInitialized) return;

        _percentStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 32,
            alignment = TextAnchor.MiddleCenter,
            fontStyle = FontStyle.Bold
        };
        _percentStyle.normal.textColor = Color.white;

        _statusStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 18,
            alignment = TextAnchor.MiddleCenter
        };
        _statusStyle.normal.textColor = new Color(0.8f, 0.8f, 0.8f);

        _titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 28,
            alignment = TextAnchor.MiddleCenter,
            fontStyle = FontStyle.Bold
        };
        _titleStyle.normal.textColor = Color.white;

        _stylesInitialized = true;
    }

    private void OnGUI()
    {
        // 스타일 초기화 (1회만)
        InitializeStyles();

        // 배경
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), _bgTexture);

        // 중앙 영역 (정수로 반올림하여 픽셀 정렬)
        float centerX = Mathf.Round(Screen.width / 2f);
        float centerY = Mathf.Round(Screen.height / 2f);

        // 타이틀 (회전 전에 먼저 그리기)
        GUI.Label(new Rect(centerX - 200, centerY - 150, 400, 40), "Loading...", _titleStyle);

        // 퍼센트 텍스트 (회전 전에 먼저 그리기)
        string percentText = $"{Mathf.RoundToInt(_displayProgress * 100)}%";
        GUI.Label(new Rect(centerX - 100, centerY - 50, 200, 50), percentText, _percentStyle);

        // 상태 텍스트 (회전 전에 먼저 그리기)
        GUI.Label(new Rect(centerX - 200, centerY + 60, 400, 30), _statusText, _statusStyle);

        // 원형 프로그레스 그리기 (마지막에 그려서 회전이 다른 UI에 영향 안줌)
        DrawCircularProgress(centerX, centerY - 30f);
    }

    private void DrawCircularProgress(float centerX, float centerY)
    {
        if (_circleTexture == null) return;

        float size = circleRadius * 2f;
        Rect circleRect = new Rect(centerX - circleRadius, centerY - circleRadius, size, size);

        // 배경 원 (회전하면서)
        Matrix4x4 matrixBackup = GUI.matrix;

        // 배경 원
        GUI.color = circleBackgroundColor;
        GUI.DrawTexture(circleRect, _circleTexture);

        // 프로그레스 원 (회전 애니메이션) - 캐싱된 값 사용
        GUIUtility.RotateAroundPivot(_cachedRotation, new Vector2(centerX, centerY));

        // 프로그레스에 따라 그리기 (간단한 방식: 전체 원을 그리고 색상으로 표현)
        GUI.color = circleProgressColor;

        // Arc 형태로 그리기 위해 Material 사용하는 대신
        // 간단히 회전하는 원으로 표현
        GUI.DrawTexture(circleRect, _circleTexture);

        GUI.matrix = matrixBackup;
        GUI.color = Color.white;
    }

    private void OnDestroy()
    {
        if (_bgTexture != null) Destroy(_bgTexture);
        if (_circleTexture != null) Destroy(_circleTexture);
    }

    private void OnApplicationQuit()
    {
        StopPython();
    }
}
