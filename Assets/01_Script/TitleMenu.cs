using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// 타이틀 화면 - 2D/3D 모드 선택
/// </summary>
public class TitleMenu : MonoBehaviour
{
    [Header("Scene Names")]
    [Tooltip("2D 게임 씬 이름")]
    public string scene2D = "GameScene";
    [Tooltip("3D 게임 씬 이름")]
    public string scene3D = "3DGameScene";
    [Tooltip("캐릭터 선택 씬")]
    public string selectScene = "SelectScene";
    [Tooltip("로딩 씬")]
    public string loadingScene = "LodingScene";

    [Header("Python Settings")]
    [Tooltip("Python 실행 파일 경로")]
    public string pythonPath = @"C:\Users\user\miniconda3\envs\rl_game_npc\python.exe";
    [Tooltip("unity_streamer.py 경로 (프로젝트 루트 기준)")]
    public string streamerScript = "RL_Game_NPC/unity_streamer.py";
    [Tooltip("Python 실행 인자 (AI 관전 모드)")]
    public string pythonArgs = "--mode trained --model models_v3_20m/model_latest.pt --episodes 1000 --delay 0.1";
    [Tooltip("플레이어 모드 스크립트")]
    public string playerModeScript = "RL_Game_NPC/player_mode_streamer.py";
    [Tooltip("플레이어 모드 Python 인자")]
    public string playerModeArgs = "--model models/model_latest.pt --episodes 10 --delay 0.15";

    [Header("UI Settings")]
    public bool showGUI = true;

    [Header("Background")]
    public Color backgroundColor = new Color(0.15f, 0.15f, 0.2f, 1f);

    private string _statusMessage = "";
    private bool _isLoading = false;
    private Texture2D _bgTexture;

    private void OnGUI()
    {
        if (!showGUI) return;

        // 배경 패널 그리기 (전체 화면)
        DrawBackground();

        // 화면 중앙에 UI 배치
        float width = 450;
        float height = 620;
        float x = (Screen.width - width) / 2;
        float y = (Screen.height - height) / 2;

        GUILayout.BeginArea(new Rect(x, y, width, height));

        // 스타일 설정
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 36,
            alignment = TextAnchor.MiddleCenter,
            fontStyle = FontStyle.Bold
        };

        GUIStyle subtitleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 18,
            alignment = TextAnchor.MiddleCenter
        };

        GUIStyle buttonStyle = new GUIStyle(GUI.skin.button)
        {
            fontSize = 22,
            fixedHeight = 55
        };

        GUIStyle statusStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 14,
            alignment = TextAnchor.MiddleCenter,
            wordWrap = true
        };

        // 타이틀
        GUILayout.Space(20);
        GUILayout.Label("RL Multi-Agent Battle", titleStyle);
        GUILayout.Space(10);
        GUILayout.Label("Select Visualization Mode", subtitleStyle);
        GUILayout.Space(40);

        if (!_isLoading)
        {
            // === AI 관전 모드 ===
            GUILayout.Label("[ AI Watch Mode ]", subtitleStyle);
            GUILayout.Space(5);

            // 2D 모드 버튼
            if (GUILayout.Button("2D Mode (Sprite)", buttonStyle))
            {
                SelectMenu.IsPlayerMode = false;
                StartGame(scene2D);
            }

            GUILayout.Space(10);

            // 3D 모드 버튼
            if (GUILayout.Button("3D Mode (Capsule)", buttonStyle))
            {
                SelectMenu.IsPlayerMode = false;
                StartGame(scene3D);
            }

            GUILayout.Space(25);

            // === 플레이어 모드 ===
            GUILayout.Label("[ Player Mode ]", subtitleStyle);
            GUILayout.Space(5);

            // 2D 플레이어 모드
            if (GUILayout.Button("Play 2D (Select Character)", buttonStyle))
            {
                StartPlayerMode(false);
            }

            GUILayout.Space(10);

            // 3D 플레이어 모드
            if (GUILayout.Button("Play 3D (Select Character)", buttonStyle))
            {
                StartPlayerMode(true);
            }

            GUILayout.Space(25);

            // 종료 버튼
            GUIStyle quitStyle = new GUIStyle(buttonStyle)
            {
                fontSize = 18,
                fixedHeight = 40
            };
            if (GUILayout.Button("Quit", quitStyle))
            {
                QuitGame();
            }
        }
        else
        {
            GUILayout.Label("Loading...", statusStyle);
        }

        // 상태 메시지
        if (!string.IsNullOrEmpty(_statusMessage))
        {
            GUILayout.Space(20);
            GUILayout.Label(_statusMessage, statusStyle);
        }

        GUILayout.EndArea();
    }

    private void StartGame(string sceneName)
    {
        _isLoading = true;
        _statusMessage = "Loading...";

        // 로딩 씬으로 데이터 전달
        LoadingScene.TargetScene = sceneName;
        LoadingScene.PythonScript = streamerScript;
        LoadingScene.PythonArgs = pythonArgs;
        LoadingScene.PythonPath = pythonPath;

        // 로딩 씬으로 이동
        SceneManager.LoadScene(loadingScene);
    }

    private void StartPlayerMode(bool use3D)
    {
        SelectMenu.Use3DMode = use3D;
        SelectMenu.IsPlayerMode = true;

        // 플레이어 모드용 Python/로딩 정보 저장 (SelectMenu에서 사용)
        SelectMenu.PythonPath = pythonPath;
        SelectMenu.PythonScript = playerModeScript;
        SelectMenu.PythonArgs = playerModeArgs;
        SelectMenu.LoadingScene = loadingScene;

        // 캐릭터 선택 씬으로 바로 이동 (로딩 없이)
        SceneManager.LoadScene(selectScene);
    }

    private void QuitGame()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    private void DrawBackground()
    {
        // 배경 텍스처 생성 (1회)
        if (_bgTexture == null)
        {
            _bgTexture = new Texture2D(1, 1);
            _bgTexture.SetPixel(0, 0, backgroundColor);
            _bgTexture.Apply();
        }

        // 전체 화면 배경
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), _bgTexture);
    }

    private void OnDestroy()
    {
        if (_bgTexture != null)
        {
            Destroy(_bgTexture);
        }
    }
}
