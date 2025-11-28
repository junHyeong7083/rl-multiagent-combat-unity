using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// 캐릭터 선택 씬 - 5개 역할 중 하나를 선택
/// </summary>
public class SelectMenu : MonoBehaviour
{
    [Header("Scene Names")]
    public string gameScene2D = "GameScene";
    public string gameScene3D = "3DGameScene";

    [Header("UI Settings")]
    public bool showGUI = true;

    [Header("Background")]
    public Color backgroundColor = new Color(0.15f, 0.15f, 0.2f, 1f);

    // 선택된 역할 (static으로 씬 간 전달)
    public static RoleType SelectedRole = RoleType.Tank;
    public static bool IsPlayerMode = false;
    public static bool Use3DMode = false;

    // Python/로딩 정보 (TitleMenu에서 설정)
    public static string PythonPath = "";
    public static string PythonScript = "";
    public static string PythonArgs = "";
    public static string LoadingScene = "LodingScene";

    private string _statusMessage = "";
    private Texture2D _bgTexture;

    private void OnGUI()
    {
        if (!showGUI) return;

        // 배경 패널 그리기 (전체 화면)
        DrawBackground();

        float width = 500;
        float height = 550;
        float x = (Screen.width - width) / 2;
        float y = (Screen.height - height) / 2;

        GUILayout.BeginArea(new Rect(x, y, width, height));

        // 스타일
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 32,
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
            fontSize = 20,
            fixedHeight = 50
        };

        GUIStyle descStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 12,
            alignment = TextAnchor.MiddleLeft,
            wordWrap = true
        };

        // 타이틀
        GUILayout.Space(10);
        GUILayout.Label("Select Your Character", titleStyle);
        GUILayout.Space(5);
        GUILayout.Label("A팀에서 직접 조종할 캐릭터를 선택하세요", subtitleStyle);
        GUILayout.Space(20);

        // Tank 버튼
        if (GUILayout.Button("Tank (탱커)", buttonStyle))
        {
            SelectAndStart(RoleType.Tank);
        }
        GUILayout.Label("  HP: 150 | ATK: 10 | DEF: 15 | Range: 1 - 높은 체력, 방어 특화", descStyle);
        GUILayout.Space(8);

        // Dealer 버튼
        if (GUILayout.Button("Dealer (딜러)", buttonStyle))
        {
            SelectAndStart(RoleType.Dealer);
        }
        GUILayout.Label("  HP: 80 | ATK: 25 | DEF: 5 | Range: 1 - 높은 공격력, 낮은 방어", descStyle);
        GUILayout.Space(8);

        // Healer 버튼
        if (GUILayout.Button("Healer (힐러)", buttonStyle))
        {
            SelectAndStart(RoleType.Healer);
        }
        GUILayout.Label("  HP: 70 | ATK: 8 | DEF: 5 | Range: 2 - 힐 스킬 보유, 아군 회복", descStyle);
        GUILayout.Space(8);

        // Ranger 버튼
        if (GUILayout.Button("Ranger (원거리)", buttonStyle))
        {
            SelectAndStart(RoleType.Ranger);
        }
        GUILayout.Label("  HP: 60 | ATK: 20 | DEF: 3 | Range: 4 - 긴 공격 범위, 낮은 체력", descStyle);
        GUILayout.Space(8);

        // Support 버튼
        if (GUILayout.Button("Support (서포터)", buttonStyle))
        {
            SelectAndStart(RoleType.Support);
        }
        GUILayout.Label("  HP: 90 | ATK: 12 | DEF: 8 | Range: 2 - 균형잡힌 스탯, 버프/디버프", descStyle);
        GUILayout.Space(20);

        // 뒤로가기 버튼
        GUIStyle backStyle = new GUIStyle(buttonStyle)
        {
            fontSize = 16,
            fixedHeight = 40
        };
        if (GUILayout.Button("← Back to Title", backStyle))
        {
            SceneManager.LoadScene("TitleScene");
        }

        // 상태 메시지
        if (!string.IsNullOrEmpty(_statusMessage))
        {
            GUILayout.Space(10);
            GUILayout.Label(_statusMessage, subtitleStyle);
        }

        GUILayout.EndArea();
    }

    private void SelectAndStart(RoleType role)
    {
        SelectedRole = role;
        IsPlayerMode = true;
        _statusMessage = $"Selected: {role} - Loading...";

        Debug.Log($"[SelectMenu] Player selected role: {role}");

        // 로딩 씬으로 데이터 전달
        string targetScene = Use3DMode ? gameScene3D : gameScene2D;
        global::LoadingScene.TargetScene = targetScene;
        global::LoadingScene.PythonPath = PythonPath;
        global::LoadingScene.PythonScript = PythonScript;
        global::LoadingScene.PythonArgs = PythonArgs;

        // 로딩 씬으로 이동
        SceneManager.LoadScene(LoadingScene);
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