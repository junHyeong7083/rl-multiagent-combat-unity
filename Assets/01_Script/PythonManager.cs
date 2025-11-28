using UnityEngine;
using UnityEngine.SceneManagement;
using System.Diagnostics;

/// <summary>
/// Python 프로세스를 관리하는 싱글톤
/// 씬 전환 시에도 유지됨
/// </summary>
public class PythonManager : MonoBehaviour
{
    public static PythonManager Instance { get; private set; }

    public Process PythonProcess { get; private set; }
    public bool IsRunning => PythonProcess != null && !PythonProcess.HasExited;

    [Header("Settings")]
    public string mainMenuScene = "MainMenu";
    public KeyCode returnKey = KeyCode.Escape;

    private void Awake()
    {
        // 싱글톤 패턴
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void Update()
    {
        // ESC 키로 메인 메뉴 복귀
        if (Input.GetKeyDown(returnKey))
        {
            ReturnToMainMenu();
        }
    }

    public void SetProcess(Process process)
    {
        PythonProcess = process;
    }

    public void StopPython()
    {
        if (PythonProcess != null && !PythonProcess.HasExited)
        {
            try
            {
                PythonProcess.Kill();
                PythonProcess.Dispose();
                UnityEngine.Debug.Log("[PythonManager] Python process stopped");
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogWarning($"[PythonManager] Error stopping Python: {e.Message}");
            }
        }
        PythonProcess = null;
    }

    public void ReturnToMainMenu()
    {
        StopPython();
        SceneManager.LoadScene(mainMenuScene);
    }

    private void OnApplicationQuit()
    {
        StopPython();
    }

    private void OnGUI()
    {
        // 우측 상단에 상태 표시
        GUILayout.BeginArea(new Rect(Screen.width - 220, 10, 210, 60));

        GUIStyle boxStyle = new GUIStyle(GUI.skin.box);
        GUILayout.BeginVertical(boxStyle);

        string status = IsRunning ? "<color=green>Python: Running</color>" : "<color=red>Python: Stopped</color>";
        GUIStyle statusStyle = new GUIStyle(GUI.skin.label) { richText = true };
        GUILayout.Label(status, statusStyle);
        GUILayout.Label("Press ESC to return to menu");

        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}