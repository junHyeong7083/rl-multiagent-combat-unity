using UnityEngine;
using UnityEngine.SceneManagement;
using System.Diagnostics;
using System.IO;

public class MainMenu : MonoBehaviour
{
    [Header("Scene Settings")]
    [Tooltip("게임 씬 이름")]
    public string gameSceneName = "GameScene";

    [Header("Python Settings")]
    [Tooltip("Python 실행 파일 경로 (비워두면 python 명령 사용)")]
    public string pythonPath = @"C:\Users\user\miniconda3\envs\rl_game_npc\python.exe";

    [Tooltip("unity_streamer.py 경로 (프로젝트 루트 기준)")]
    public string streamerScript = "RL_Game_NPC/unity_streamer.py";

    [Tooltip("Python 실행 인자")]
    public string pythonArgs = "--mode trained --model models_v3_20m/model_latest.pt --episodes 10 --delay 0.1 --deterministic";

    [Header("UI")]
    public bool showGUI = true;

    private Process _pythonProcess;
    private string _statusMessage = "";
    private bool _isLoading = false;

    private void OnGUI()
    {
        if (!showGUI) return;

        // 화면 중앙에 UI 배치
        float width = 400;
        float height = 300;
        float x = (Screen.width - width) / 2;
        float y = (Screen.height - height) / 2;

        GUILayout.BeginArea(new Rect(x, y, width, height));

        // 스타일 설정
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 32,
            alignment = TextAnchor.MiddleCenter,
            fontStyle = FontStyle.Bold
        };

        GUIStyle buttonStyle = new GUIStyle(GUI.skin.button)
        {
            fontSize = 20,
            fixedHeight = 50
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
        GUILayout.Space(40);

        if (!_isLoading)
        {
            // 시작 버튼
            if (GUILayout.Button("Start Game", buttonStyle))
            {
                StartGame();
            }

            GUILayout.Space(20);

            // Python만 실행 버튼
            if (GUILayout.Button("Start Python Only", buttonStyle))
            {
                StartPythonOnly();
            }

            GUILayout.Space(20);

            // 종료 버튼
            if (GUILayout.Button("Quit", buttonStyle))
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

    private void StartGame()
    {
        _isLoading = true;
        _statusMessage = "Starting Python and loading game...";

        // Python 실행
        bool pythonStarted = StartPython();

        if (pythonStarted)
        {
            // PythonManager에 프로세스 등록
            EnsurePythonManager();
            if (PythonManager.Instance != null)
            {
                PythonManager.Instance.SetProcess(_pythonProcess);
            }

            // 씬 전환
            SceneManager.LoadScene(gameSceneName);
        }
        else
        {
            _isLoading = false;
            _statusMessage = "Failed to start Python. Check console for details.";
        }
    }

    private void EnsurePythonManager()
    {
        if (PythonManager.Instance == null)
        {
            GameObject go = new GameObject("PythonManager");
            go.AddComponent<PythonManager>();
        }
    }

    private void StartPythonOnly()
    {
        bool started = StartPython();
        if (started)
        {
            _statusMessage = "Python started successfully!";
        }
        else
        {
            _statusMessage = "Failed to start Python.";
        }
    }

    private bool StartPython()
    {
        try
        {
            // 기존 Python 프로세스 종료
            KillExistingPythonProcesses();

            // 프로젝트 루트 경로 찾기
            string projectRoot = GetProjectRoot();
            string scriptPath = Path.Combine(projectRoot, streamerScript);

            // 경로 정규화
            scriptPath = Path.GetFullPath(scriptPath);

            if (!File.Exists(scriptPath))
            {
                UnityEngine.Debug.LogError($"[MainMenu] Python script not found: {scriptPath}");
                _statusMessage = $"Script not found: {scriptPath}";
                return false;
            }

            // 작업 디렉토리 (스크립트가 있는 폴더)
            string workingDir = Path.GetDirectoryName(scriptPath);

            // ProcessStartInfo 설정
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = $"\"{scriptPath}\" {pythonArgs}",
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = false,  // 콘솔 창 표시 (디버깅용)
                RedirectStandardOutput = false,
                RedirectStandardError = false
            };

            UnityEngine.Debug.Log($"[MainMenu] Starting Python: {pythonPath} \"{scriptPath}\" {pythonArgs}");
            UnityEngine.Debug.Log($"[MainMenu] Working directory: {workingDir}");

            _pythonProcess = Process.Start(startInfo);

            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                UnityEngine.Debug.Log($"[MainMenu] Python process started (PID: {_pythonProcess.Id})");
                return true;
            }
            else
            {
                UnityEngine.Debug.LogError("[MainMenu] Python process failed to start");
                return false;
            }
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError($"[MainMenu] Error starting Python: {e.Message}");
            _statusMessage = $"Error: {e.Message}";
            return false;
        }
    }

    private string GetProjectRoot()
    {
        // Unity 프로젝트 루트 (Assets 폴더의 부모)
        string assetsPath = Application.dataPath;  // .../Assets
        return Path.GetDirectoryName(assetsPath);   // .../
    }

    private void KillExistingPythonProcesses()
    {
        int killedCount = 0;
        try
        {
            // "python" 이름의 모든 프로세스 찾기
            Process[] pythonProcesses = Process.GetProcessesByName("python");
            foreach (var proc in pythonProcesses)
            {
                try
                {
                    proc.Kill();
                    proc.WaitForExit(1000);  // 최대 1초 대기
                    proc.Dispose();
                    killedCount++;
                }
                catch (System.Exception e)
                {
                    UnityEngine.Debug.LogWarning($"[MainMenu] Failed to kill python process {proc.Id}: {e.Message}");
                }
            }

            if (killedCount > 0)
            {
                UnityEngine.Debug.Log($"[MainMenu] Killed {killedCount} existing Python process(es)");
                // 잠시 대기하여 포트가 해제되도록 함
                System.Threading.Thread.Sleep(500);
            }
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogWarning($"[MainMenu] Error while killing Python processes: {e.Message}");
        }
    }

    private void QuitGame()
    {
        StopPython();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    private void StopPython()
    {
        if (_pythonProcess != null && !_pythonProcess.HasExited)
        {
            try
            {
                _pythonProcess.Kill();
                _pythonProcess.Dispose();
                UnityEngine.Debug.Log("[MainMenu] Python process stopped");
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogWarning($"[MainMenu] Error stopping Python: {e.Message}");
            }
        }
        _pythonProcess = null;
    }

    private void OnApplicationQuit()
    {
        StopPython();
    }

    private void OnDestroy()
    {
        // 씬 전환 시에는 Python을 종료하지 않음
        // StopPython();
    }
}