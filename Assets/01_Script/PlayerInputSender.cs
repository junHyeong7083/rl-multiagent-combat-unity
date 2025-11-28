using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// 플레이어 입력을 Python으로 TCP 전송
/// </summary>
public class PlayerInputSender : MonoBehaviour
{
    [Header("TCP Settings")]
    public string host = "127.0.0.1";
    public int port = 5006;  // Python이 listen할 포트

    [Header("Input Settings")]
    public KeyCode moveUp = KeyCode.W;
    public KeyCode moveDown = KeyCode.S;
    public KeyCode moveLeft = KeyCode.A;
    public KeyCode moveRight = KeyCode.D;
    public KeyCode attackNearest = KeyCode.Space;
    public KeyCode attackLowest = KeyCode.Q;
    public KeyCode skillAoe = KeyCode.E;
    public KeyCode skillHeal = KeyCode.R;

    private TcpClient _client;
    private NetworkStream _stream;
    private bool _connected = false;
    private int _lastSentAction = -1;
    private float _reconnectTimer = 0f;

    // 액션 값 (Python ActionType과 동일)
    public const int ACTION_STAY = 0;
    public const int ACTION_MOVE_UP = 1;
    public const int ACTION_MOVE_DOWN = 2;
    public const int ACTION_MOVE_LEFT = 3;
    public const int ACTION_MOVE_RIGHT = 4;
    public const int ACTION_ATTACK_NEAREST = 5;
    public const int ACTION_ATTACK_LOWEST = 6;
    public const int ACTION_SKILL_AOE = 7;
    public const int ACTION_SKILL_HEAL = 8;

    // 현재 입력된 액션 (외부에서 읽기 가능)
    public int CurrentAction { get; private set; } = ACTION_STAY;

    // 연결 상태
    public bool IsConnected => _connected;

    // 플레이어 생존 상태
    private bool _playerAlive = true;
    public bool IsPlayerAlive => _playerAlive;

    /// <summary>
    /// 플레이어 생존 상태 설정 (GameViewer에서 호출)
    /// </summary>
    public void SetPlayerAlive(bool alive)
    {
        _playerAlive = alive;
    }

    private void Start()
    {
        // 플레이어 모드가 아니면 비활성화
        if (!SelectMenu.IsPlayerMode)
        {
            enabled = false;
            return;
        }

        TryConnect();
    }

    private void Update()
    {
        if (!SelectMenu.IsPlayerMode) return;

        // 재연결 시도
        if (!_connected)
        {
            _reconnectTimer += Time.deltaTime;
            if (_reconnectTimer >= 1f)
            {
                _reconnectTimer = 0f;
                TryConnect();
            }
            return;
        }

        // 플레이어가 죽었으면 입력 무시
        if (!_playerAlive)
        {
            CurrentAction = ACTION_STAY;
            return;
        }

        // 키 입력 감지
        int action = GetInputAction();
        CurrentAction = action;

        // 액션이 변경되었을 때만 전송 (또는 매 프레임 전송 원하면 조건 제거)
        if (action != _lastSentAction)
        {
            SendAction(action);
            _lastSentAction = action;
        }
    }

    private int GetInputAction()
    {
        // 공격/스킬 우선
        if (Input.GetKey(attackNearest)) return ACTION_ATTACK_NEAREST;
        if (Input.GetKey(attackLowest)) return ACTION_ATTACK_LOWEST;
        if (Input.GetKey(skillAoe)) return ACTION_SKILL_AOE;
        if (Input.GetKey(skillHeal)) return ACTION_SKILL_HEAL;

        // 3D 모드: 쿼터뷰 시점에 맞게 키 매핑 변경 (반전 적용)
        if (SelectMenu.Use3DMode)
        {
            // W -> RIGHT, S -> LEFT, A -> DOWN, D -> UP (반전)
            if (Input.GetKey(moveUp)) return ACTION_MOVE_RIGHT;    // W -> 오른쪽
            if (Input.GetKey(moveDown)) return ACTION_MOVE_LEFT;   // S -> 왼쪽
            if (Input.GetKey(moveLeft)) return ACTION_MOVE_DOWN;   // A -> 위 (Python -Y)
            if (Input.GetKey(moveRight)) return ACTION_MOVE_UP;    // D -> 아래 (Python +Y)
        }
        else
        {
            // 2D 탑다운: 기존 매핑
            if (Input.GetKey(moveUp)) return ACTION_MOVE_DOWN;    // W -> Python에서 -Y (화면 위)
            if (Input.GetKey(moveDown)) return ACTION_MOVE_UP;    // S -> Python에서 +Y (화면 아래)
            if (Input.GetKey(moveLeft)) return ACTION_MOVE_LEFT;
            if (Input.GetKey(moveRight)) return ACTION_MOVE_RIGHT;
        }

        return ACTION_STAY;
    }

    private void TryConnect()
    {
        try
        {
            _client = new TcpClient();
            _client.Connect(host, port);
            _stream = _client.GetStream();
            _connected = true;

            // 초기 메시지: 선택된 역할 전송
            SendRoleSelection();

            Debug.Log($"[PlayerInputSender] Connected to {host}:{port}");
        }
        catch (Exception e)
        {
            _connected = false;
            Debug.LogWarning($"[PlayerInputSender] Connection failed: {e.Message}");
        }
    }

    private void SendRoleSelection()
    {
        if (!_connected) return;

        try
        {
            // JSON 형식으로 역할 전송
            string json = $"{{\"type\":\"role\",\"role\":{(int)SelectMenu.SelectedRole}}}\n";
            byte[] data = Encoding.UTF8.GetBytes(json);
            _stream.Write(data, 0, data.Length);
            Debug.Log($"[PlayerInputSender] Sent role selection: {SelectMenu.SelectedRole}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[PlayerInputSender] Failed to send role: {e.Message}");
            Disconnect();
        }
    }

    public void SendAction(int action)
    {
        if (!_connected) return;

        try
        {
            // JSON 형식으로 액션 전송
            string json = $"{{\"type\":\"action\",\"action\":{action}}}\n";
            byte[] data = Encoding.UTF8.GetBytes(json);
            _stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"[PlayerInputSender] Failed to send action: {e.Message}");
            Disconnect();
        }
    }

    private void Disconnect()
    {
        _connected = false;
        try
        {
            _stream?.Close();
            _client?.Close();
        }
        catch { }
    }

    private void OnDisable()
    {
        Disconnect();
    }

    private void OnApplicationQuit()
    {
        Disconnect();
    }

    // UI 표시용
    private void OnGUI()
    {
        if (!SelectMenu.IsPlayerMode) return;

        // 화면 하단에 조작 안내
        float width = 400;
        float height = 120;
        float x = (Screen.width - width) / 2;
        float y = Screen.height - height - 10;

        GUILayout.BeginArea(new Rect(x, y, width, height));

        GUIStyle boxStyle = new GUIStyle(GUI.skin.box)
        {
            fontSize = 14
        };

        GUIStyle labelStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 12,
            alignment = TextAnchor.MiddleCenter
        };

        GUI.Box(new Rect(0, 0, width, height), "");

        GUILayout.Space(5);
        GUILayout.Label($"Playing as: {SelectMenu.SelectedRole}", labelStyle);
        GUILayout.Label($"Connection: {(_connected ? "Connected" : "Disconnected")}", labelStyle);

        if (!_playerAlive)
        {
            // 플레이어 사망 표시
            GUIStyle deadStyle = new GUIStyle(labelStyle)
            {
                fontSize = 16,
                fontStyle = FontStyle.Bold
            };
            deadStyle.normal.textColor = Color.red;
            GUILayout.Label("YOU DIED - Spectating", deadStyle);
        }
        else
        {
            GUILayout.Space(5);
            GUILayout.Label("WASD: Move | Space: Attack | Q: Attack Lowest HP", labelStyle);
            GUILayout.Label("E: AOE Skill | R: Heal Skill", labelStyle);
            GUILayout.Label($"Current Action: {GetActionName(CurrentAction)}", labelStyle);
        }

        GUILayout.EndArea();
    }

    private string GetActionName(int action)
    {
        return action switch
        {
            ACTION_STAY => "Stay",
            ACTION_MOVE_UP => "Move Up",
            ACTION_MOVE_DOWN => "Move Down",
            ACTION_MOVE_LEFT => "Move Left",
            ACTION_MOVE_RIGHT => "Move Right",
            ACTION_ATTACK_NEAREST => "Attack Nearest",
            ACTION_ATTACK_LOWEST => "Attack Lowest HP",
            ACTION_SKILL_AOE => "AOE Skill",
            ACTION_SKILL_HEAL => "Heal Skill",
            _ => "Unknown"
        };
    }
}