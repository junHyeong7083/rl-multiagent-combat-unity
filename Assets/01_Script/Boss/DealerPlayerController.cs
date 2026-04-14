using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// 플레이어(딜러 고정) 입력 → TCP로 Python 전송.
    ///
    /// 키 매핑:
    ///   WASD  = 이동 (격자 1칸씩)
    ///   Space = 기본 공격
    ///   Q     = 스킬 공격
    ///
    /// 턴 기반이라 매 프레임 보내는 게 아니라,
    /// 턴 간격(Python측 TURN_INTERVAL)에 맞춰 "최근 입력"을 제출.
    /// </summary>
    public class DealerPlayerController : MonoBehaviour
    {
        [Header("TCP")]
        public string host = "127.0.0.1";
        public int port = 5006;
        public bool autoConnect = true;

        [Header("Input")]
        [Tooltip("입력 샘플링 간격(초). Python 턴 간격보다 짧아야 함.")]
        public float sampleInterval = 0.1f;

        private TcpClient _tcp;
        private NetworkStream _stream;
        private float _lastSample;
        private int _lastAction = (int)BossActionId.Stay;

        private void Start()
        {
            if (autoConnect) Connect();
        }

        private void OnDestroy()
        {
            Disconnect();
        }

        public void Connect()
        {
            try
            {
                _tcp = new TcpClient();
                _tcp.Connect(host, port);
                _stream = _tcp.GetStream();
                Debug.Log($"[Dealer] TCP connected {host}:{port}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Dealer] TCP connect failed: {e.Message}");
            }
        }

        public void Disconnect()
        {
            try { _stream?.Close(); _tcp?.Close(); } catch { }
            _stream = null; _tcp = null;
        }

        private void Update()
        {
            int action = ReadInput();

            // 의미 있는 입력(STAY 아님) 또는 일정 시간마다 전송
            if (action != (int)BossActionId.Stay)
            {
                _lastAction = action;
                SendAction(action);
                _lastSample = Time.time;
            }
            else if (Time.time - _lastSample > sampleInterval)
            {
                SendAction(_lastAction);
                _lastSample = Time.time;
                _lastAction = (int)BossActionId.Stay;
            }
        }

        private int ReadInput()
        {
            if (Input.GetKey(KeyCode.W)) return (int)BossActionId.MoveUp;
            if (Input.GetKey(KeyCode.S)) return (int)BossActionId.MoveDown;
            if (Input.GetKey(KeyCode.A)) return (int)BossActionId.MoveLeft;
            if (Input.GetKey(KeyCode.D)) return (int)BossActionId.MoveRight;
            if (Input.GetKeyDown(KeyCode.Space)) return (int)BossActionId.AttackBasic;
            if (Input.GetKeyDown(KeyCode.Q)) return (int)BossActionId.AttackSkill;
            return (int)BossActionId.Stay;
        }

        private void SendAction(int action)
        {
            if (_stream == null) return;
            try
            {
                var msg = JsonUtility.ToJson(new PlayerInputMessage(action)) + "\n";
                var bytes = Encoding.UTF8.GetBytes(msg);
                _stream.Write(bytes, 0, bytes.Length);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Dealer] send failed: {e.Message}");
                Disconnect();
            }
        }
    }
}
