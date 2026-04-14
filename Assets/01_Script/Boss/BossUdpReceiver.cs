using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// Python boss_streamer.py 로부터 BossSnapshot JSON을 UDP로 수신.
    /// JsonUtility는 int[][] 2차원 배열을 바로 파싱 못 해서, 래퍼 후처리 필요.
    /// </summary>
    public class BossUdpReceiver : MonoBehaviour
    {
        public static BossUdpReceiver Instance { get; private set; }

        [Header("UDP")]
        public int port = 5005;

        private UdpClient _udp;
        private Thread _thread;
        private volatile bool _running;
        private readonly ConcurrentQueue<BossSnapshot> _queue = new ConcurrentQueue<BossSnapshot>();

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        private void OnEnable() => StartReceiving();
        private void OnDisable() => StopReceiving();
        private void OnDestroy() { StopReceiving(); if (Instance == this) Instance = null; }

        public bool TryDequeue(out BossSnapshot snap) => _queue.TryDequeue(out snap);
        public int PendingCount => _queue.Count;

        private void StartReceiving()
        {
            if (_running) return;
            try
            {
                _udp = new UdpClient(port);
                _udp.Client.ReceiveBufferSize = 1 << 20;
                _running = true;
                _thread = new Thread(ReceiveLoop) { IsBackground = true };
                _thread.Start();
                Debug.Log($"[BossUdp] listening on {port}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[BossUdp] bind failed: {e.Message}");
            }
        }

        private void StopReceiving()
        {
            _running = false;
            try { _udp?.Close(); } catch { }
            if (_thread != null && _thread.IsAlive) _thread.Join(200);
            _udp = null; _thread = null;
        }

        private void ReceiveLoop()
        {
            var ep = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    var data = _udp.Receive(ref ep);
                    var json = Encoding.UTF8.GetString(data);
                    var snap = ParseSnapshot(json);
                    if (snap != null) _queue.Enqueue(snap);
                }
                catch (SocketException) { if (!_running) break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception e) { Debug.LogWarning($"[BossUdp] parse error: {e.Message}"); }
            }
        }

        /// <summary>
        /// JsonUtility의 한계(2차원 배열 미지원) 우회:
        /// Python에서 "danger_tiles": [[x,y], ...]로 보내므로 문자열 치환 후 수동 파싱.
        /// </summary>
        private BossSnapshot ParseSnapshot(string json)
        {
            // JsonUtility로는 int[][]를 못 읽으므로, 최상위 객체만 먼저 읽고 danger_tiles는 보완 파싱.
            // 우회: "danger_tiles":[[1,2],[3,4]] → 임시 문자열로 치환 → 나중에 수동 복원.
            // 간단화를 위해 MiniJSON 스타일 수동 파서 사용.
            try
            {
                return BossJsonParser.Parse(json);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[BossUdp] json parse fail: {e.Message}");
                return null;
            }
        }
    }
}
