using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;
using UnityEngine;

/// <summary>
/// UDP로 Python RL_Game_NPC 환경의 프레임 데이터를 수신
/// </summary>
public class UdpReceiver : MonoBehaviour
{
    // 싱글톤 인스턴스
    public static UdpReceiver Instance { get; private set; }

    [Header("UDP Settings")]
    public int port = 5005;

    private UdpClient _udp;
    private IPEndPoint _endPoint;
    private Thread _receiveThread;
    private volatile bool _running;

    private readonly ConcurrentQueue<FrameData> _frameQueue = new ConcurrentQueue<FrameData>();

    private void Awake()
    {
        // 기존 인스턴스가 있으면 자신을 삭제
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            Instance = null;
        }
    }

    /// <summary>
    /// 수신된 프레임을 큐에서 꺼냄
    /// </summary>
    public bool TryDequeue(out FrameData frame)
    {
        return _frameQueue.TryDequeue(out frame);
    }

    /// <summary>
    /// 큐에 프레임이 있는지 확인
    /// </summary>
    public bool HasFrame => !_frameQueue.IsEmpty;

    /// <summary>
    /// 데이터가 있는지 확인 (HasFrame과 동일)
    /// </summary>
    public bool HasData() => !_frameQueue.IsEmpty;

    private void OnEnable()
    {
        StartReceiving();
    }

    private void OnDisable()
    {
        StopReceiving();
    }

    private void StartReceiving()
    {
        try
        {
            _endPoint = new IPEndPoint(IPAddress.Any, port);
            _udp = new UdpClient(_endPoint);
            _running = true;

            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true
            };
            _receiveThread.Start();

        }
        catch (Exception e)
        {
            Debug.LogError($"[UdpReceiver] Failed to start: {e.Message}");
        }
    }

    private void StopReceiving()
    {
        _running = false;

        try
        {
            _udp?.Close();
        }
        catch { }

        try
        {
            _receiveThread?.Join(100);
        }
        catch { }
    }

    private void ReceiveLoop()
    {
        while (_running)
        {
            try
            {
                byte[] data = _udp.Receive(ref _endPoint);
                string json = Encoding.UTF8.GetString(data);

                FrameData frame = JsonUtility.FromJson<FrameData>(json);
                if (frame != null)
                {
                    _frameQueue.Enqueue(frame);
                }
            }
            catch (SocketException)
            {
                if (!_running) break;
            }
            catch (Exception)
            {
                // 에러 무시
            }
        }
    }
}
