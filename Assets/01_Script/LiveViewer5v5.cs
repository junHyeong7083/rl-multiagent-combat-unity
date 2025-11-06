using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class LiveViewer5v5 : MonoBehaviour
{
    public enum TransportMode { TCP, UDP }

    [Header("Network")]
    public TransportMode transport = TransportMode.UDP;
    public string host = "127.0.0.1";
    public int port = 7788;  // UDP 서버 기본 포트(수정한 Python 서버와 맞춤)

    [Header("Prefabs")]
    public GameObject agentAPrefab;
    public GameObject agentBPrefab;
    public GameObject bulletPrefab;   // Bullet.cs 포함 프리팹

    [Header("Grid -> World")]
    public float cellSize = 1.0f;
    public Vector3 origin = Vector3.zero;
    public float agentYOffset = 0.0f;
    public float bulletYOffset = 0.3f;

    [Header("Smoothing")]
    public float followSpeed = 15f;
    public bool snapOnFirstFrame = true;

    [Header("Bullet Visual Tuning")]
    [Tooltip("cellSize가 커져도 체감 속도 유지를 위한 배율")]
    public float bulletSpeedPerCell = 8f;
    public float bulletMinSpeed = 30f;
    public float bulletMaxLife = 1.0f;

    [Header("Debug")]
    public bool showLog = false;

    // ===== 내부 상태 =====
    Thread recvThread;
    volatile bool running;
    ConcurrentQueue<string> lineQueue = new ConcurrentQueue<string>();

    // TCP
    TcpClient tcpClient;

    // UDP
    UdpClient udpClient;
    IPEndPoint udpRemote;

    int gridW = 12, gridH = 8, nAgents = 5;
    bool gotFirstFrame = false;

    readonly List<GameObject> aAgents = new List<GameObject>();
    readonly List<GameObject> bAgents = new List<GameObject>();
    readonly List<Vector3> aTargets = new List<Vector3>();
    readonly List<Vector3> bTargets = new List<Vector3>();

    void Start()
    {
        running = true;
        recvThread = new Thread(NetThread) { IsBackground = true };
        recvThread.Start();
    }

    void OnDestroy()
    {
        running = false;
        try { tcpClient?.Close(); } catch { }
        try { udpClient?.Close(); } catch { }
        try { recvThread?.Join(200); } catch { }
        ClearAgents();
    }

    // ===== 네트워크 수신 스레드 =====
    void NetThread()
    {
        try
        {
            if (transport == TransportMode.TCP)
            {
                tcpClient = new TcpClient();
                tcpClient.Connect(host, port);
                var ns = tcpClient.GetStream();
                var sb = new StringBuilder();
                byte[] buf = new byte[8192];

                while (running)
                {
                    int n = ns.Read(buf, 0, buf.Length);
                    if (n <= 0) break;
                    string s = Encoding.UTF8.GetString(buf, 0, n);
                    foreach (char c in s)
                    {
                        if (c == '\n')
                        {
                            var line = sb.ToString();
                            sb.Length = 0;
                            if (!string.IsNullOrWhiteSpace(line))
                                lineQueue.Enqueue(line);
                        }
                        else sb.Append(c);
                    }
                }
            }
            else
            {
                udpClient = new UdpClient(port);
                udpRemote = new IPEndPoint(IPAddress.Any, 0);

                while (running)
                {
                    var data = udpClient.Receive(ref udpRemote);
                    if (data == null || data.Length == 0) continue;
                    string s = Encoding.UTF8.GetString(data);
                    // Python 서버는 한 패킷=한 프레임(JSON) 형태
                    lineQueue.Enqueue(s);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("[Live] NetThread error: " + e.Message);
        }
    }

    // ===== 메인 스레드 처리 =====
    void Update()
    {
        while (lineQueue.TryDequeue(out string line))
        {
            try
            {
                var jo = JObject.Parse(line);

                // 1) 구(舊) TCP 라인 프로토콜: "type": "meta|reset|frame|done"
                string type = jo.Value<string>("type");

                if (!string.IsNullOrEmpty(type))
                {
                    if (type == "meta")
                    {
                        gridW = jo.Value<int>("width");
                        gridH = jo.Value<int>("height");
                        nAgents = jo.Value<int>("n");
                        gotFirstFrame = false;
                        if (showLog) Debug.Log($"[Live] meta: {gridW}x{gridH}, n={nAgents}");
                        RespawnAgents(nAgents);
                    }
                    else if (type == "reset")
                    {
                        gotFirstFrame = false;
                        if (showLog) Debug.Log("[Live] reset");
                    }
                    else if (type == "frame")
                    {
                        ApplyFrameLikeLegacy(jo);
                        FirstFrameSnapIfNeeded();
                    }
                    else if (type == "done")
                    {
                        if (showLog) Debug.Log("[Live] done");
                    }
                    continue;
                }

                // 2) 신(新) UDP 프레임 프로토콜: {t,width,height,baseA,baseB,A,B,shots,outcome}
                ApplyFrameFromUdpServer(jo);
                FirstFrameSnapIfNeeded();
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[Live] parse error: " + ex.Message);
            }
        }

        // 프레임 간 보간 이동
        for (int i = 0; i < aAgents.Count; i++)
        {
            if (aAgents[i])
                aAgents[i].transform.position =
                    Vector3.Lerp(aAgents[i].transform.position, aTargets[i], Time.deltaTime * followSpeed);
            if (bAgents[i])
                bAgents[i].transform.position =
                    Vector3.Lerp(bAgents[i].transform.position, bTargets[i], Time.deltaTime * followSpeed);
        }
    }

    void FirstFrameSnapIfNeeded()
    {
        if (!gotFirstFrame && snapOnFirstFrame)
        {
            for (int i = 0; i < aAgents.Count; i++)
            {
                if (aAgents[i]) aAgents[i].transform.position = aTargets[i];
                if (bAgents[i]) bAgents[i].transform.position = bTargets[i];
            }
            gotFirstFrame = true;
        }
    }

    // ===== 에이전트 관리 =====
    void ClearAgents()
    {
        foreach (var go in aAgents) if (go) Destroy(go);
        foreach (var go in bAgents) if (go) Destroy(go);
        aAgents.Clear(); bAgents.Clear();
        aTargets.Clear(); bTargets.Clear();
    }

    void RespawnAgents(int n)
    {
        ClearAgents();

        if (!agentAPrefab || !agentBPrefab)
        {
            Debug.LogError("[Live] Prefabs not assigned.");
            return;
        }

        for (int i = 0; i < n; i++)
        {
            var a = Instantiate(agentAPrefab, transform);
            var b = Instantiate(agentBPrefab, transform);
            aAgents.Add(a); bAgents.Add(b);

            var p = origin + Vector3.up * agentYOffset;
            a.transform.position = p;
            b.transform.position = p;
            a.SetActive(true); b.SetActive(true);

            aTargets.Add(p);
            bTargets.Add(p);
        }
    }

    // ===== 구 프로토콜(frame) 적용 =====
    void ApplyFrameLikeLegacy(JObject frame)
    {
        var A = (JArray)frame["A"];
        var B = (JArray)frame["B"];
        if (A != null && A.Count != aAgents.Count) RespawnAgents(A.Count);
        if (B != null && B.Count != bAgents.Count) RespawnAgents(B.Count);

        int aAlive = 0, bAlive = 0;

        for (int i = 0; i < aAgents.Count; i++)
        {
            var a = (JArray)A[i]; // [x,y,hp,fx,fy,cd]
            var b = (JArray)B[i];
            int ax = a[0].ToObject<int>(), ay = a[1].ToObject<int>(), ahp = a[2].ToObject<int>();
            int bx = b[0].ToObject<int>(), by = b[1].ToObject<int>(), bhp = b[2].ToObject<int>();

            aTargets[i] = GridToWorld(ax, ay, agentYOffset);
            bTargets[i] = GridToWorld(bx, by, agentYOffset);

            if (aAgents[i].activeSelf != (ahp > 0)) aAgents[i].SetActive(ahp > 0);
            if (bAgents[i].activeSelf != (bhp > 0)) bAgents[i].SetActive(bhp > 0);

            if (ahp > 0) aAlive++;
            if (bhp > 0) bAlive++;
        }

        SpawnBulletsFromShots(frame["shots"] as JArray);

        if (showLog)
        {
            float rA = frame["rA"]?.ToObject<float>() ?? 0f;
            float rB = frame["rB"]?.ToObject<float>() ?? 0f;
            int t = frame["t"]?.ToObject<int>() ?? 0;
            Debug.Log($"[t {t}] rA={rA:F2} rB={rB:F2}  A_alive={aAlive} B_alive={bAlive}");
        }
    }

    // ===== 신 프로토콜(UDP) 적용 =====
    void ApplyFrameFromUdpServer(JObject jo)
    {
        gridW = jo.Value<int?>("width") ?? gridW;
        gridH = jo.Value<int?>("height") ?? gridH;

        var A = (JArray)jo["A"];
        var B = (JArray)jo["B"];
        if (A == null || B == null) return;

        int n = Mathf.Min(A.Count, B.Count);
        if (n != aAgents.Count) RespawnAgents(n);

        int aAlive = 0, bAlive = 0;

        for (int i = 0; i < n; i++)
        {
            var a = (JArray)A[i]; // [x,y,hp,fx,fy,cd]
            var b = (JArray)B[i];
            int ax = a[0].ToObject<int>(), ay = a[1].ToObject<int>(), ahp = a[2].ToObject<int>();
            int bx = b[0].ToObject<int>(), by = b[1].ToObject<int>(), bhp = b[2].ToObject<int>();

            aTargets[i] = GridToWorld(ax, ay, agentYOffset);
            bTargets[i] = GridToWorld(bx, by, agentYOffset);

            if (aAgents[i].activeSelf != (ahp > 0)) aAgents[i].SetActive(ahp > 0);
            if (bAgents[i].activeSelf != (bhp > 0)) bAgents[i].SetActive(bhp > 0);

            if (ahp > 0) aAlive++;
            if (bhp > 0) bAlive++;
        }

        SpawnBulletsFromShots(jo["shots"] as JArray);

        if (showLog)
        {
            int t = jo["t"]?.ToObject<int>() ?? 0;
            string outcome = jo.Value<string>("outcome");
            Debug.Log($"[UDP t {t}] A_alive={aAlive} B_alive={bAlive} outcome={outcome}");
        }
    }

    // ===== 총알 연출 =====
    void SpawnBulletsFromShots(JArray shots)
    {
        if (shots == null || !bulletPrefab) return;

        foreach (var s in shots)
        {
            var jo = (JObject)s;
            var from = jo["from"] as JArray;
            var to = jo["to"] as JArray;
            if (from == null || to == null) continue;

            int fx = from[0].ToObject<int>(), fy = from[1].ToObject<int>();
            int tx = to[0].ToObject<int>(), ty = to[1].ToObject<int>();

            Vector3 wFrom = GridToWorld(fx, fy, bulletYOffset);
            Vector3 wTo = GridToWorld(tx, ty, bulletYOffset);

            // 총구 위치 약간 앞으로 보정
            Vector3 dir = (wTo - wFrom);
            Vector3 ndir = dir.sqrMagnitude > 1e-10f ? dir.normalized : Vector3.forward;
            wFrom += ndir * (0.3f * cellSize);

            var go = Instantiate(bulletPrefab, wFrom, Quaternion.LookRotation(ndir, Vector3.up), transform);

            var bullet = go.GetComponent<Bullet>();
            if (bullet)
            {
                float spd = Mathf.Max(bulletMinSpeed, bulletSpeedPerCell * cellSize);
                bullet.SetMotion(spd, bulletMaxLife);
                bullet.FireTo(wTo);
            }

            // 팀/히트 컬러 피드백(있는 경우)
            string team = jo.Value<string>("team"); // "A" or "B"
            bool hit = jo.Value<bool?>("hit") ?? false;
            var rend = go.GetComponentInChildren<Renderer>();
            if (rend)
            {
                if (team == "A") rend.material.color = hit ? new Color(0.2f, 1f, 0.2f) : new Color(0.5f, 0.9f, 0.5f);
                else rend.material.color = hit ? new Color(1f, 0.3f, 0.3f) : new Color(0.95f, 0.6f, 0.6f);
            }
        }
    }

    // ===== 보조 함수 =====
    Vector3 GridToWorld(int gx, int gy, float yOffset = 0f)
    {
        // 격자 중심에 정렬(+0.5)
        return origin + new Vector3((gx + 0.5f) * cellSize, yOffset, (gy + 0.5f) * cellSize);
    }
}
