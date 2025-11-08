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
    public int port = 7788;

    [Header("Prefabs")]
    public GameObject agentAPrefab;
    public GameObject agentBPrefab;
    public GameObject bulletPrefab;
    [Tooltip("A팀 넥서스(기지) 프리팹")]
    public GameObject aNexusPrefab;
    [Tooltip("B팀 넥서스(기지) 프리팹")]
    public GameObject bNexusPrefab;
    [Tooltip("장애물 프리팹 (큐브 등)")]
    public GameObject obstaclePrefab;

    [Header("Parents (optional)")]
    public Transform unitRoot;     // 없으면 씬 루트
    public Transform bulletRoot;   // 없으면 씬 루트
    public Transform nexusRoot;    // 없으면 씬 루트
    public Transform obstacleRoot; // 없으면 씬 루트

    [Header("Grid -> World")]
    public float cellSize = 1.0f;        // 서버 map.cell이 오면 그 값으로 덮어씀
    public Vector3 origin = Vector3.zero;
    public float agentYOffset = 0.0f;
    public float bulletYOffset = 0.3f;
    public float nexusYOffset = 0.02f;
    public float obstacleYOffset = 0.0f;

    [Header("Smoothing")]
    public float followSpeed = 15f;
    public bool snapOnFirstFrame = true;

    [Header("Bullet Visual Tuning")]
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

    // Nexus
    GameObject aNexusObj, bNexusObj;

    // Obstacles
    readonly List<GameObject> obstaclePool = new List<GameObject>();
    int cachedObsVer = -1;      // 서버에서 보내는 obs_ver과 동기화
    float serverCellSize = -1f; // 서버 map.cell (없으면 -1 → cellSize 사용)

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
        ClearScene();
        ClearObstacles();
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
                    lineQueue.Enqueue(s); // 한 패킷 = 한 프레임(JSON)
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

                // 1) 구(TCP) 프로토콜: {type:"meta|reset|frame|done", ...}
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
                        EnsureNexus(null, null);
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

                // 2) 신(UDP) 프로토콜
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

    // ===== 에이전트/넥서스 관리 =====
    void ClearScene()
    {
        foreach (var go in aAgents) if (go) Destroy(go);
        foreach (var go in bAgents) if (go) Destroy(go);
        aAgents.Clear(); bAgents.Clear();
        aTargets.Clear(); bTargets.Clear();

        if (aNexusObj) Destroy(aNexusObj);
        if (bNexusObj) Destroy(bNexusObj);
        aNexusObj = null; bNexusObj = null;
    }

    void RespawnAgents(int n)
    {
        foreach (var go in aAgents) if (go) Destroy(go);
        foreach (var go in bAgents) if (go) Destroy(go);
        aAgents.Clear(); bAgents.Clear();
        aTargets.Clear(); bTargets.Clear();

        if (!agentAPrefab || !agentBPrefab)
        {
            Debug.LogError("[Live] Agent Prefabs not assigned.");
            return;
        }

        Transform parent = unitRoot != null ? unitRoot : null;

        for (int i = 0; i < n; i++)
        {
            var a = Instantiate(agentAPrefab, parent);
            var b = Instantiate(agentBPrefab, parent);
            aAgents.Add(a); bAgents.Add(b);

            var p = origin + Vector3.up * agentYOffset;
            a.transform.position = p;
            b.transform.position = p;

            aTargets.Add(p);
            bTargets.Add(p);

            a.SetActive(true); b.SetActive(true);
        }
    }

    void EnsureNexus((int x, int y)? baseA, (int x, int y)? baseB)
    {
        Transform parent = nexusRoot != null ? nexusRoot : null;

        if (baseA.HasValue)
        {
            Vector3 wA = GridToWorld(baseA.Value.x, baseA.Value.y, nexusYOffset);
            if (!aNexusObj && aNexusPrefab)
                aNexusObj = Instantiate(aNexusPrefab, wA, Quaternion.identity, parent);
            if (aNexusObj) aNexusObj.transform.position = wA;
        }
        if (baseB.HasValue)
        {
            Vector3 wB = GridToWorld(baseB.Value.x, baseB.Value.y, nexusYOffset);
            if (!bNexusObj && bNexusPrefab)
                bNexusObj = Instantiate(bNexusPrefab, wB, Quaternion.identity, parent);
            if (bNexusObj) bNexusObj.transform.position = wB;
        }
    }

    (int x, int y)? TryReadBaseXY(JToken token)
    {
        if (token == null) return null;
        if (token is JObject o)
        {
            if (o["x"] != null && o["y"] != null)
                return (o.Value<int>("x"), o.Value<int>("y"));
        }
        if (token is JArray arr && arr.Count >= 2)
            return (arr[0].ToObject<int>(), arr[1].ToObject<int>());
        return null;
    }

    // ===== 구 프로토콜(frame) 적용 =====
    void ApplyFrameLikeLegacy(JObject frame)
    {
        var A = (JArray)frame["A"];
        var B = (JArray)frame["B"];
        if (A != null && A.Count != aAgents.Count) RespawnAgents(A.Count);
        if (B != null && B.Count != bAgents.Count) RespawnAgents(B.Count);

        var baseA = TryReadBaseXY(frame["base_A"]) ?? TryReadBaseXY(frame["baseA"]) ?? TryReadBaseXY(frame["nexusA"]);
        var baseB = TryReadBaseXY(frame["base_B"]) ?? TryReadBaseXY(frame["baseB"]) ?? TryReadBaseXY(frame["nexusB"]);
        EnsureNexus(baseA, baseB);

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

    // ===== 신(UDP) 프로토콜 적용 =====
    void ApplyFrameFromUdpServer(JObject jo)
    {
        // 맵/셀크기
        gridW = jo.Value<int?>("width") ?? gridW;
        gridH = jo.Value<int?>("height") ?? gridH;
        var map = jo["map"] as JObject;
        if (map != null)
        {
            // 서버가 cell을 주면 사용, 아니면 기존 cellSize 유지
            serverCellSize = map.Value<float?>("cell") ?? serverCellSize;
            if (serverCellSize > 0f) cellSize = serverCellSize;
        }

        // 넥서스
        var baseA = TryReadBaseXY(jo["baseA"]) ?? TryReadBaseXY(jo["base_A"]) ?? TryReadBaseXY(jo["nexusA"]);
        var baseB = TryReadBaseXY(jo["baseB"]) ?? TryReadBaseXY(jo["base_B"]) ?? TryReadBaseXY(jo["nexusB"]);
        EnsureNexus(baseA, baseB);

        // 장애물: obs_ver 바뀌면 갱신
        int obsVer = jo.Value<int?>("obs_ver") ?? cachedObsVer;
        var obstacles = jo["obstacles"] as JArray;
        if (obstacles != null && obsVer != cachedObsVer)
        {
            UpdateObstacles(obstacles);
            cachedObsVer = obsVer;
        }

        // 유닛
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

        // 총알
        SpawnBulletsFromShots(jo["shots"] as JArray);

        if (showLog)
        {
            int t = jo["t"]?.ToObject<int>() ?? 0;
            string outcome = jo.Value<string>("outcome");
            Debug.Log($"[UDP t {t}] A_alive={aAlive} B_alive={bAlive} outcome={outcome} obs_ver={cachedObsVer}");
        }
    }

    // ===== 장애물 생성/갱신 =====
    void UpdateObstacles(JArray arr)
    {
        ClearObstacles();

        if (!obstaclePrefab)
        {
            Debug.LogWarning("[Live] obstaclePrefab not assigned — 장애물 좌표는 받았지만 프리팹이 없어 시각화하지 않습니다.");
            return;
        }

        Transform parent = obstacleRoot != null ? obstacleRoot : null;

        for (int i = 0; i < arr.Count; i++)
        {
            var xy = arr[i] as JArray;
            if (xy == null || xy.Count < 2) continue;

            int x = xy[0].ToObject<int>();
            int y = xy[1].ToObject<int>();

            Vector3 pos = GridToWorld(x, y, obstacleYOffset);
            var go = Instantiate(obstaclePrefab, pos, Quaternion.identity, parent);
            obstaclePool.Add(go);
        }
    }

    void ClearObstacles()
    {
        for (int i = 0; i < obstaclePool.Count; i++)
            if (obstaclePool[i]) Destroy(obstaclePool[i]);
        obstaclePool.Clear();
    }

    // ===== 총알 연출 =====
    void SpawnBulletsFromShots(JArray shots)
    {
        if (shots == null || !bulletPrefab) return;

        Transform parent = bulletRoot != null ? bulletRoot : null;

        foreach (var s in shots)
        {
            var jo = (JObject)s;

            // 새 포맷 호환 (from_xy/to_xy) 또는 구 포맷 (from/to)
            var from = (jo["from_xy"] as JArray) ?? (jo["from"] as JArray);
            var to = (jo["to_xy"] as JArray) ?? (jo["to"] as JArray);
            if (from == null || to == null) continue;

            int fx = from[0].ToObject<int>(), fy = from[1].ToObject<int>();
            int tx = to[0].ToObject<int>(), ty = to[1].ToObject<int>();

            Vector3 wFrom = GridToWorld(fx, fy, bulletYOffset);
            Vector3 wTo = GridToWorld(tx, ty, bulletYOffset);

            Vector3 dir = (wTo - wFrom);
            Vector3 ndir = dir.sqrMagnitude > 1e-10f ? dir.normalized : Vector3.forward;
            wFrom += ndir * (0.3f * cellSize);

            var go = Instantiate(bulletPrefab, wFrom, Quaternion.LookRotation(ndir, Vector3.up), parent);

            var bullet = go.GetComponent<Bullet>();
            if (bullet)
            {
                float spd = Mathf.Max(bulletMinSpeed, bulletSpeedPerCell * cellSize);
                bullet.SetMotion(spd, bulletMaxLife);
                bullet.FireTo(wTo);
            }

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

    // ===== 보조 =====
    Vector3 GridToWorld(int gx, int gy, float yOffset = 0f)
    {
        float cs = (serverCellSize > 0f) ? serverCellSize : cellSize;
        return origin + new Vector3((gx + 0.5f) * cs, yOffset, (gy + 0.5f) * cs);
    }
}
