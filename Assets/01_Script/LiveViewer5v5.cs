using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class LiveViewer5v5 : MonoBehaviour
{
    [Header("Network")]
    public string host = "127.0.0.1";
    public int port = 8765;

    [Header("Prefabs")]
    public GameObject agentAPrefab;   // A팀 에이전트 프리팹(1개 지정하면 n명 자동 생성)
    public GameObject agentBPrefab;   // B팀 에이전트 프리팹(1개 지정하면 n명 자동 생성)
    public GameObject bulletPrefab;   // 총알 프리팹(필수: Bullet.cs 포함)

    [Header("Grid -> World Mapping")]
    public float cellSize = 1.0f;     // 격자 1칸의 월드 크기
    public Vector3 origin = Vector3.zero; // (0,0) 그리드의 월드 기준점
    public float agentYOffset = 0.0f;     // 에이전트 높이 보정
    public float bulletYOffset = 0.3f;    // 총알 발사 높이 보정

    [Header("Debug")]
    public bool showLog = false;

    // 내부 상태
    TcpClient client;
    Thread recvThread;
    volatile bool running;
    ConcurrentQueue<string> lineQueue = new ConcurrentQueue<string>();

    int gridW = 12, gridH = 8, nAgents = 5;
    int currentEpisode = -1;

    readonly List<GameObject> aAgents = new List<GameObject>();
    readonly List<GameObject> bAgents = new List<GameObject>();

    void Start()
    {
        // 네트워크 수신 스레드 시작
        running = true;
        recvThread = new Thread(NetThread) { IsBackground = true };
        recvThread.Start();
    }

    void OnDestroy()
    {
        running = false;
        try { client?.Close(); } catch { }
        try { recvThread?.Join(200); } catch { }
        ClearAgents();
    }

    void NetThread()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            var ns = client.GetStream();
            var sb = new StringBuilder();
            var buf = new byte[8192];

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
        catch (Exception e)
        {
            Debug.LogWarning("[LiveViewer] NetThread error: " + e.Message);
        }
    }

    void Update()
    {
        // 들어온 메시지를 메인스레드에서 처리
        while (lineQueue.TryDequeue(out string line))
        {
            try
            {
                var jo = JObject.Parse(line);
                string type = jo.Value<string>("type");

                if (type == "meta")
                {
                    // 맵/인원 메타 갱신
                    gridW = jo.Value<int>("width");
                    gridH = jo.Value<int>("height");
                    nAgents = jo.Value<int>("n");
                    currentEpisode = jo.Value<int?>("episode") ?? currentEpisode;
                    if (showLog) Debug.Log($"[Live] meta: epi={currentEpisode}, {gridW}x{gridH}, n={nAgents}");
                    // meta 수신 시점에 에이전트 풀 갈아끼움
                    RespawnAgents(nAgents);
                }
                else if (type == "reset")
                {
                    // 새 에피소드 시작 알림
                    currentEpisode = jo.Value<int?>("episode") ?? currentEpisode;
                    if (showLog) Debug.Log($"[Live] reset: epi={currentEpisode}");
                    // 위치는 다음 frame에서 적용되므로 여기서는 스폰만 유지
                }
                else if (type == "frame")
                {
                    ApplyFrame(jo);
                }
                else if (type == "done")
                {
                    if (showLog) Debug.Log($"[Live] episode done: epi={currentEpisode}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[Live] parse error: " + ex.Message);
            }
        }
    }

    // ====== Agents ======
    void ClearAgents()
    {
        foreach (var go in aAgents) if (go) Destroy(go);
        foreach (var go in bAgents) if (go) Destroy(go);
        aAgents.Clear();
        bAgents.Clear();
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
            aAgents.Add(a);
            bAgents.Add(b);
            // 초기에는 원점에 두고, 첫 frame에서 실제 위치로 이동
            a.transform.position = origin + Vector3.up * agentYOffset;
            b.transform.position = origin + Vector3.up * agentYOffset;
            a.SetActive(true);
            b.SetActive(true);
        }
    }

    // ====== Frame Apply ======
    void ApplyFrame(JObject frame)
    {
        var A = (JArray)frame["A"];
        var B = (JArray)frame["B"];

        // 방어 코드: 런타임에 에이전트 수 변동 시 재스폰
        if (A != null && A.Count != aAgents.Count) RespawnAgents(A.Count);
        if (B != null && B.Count != bAgents.Count) RespawnAgents(B.Count);

        int aAlive = 0, bAlive = 0;
        for (int i = 0; i < aAgents.Count; i++)
        {
            var a = (JArray)A[i]; // [x,y,hp,fx,fy,cd]
            var b = (JArray)B[i];
            int ax = a[0].ToObject<int>(), ay = a[1].ToObject<int>(), ahp = a[2].ToObject<int>();
            int bx = b[0].ToObject<int>(), by = b[1].ToObject<int>(), bhp = b[2].ToObject<int>();

            UpdateAgentView(aAgents[i], GridToWorld(ax, ay, agentYOffset), ahp > 0);
            UpdateAgentView(bAgents[i], GridToWorld(bx, by, agentYOffset), bhp > 0);

            if (ahp > 0) aAlive++;
            if (bhp > 0) bAlive++;
        }

        // shots → 총알 생성
        var shots = frame["shots"] as JArray;
        if (shots != null && bulletPrefab)
        {
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

                var go = Instantiate(bulletPrefab, wFrom, Quaternion.LookRotation((wTo - wFrom).normalized, Vector3.up), transform);
                var bullet = go.GetComponent<Bullet>();
                if (bullet) bullet.target = wTo;
            }
        }

        if (showLog)
        {
            float rA = frame["rA"]?.ToObject<float>() ?? 0f;
            float rB = frame["rB"]?.ToObject<float>() ?? 0f;
            int t = frame["t"]?.ToObject<int>() ?? 0;
            Debug.Log($"[t {t}] rA={rA:F2} rB={rB:F2}  A_alive={aAlive} B_alive={bAlive}");
        }
    }

    // ====== Helpers ======
    Vector3 GridToWorld(int gx, int gy, float yOffset = 0f)
    {
        // 격자 (x,y)를 월드 좌표로 변환 (xz 평면 사용)
        return origin + new Vector3(gx * cellSize, yOffset, gy * cellSize);
    }

    void UpdateAgentView(GameObject go, Vector3 pos, bool alive)
    {
        if (!go) return;
        go.transform.position = pos;
        if (go.activeSelf != alive) go.SetActive(alive);
    }
}
