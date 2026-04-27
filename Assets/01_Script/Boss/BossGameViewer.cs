using System.Collections.Generic;
using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// 씬 전체 조율: 스냅샷 수신 → 보스/유닛/텔레그래프 갱신.
    /// 격자 좌표 → 월드 좌표 변환 + Lerp 보간.
    /// </summary>
    public class BossGameViewer : MonoBehaviour
    {
        [Header("Grid → World")]
        public float cellSize = 1.0f;
        public Vector3 gridOrigin = Vector3.zero;

        [Header("Prefabs")]
        public GameObject bossPrefab;
        public GameObject[] unitPrefabsByRole = new GameObject[4];   // 0=Dealer, 1=Tank, 2=Healer, 3=Support
        public GameObject tileMarkerPrefab;       // 위험 타일 표시 (붉은 데칼)
        public GameObject markDecalPrefab;        // 표식(보라) 데칼
        public GameObject chainLinePrefab;        // 연결선 (LineRenderer 포함)

        [Header("Party Visuals (임시: 같은 프리팹 재사용 시 구분용)")]
        [Tooltip("파티 유닛 스케일 (보스 대비 축소)")]
        public float partyUnitScale = 0.6f;
        [Tooltip("역할별 틴트 색상: 0=Dealer, 1=Tank, 2=Healer, 3=Support")]
        public Color[] roleTintColors = new Color[]
        {
            new Color(1.0f, 0.9f, 0.3f),   // Dealer - 노랑 (플레이어)
            new Color(0.3f, 0.5f, 1.0f),   // Tank - 파랑
            new Color(0.4f, 1.0f, 0.5f),   // Healer - 초록
            new Color(1.0f, 0.4f, 0.8f),   // Support - 핑크
        };
        [Tooltip("파티 유닛의 보스용 이펙트(Invuln/Groggy/Stagger) 자식이 있으면 자동 비활성화")]
        public string[] bossOnlyEffectNames = new[] { "InvulnEffect", "GroggyEffect", "StaggerEffect" };

        [Header("Scene Refs")]
        public Transform unitsRoot;
        public Transform decalsRoot;
        public BossHUD hud;

        private BossController _boss;
        private readonly Dictionary<int, UnitView> _units = new Dictionary<int, UnitView>();
        private readonly List<GameObject> _decalPool = new List<GameObject>();
        private readonly List<GameObject> _chainPool = new List<GameObject>();

        /// <summary>BossController에서 가장 가까운 유닛 위치 조회용.</summary>
        public bool TryGetNearestUnitPosition(Vector3 from, out Vector3 pos)
        {
            pos = Vector3.zero;
            float best = float.MaxValue;
            bool found = false;
            foreach (var kv in _units)
            {
                var v = kv.Value;
                if (v == null) continue;
                float d = (v.transform.position - from).sqrMagnitude;
                if (d < best) { best = d; pos = v.transform.position; found = true; }
            }
            return found;
        }

        /// <summary>UnitView가 보스를 바라보도록 위치 조회.</summary>
        public bool TryGetBossPosition(out Vector3 pos)
        {
            if (_boss != null) { pos = _boss.transform.position; return true; }
            pos = Vector3.zero; return false;
        }

        private BossSnapshot _latestSnap;

        private void Update()
        {
            var rx = BossUdpReceiver.Instance;
            if (rx == null) return;

            // 최신 스냅샷까지 모두 소비 (렌더링은 최신 것만 반영)
            while (rx.TryDequeue(out var snap))
            {
                _latestSnap = snap;
                ApplySnapshot(snap);
            }

            // Lerp 보간은 각 컴포넌트의 Update에서
        }

        public Vector3 GridToWorld(int gx, int gy)
            => gridOrigin + new Vector3(gx * cellSize, 0f, gy * cellSize);

        /// <summary>유클리드 float 좌표를 Unity 월드 좌표로 변환.</summary>
        public Vector3 ContinuousToWorld(float cx, float cy)
            => gridOrigin + new Vector3(cx * cellSize, 0f, cy * cellSize);

        // ─────────────── 스냅샷 적용 ───────────────

        private void ApplySnapshot(BossSnapshot snap)
        {
            // 보스
            if (snap.boss != null)
            {
                if (_boss == null && bossPrefab != null)
                {
                    var go = Instantiate(bossPrefab, transform);
                    _boss = go.GetComponent<BossController>();
                    if (_boss == null) _boss = go.AddComponent<BossController>();
                    _boss.viewer = this;
                }
                _boss?.ApplySnapshot(snap.boss);
            }

            // 유닛
            if (snap.units != null)
            {
                foreach (var u in snap.units)
                {
                    if (!_units.TryGetValue(u.uid, out var view))
                    {
                        var prefab = unitPrefabsByRole != null && u.role < unitPrefabsByRole.Length
                            ? unitPrefabsByRole[u.role] : null;
                        if (prefab == null) continue;
                        var go = Instantiate(prefab, unitsRoot != null ? unitsRoot : transform);

                        // 스케일 축소 (보스 대비 파티 작게)
                        go.transform.localScale *= partyUnitScale;

                        // 역할별 틴트 색상 적용
                        if (roleTintColors != null && u.role >= 0 && u.role < roleTintColors.Length)
                        {
                            ApplyRoleTint(go, roleTintColors[u.role]);
                        }

                        // 보스 전용 이펙트 자식이 있으면 비활성화
                        DisableBossOnlyEffects(go);

                        view = go.GetComponent<UnitView>();
                        if (view == null) view = go.AddComponent<UnitView>();
                        view.viewer = this;
                        view.uid = u.uid;
                        _units[u.uid] = view;
                    }
                    view.ApplySnapshot(u);
                }
            }

            // 텔레그래프 (데칼 풀 리셋 후 재배치)
            ResetDecalPool();
            if (snap.telegraphs != null)
            {
                foreach (var tg in snap.telegraphs)
                {
                    RenderTelegraph(tg);
                }
                if (_boss != null) _boss.OnTelegraphs(snap.telegraphs);
            }

            // 이벤트 dispatch (유닛별 애니메이션 트리거)
            if (snap.events != null)
            {
                foreach (var ev in snap.events)
                {
                    if (ev == null || ev.uid < 0) continue;
                    if (_units.TryGetValue(ev.uid, out var view))
                        view.OnEvent(ev);
                }
            }

            if (hud != null) hud.ApplySnapshot(snap);
        }

        // ─────────────── 텔레그래프 렌더 (유클리드 기하 도형) ───────────────

        /// <summary>
        /// tileMarkerPrefab을 "shape 단위"로 재활용해서 기하 도형을 Quad로 표시.
        /// Slash/TailSwipe = Fan, Charge = Line, Eruption = Circle(여러 개), Cross = Cross 등.
        /// TileMarker 컴포넌트가 Transform.scale을 받아 적절히 커버하도록 가정.
        /// </summary>
        private void RenderTelegraph(TelegraphData tg)
        {
            if (tg.shapes != null && tileMarkerPrefab != null)
            {
                foreach (var shape in tg.shapes)
                {
                    RenderShape(tg, shape);
                }
            }

            // 표식
            if ((BossPatternId)tg.pattern == BossPatternId.Mark && tg.target_uids != null)
            {
                foreach (var uid in tg.target_uids)
                {
                    if (_units.TryGetValue(uid, out var view) && markDecalPrefab != null)
                    {
                        view.ShowMark(markDecalPrefab, tg.turns_remaining);
                    }
                }
            }

            // 연결선 (CursedChain)
            if ((BossPatternId)tg.pattern == BossPatternId.CursedChain && tg.target_uids != null && tg.target_uids.Length >= 2)
            {
                if (_units.TryGetValue(tg.target_uids[0], out var a) &&
                    _units.TryGetValue(tg.target_uids[1], out var b) &&
                    chainLinePrefab != null)
                {
                    var go = GetChain();
                    var line = go.GetComponent<ChainLine>();
                    if (line != null) line.Setup(a.transform, b.transform);
                }
            }
        }

        [Header("Debug")]
        public bool debugLog = false;

        /// <summary>
        /// 기하 도형 하나를 바닥 Quad로 표시.
        /// Quad는 바닥에 평평히 눕혀진 상태(X=90 회전)라 가정.
        /// 여기서는 월드 Y축 회전 + 크기만 조정. 실제 도형은 TileMarker + BossRaid/Telegraph 쉐이더가 그림.
        /// </summary>
        private void RenderShape(TelegraphData tg, ShapeData shape)
        {
            var go = GetDecal();
            // TileMarker를 루트/자식/부모 어디서든 찾도록 강화
            var marker = go.GetComponent<TileMarker>();
            if (marker == null) marker = go.GetComponentInChildren<TileMarker>();
            if (marker == null) marker = go.GetComponentInParent<TileMarker>();
            if (debugLog)
                Debug.Log($"[BossGameViewer] RenderShape kind={shape.kind} pattern={tg.pattern} markerFound={(marker!=null)} go={go.name}");
            var tr = go.transform;
            Quaternion baseTilt = Quaternion.Euler(90f, 0f, 0f);   // 바닥에 평평히

            switch (shape.kind)
            {
                case "circle":
                {
                    tr.position = ContinuousToWorld(shape.cx, shape.cy) + Vector3.up * 0.02f;
                    float d = shape.r * 2f * cellSize;
                    tr.localScale = new Vector3(d, d, 1f);
                    tr.rotation = baseTilt;
                    break;
                }
                case "fan":
                {
                    // Quad 중심이 보스, +X축(UV)이 부채꼴 중앙 방향.
                    // Python angle = atan2(dy, dx), Unity Y rotation: 좌표계 매칭 위해 부호 유지.
                    tr.position = ContinuousToWorld(shape.cx, shape.cy) + Vector3.up * 0.02f;
                    float d = shape.r * 2f * cellSize;
                    tr.localScale = new Vector3(d, d, 1f);
                    float yawDeg = shape.angle * Mathf.Rad2Deg;
                    tr.rotation = Quaternion.Euler(0f, yawDeg, 0f) * baseTilt;
                    break;
                }
                case "line":
                {
                    // 빔: Quad의 X=길이, Y=폭. 쉐이더는 UV에서 _LineWidth(±0.15)만큼 세로 띠를 그림.
                    // 월드 빔 폭 = 실제 hw*2. Quad Y 스케일이 그만큼이어야 쉐이더 띠가 실제 두께가 됨.
                    // 쉐이더는 UV 전체(0~1)를 쓰는데, _LineWidth=0.15라 UV ±0.15 = 전체 세로의 30%만 칠함.
                    // 따라서 Quad Y = (원하는 월드 폭) / 0.3.
                    var a = ContinuousToWorld(shape.ax, shape.ay);
                    var b = ContinuousToWorld(shape.bx, shape.by);
                    var mid = (a + b) * 0.5f + Vector3.up * 0.02f;
                    var dir = b - a; dir.y = 0;
                    float len = dir.magnitude;
                    float widthWorld = shape.hw * 2f * cellSize;
                    float quadY = widthWorld / 0.30f;   // 쉐이더 _LineWidth=0.15 기준
                    tr.position = mid;
                    tr.localScale = new Vector3(Mathf.Max(0.1f, len), quadY, 1f);
                    if (dir.sqrMagnitude > 1e-5f)
                    {
                        float yawDeg = Mathf.Atan2(dir.z, dir.x) * Mathf.Rad2Deg;
                        tr.rotation = Quaternion.Euler(0f, yawDeg, 0f) * baseTilt;
                    }
                    else tr.rotation = baseTilt;
                    break;
                }
                case "cross":
                {
                    tr.position = ContinuousToWorld(shape.cx, shape.cy) + Vector3.up * 0.02f;
                    float size = Mathf.Max(shape.cx, shape.cy) * 2f * cellSize;
                    tr.localScale = new Vector3(size, size, 1f);
                    tr.rotation = baseTilt;
                    break;
                }
            }

            if (marker != null)
            {
                marker.ApplyShape(shape);
                marker.SetTelegraph(tg.pattern, tg.turns_remaining, tg.total_wind_up);
            }
        }

        private GameObject GetDecal()
        {
            foreach (var d in _decalPool)
            {
                if (!d.activeSelf) { d.SetActive(true); return d; }
            }
            var go = Instantiate(tileMarkerPrefab, decalsRoot != null ? decalsRoot : transform);
            _decalPool.Add(go);
            return go;
        }

        private void ResetDecalPool()
        {
            foreach (var d in _decalPool) d.SetActive(false);
            foreach (var c in _chainPool) c.SetActive(false);
        }

        private GameObject GetChain()
        {
            foreach (var c in _chainPool)
            {
                if (!c.activeSelf) { c.SetActive(true); return c; }
            }
            var go = Instantiate(chainLinePrefab, decalsRoot != null ? decalsRoot : transform);
            _chainPool.Add(go);
            return go;
        }

        /// <summary>역할별 색상으로 자식 Renderer들의 머티리얼 색을 틴트.</summary>
        private static void ApplyRoleTint(GameObject root, Color tint)
        {
            var renderers = root.GetComponentsInChildren<Renderer>(true);
            foreach (var r in renderers)
            {
                // 파티클/LineRenderer 등은 건드리지 않음
                if (r is ParticleSystemRenderer) continue;
                if (r is LineRenderer) continue;
                if (r is TrailRenderer) continue;

                // 머티리얼 인스턴스(복제) 사용: sharedMaterial 건드리면 프리팹 전체 오염
                var mat = r.material;
                if (mat == null) continue;

                if (mat.HasProperty("_BaseColor"))
                    mat.SetColor("_BaseColor", tint);
                else if (mat.HasProperty("_Color"))
                    mat.SetColor("_Color", tint);
            }
        }

        /// <summary>보스 전용 이펙트 자식 이름에 매칭되는 자식들을 비활성화.</summary>
        private void DisableBossOnlyEffects(GameObject root)
        {
            if (bossOnlyEffectNames == null) return;
            foreach (var name in bossOnlyEffectNames)
            {
                var t = root.transform.Find(name);
                if (t != null) t.gameObject.SetActive(false);
            }
        }
    }
}
