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

        [Header("Scene Refs")]
        public Transform unitsRoot;
        public Transform decalsRoot;
        public BossHUD hud;

        private BossController _boss;
        private readonly Dictionary<int, UnitView> _units = new Dictionary<int, UnitView>();
        private readonly List<GameObject> _decalPool = new List<GameObject>();
        private readonly List<GameObject> _chainPool = new List<GameObject>();

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
                // 보스 애니메이션 트리거 (신규 시전만)
                if (_boss != null) _boss.OnTelegraphs(snap.telegraphs);
            }

            // HUD
            if (hud != null) hud.ApplySnapshot(snap);
        }

        // ─────────────── 텔레그래프 렌더 ───────────────

        private void RenderTelegraph(TelegraphData tg)
        {
            // 위험 타일 표시
            if (tg.danger_tiles != null && tileMarkerPrefab != null)
            {
                foreach (var t in tg.danger_tiles)
                {
                    var go = GetDecal();
                    go.transform.position = GridToWorld(t[0], t[1]);
                    // 페이즈별로 색 강도를 조절하려면 child Renderer에 값 전달
                    var marker = go.GetComponent<TileMarker>();
                    if (marker != null)
                        marker.SetTelegraph(tg.pattern, tg.turns_remaining, tg.total_wind_up);
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
    }
}
