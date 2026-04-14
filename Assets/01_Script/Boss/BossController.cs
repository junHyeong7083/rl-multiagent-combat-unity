using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// 보스 3D 오브젝트 컨트롤러. 격자 좌표를 Lerp로 보간해 자연스럽게 이동.
    /// 2x2 타일을 점유하므로 중심 오프셋 (+0.5, 0, +0.5) 적용.
    /// 패턴 텔레그래프 시작을 감지해 Animator 트리거 발동.
    /// </summary>
    public class BossController : MonoBehaviour
    {
        [HideInInspector] public BossGameViewer viewer;

        [Header("Visual")]
        public float moveLerpSpeed = 6f;
        public float rotateLerpSpeed = 8f;
        public Animator animator;
        public Renderer[] bodyRenderers;
        public Color phase1Color = Color.white;
        public Color phase2Color = new Color(1f, 0.5f, 0.3f);
        public Color phase3Color = new Color(1f, 0.2f, 0.2f);

        [Header("Effects")]
        public GameObject invulnEffect;
        public GameObject groggyEffect;
        public GameObject staggerEffect;

        [Header("Animator Params")]
        public string paramPhase = "Phase";
        public string paramGroggy = "Groggy";
        public string paramDead = "Dead";
        public string paramIsMoving = "IsMoving";
        public string trigSlash = "TrigSlash";
        public string trigCharge = "TrigCharge";
        public string trigJump = "TrigJump";
        public string trigRoar = "TrigRoar";
        public string trigTail = "TrigTail";
        [Tooltip("Target까지 이 거리 이상이면 이동 중으로 판정")]
        public float movingThreshold = 0.05f;

        private Vector3 _targetPos;
        private Quaternion _targetRot = Quaternion.identity;
        private BossData _latestData;
        private bool _hasData;

        // 이전 프레임 텔레그래프 상태 (신규 발동 감지용)
        private readonly System.Collections.Generic.HashSet<int> _prevTelegraphIds = new System.Collections.Generic.HashSet<int>();

        public void ApplySnapshot(BossData b)
        {
            _latestData = b;
            _hasData = true;

            // 2x2 중심 = (x+0.5, y+0.5)
            var world = viewer.GridToWorld(b.x, b.y)
                        + new Vector3(0.5f * viewer.cellSize, 0f, 0.5f * viewer.cellSize);
            _targetPos = world;

            ApplyPhaseColor(b.phase);

            if (invulnEffect) invulnEffect.SetActive(b.invuln > 0);
            if (groggyEffect) groggyEffect.SetActive(b.grog > 0);
            if (staggerEffect) staggerEffect.SetActive(b.stagger_active);

            if (animator != null)
            {
                animator.SetBool(paramGroggy, b.grog > 0);
                animator.SetInteger(paramPhase, b.phase);
            }
        }

        /// <summary>
        /// 스냅샷의 텔레그래프 리스트를 받아 "이번 프레임에 새로 시작된 것"만
        /// 트리거 발동. BossGameViewer에서 호출.
        /// </summary>
        public void OnTelegraphs(TelegraphData[] telegraphs)
        {
            if (animator == null || telegraphs == null) return;

            var current = new System.Collections.Generic.HashSet<int>();
            foreach (var tg in telegraphs)
            {
                // 고유 식별자: pattern + wind_up 시작 시점 (turns_remaining == total_wind_up일 때가 시작)
                int key = tg.pattern * 1000 + tg.turns_remaining;
                current.Add(tg.pattern);

                // 신규 발동: turns_remaining == total_wind_up 이고, 이전 프레임에 없었음
                if (tg.turns_remaining == tg.total_wind_up && !_prevTelegraphIds.Contains(tg.pattern))
                {
                    FirePatternAnim((BossPatternId)tg.pattern);
                }
            }
            _prevTelegraphIds.Clear();
            foreach (var id in current) _prevTelegraphIds.Add(id);
        }

        private void FirePatternAnim(BossPatternId p)
        {
            switch (p)
            {
                case BossPatternId.Slash:        animator.SetTrigger(trigSlash); break;
                case BossPatternId.Charge:       animator.SetTrigger(trigCharge); break;
                case BossPatternId.Eruption:     animator.SetTrigger(trigJump); break;
                case BossPatternId.TailSwipe:    animator.SetTrigger(trigTail); break;
                case BossPatternId.Mark:         animator.SetTrigger(trigRoar); break;
                case BossPatternId.Stagger:      animator.SetTrigger(trigRoar); break;
                case BossPatternId.CrossInferno: animator.SetTrigger(trigRoar); break;
                case BossPatternId.CursedChain:  animator.SetTrigger(trigRoar); break;
            }
        }

        public void SetDead(bool dead)
        {
            if (animator != null) animator.SetBool(paramDead, dead);
        }

        private void Update()
        {
            if (!_hasData) return;
            var diff = _targetPos - transform.position;
            float dist = diff.magnitude;

            transform.position = Vector3.Lerp(transform.position, _targetPos, Time.deltaTime * moveLerpSpeed);
            if (diff.sqrMagnitude > 0.0001f)
            {
                _targetRot = Quaternion.LookRotation(new Vector3(diff.x, 0, diff.z).normalized, Vector3.up);
            }
            transform.rotation = Quaternion.Slerp(transform.rotation, _targetRot, Time.deltaTime * rotateLerpSpeed);

            if (animator != null)
            {
                bool moving = dist > movingThreshold && (_latestData == null || _latestData.grog <= 0);
                animator.SetBool(paramIsMoving, moving);
            }
        }

        // ─── 에디터 테스트용 ───
        [ContextMenu("Test Phase 0 (White)")] private void TestPhase0() { ApplyPhaseColor(0); }
        [ContextMenu("Test Phase 1 (Orange)")] private void TestPhase1() { ApplyPhaseColor(1); }
        [ContextMenu("Test Phase 2 (Red)")]    private void TestPhase2() { ApplyPhaseColor(2); }

        private void ApplyPhaseColor(int phase)
        {
            if (bodyRenderers == null || bodyRenderers.Length == 0) return;
            Color c = phase == 0 ? phase1Color : phase == 1 ? phase2Color : phase3Color;
            foreach (var r in bodyRenderers)
            {
                if (r != null && r.sharedMaterial != null && r.material.HasProperty("_BaseColor"))
                    r.material.SetColor("_BaseColor", c);
                else if (r != null && r.material != null)
                    r.material.color = c;
            }
        }
    }
}
