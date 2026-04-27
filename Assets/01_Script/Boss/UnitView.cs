using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// 파티 유닛(플레이어 포함) 시각화. 격자 Lerp + HP 바 + 상태 효과.
    /// </summary>
    public class UnitView : MonoBehaviour
    {
        [HideInInspector] public BossGameViewer viewer;
        [HideInInspector] public int uid;

        [Header("Visual")]
        [Tooltip("한 턴(격자 1칸)을 몇 초에 이동할지 (Python TURN_INTERVAL과 맞추기)")]
        public float turnDuration = 0.3f;
        public float rotateLerpSpeed = 12f;
        public Animator animator;

        [Header("Animator Params")]
        public string paramIsMoving = "IsMoving";
        public string paramDead = "Dead";
        public string trigAttack = "TrigAttack";
        public string trigHeal = "TrigHeal";
        public string trigTaunt = "TrigTaunt";
        public string trigBuff = "TrigBuff";
        public string trigHit = "TrigHit";
        public GameObject hpBarRoot;
        public Transform hpBarFill;
        public GameObject deathEffect;
        public GameObject shieldEffect;
        public GameObject buffAtkEffect;
        public GameObject guardEffect;

        private Vector3 _prevPos;
        private Vector3 _targetPos;
        private float _interpStart = -1f;
        private Quaternion _targetRot = Quaternion.identity;
        private GameObject _markInstance;
        private UnitData _latest;
        private bool _hasData;

        private void Awake()
        {
            if (animator == null) animator = GetComponentInChildren<Animator>();
        }

        public void ApplySnapshot(UnitData u)
        {
            _latest = u;

            // 유클리드 float 좌표를 월드 좌표로 직접 변환 (중심 오프셋 불필요)
            var world = viewer.ContinuousToWorld(u.x, u.y);

            if (!_hasData)
            {
                transform.position = world;
                _prevPos = world;
            }
            else
            {
                _prevPos = transform.position;
            }
            _targetPos = world;
            _interpStart = Time.time;
            _hasData = true;

            // HP 바
            if (hpBarFill != null)
            {
                float r = Mathf.Clamp01((float)u.hp / Mathf.Max(1, u.max_hp));
                hpBarFill.localScale = new Vector3(r, 1f, 1f);
            }
            if (hpBarRoot != null) hpBarRoot.SetActive(u.alive);

            // 상태 효과
            if (shieldEffect) shieldEffect.SetActive(u.buff_shield > 0);
            if (buffAtkEffect) buffAtkEffect.SetActive(u.buff_atk > 0);

            // 사망
            if (!u.alive)
            {
                if (deathEffect && !deathEffect.activeSelf) deathEffect.SetActive(true);
                if (animator != null) animator.SetBool("Dead", true);
            }
        }

        // Animator 파라미터 캐시 (없는 파라미터 호출 시 경고 방지)
        private System.Collections.Generic.HashSet<string> _animParams;

        private bool HasParam(string name)
        {
            if (animator == null || string.IsNullOrEmpty(name)) return false;
            if (_animParams == null)
            {
                _animParams = new System.Collections.Generic.HashSet<string>();
                foreach (var p in animator.parameters) _animParams.Add(p.name);
            }
            return _animParams.Contains(name);
        }

        private void SafeSetTrigger(string name)
        {
            if (HasParam(name)) animator.SetTrigger(name);
        }

        private void SafeSetBool(string name, bool v)
        {
            if (HasParam(name)) animator.SetBool(name, v);
        }

        /// <summary>Python에서 발생한 이벤트에 맞춰 애니메이션 트리거.</summary>
        public void OnEvent(EventData ev)
        {
            if (animator == null || ev == null || string.IsNullOrEmpty(ev.type)) return;
            switch (ev.type)
            {
                case "damage":       SafeSetTrigger(trigAttack); break;
                case "heal":         SafeSetTrigger(trigHeal); break;
                case "taunt":        SafeSetTrigger(trigTaunt); break;
                case "buff":         SafeSetTrigger(trigBuff); break;
                case "damage_taken": SafeSetTrigger(trigHit); break;
                case "death":        SafeSetBool(paramDead, true); break;
            }
        }

        public void ShowMark(GameObject markPrefab, int turnsRemaining)
        {
            if (_markInstance == null)
                _markInstance = Instantiate(markPrefab, transform);
            _markInstance.SetActive(true);
            // 턴이 0에 가까울수록 붉어지도록 자식 Renderer가 처리한다고 가정
        }

        private void Update()
        {
            if (!_hasData) return;

            float elapsed = Time.time - _interpStart;
            float t = Mathf.Clamp01(elapsed / Mathf.Max(0.01f, turnDuration));
            float te = t * t * (3f - 2f * t); // smoothstep
            Vector3 newPos = Vector3.LerpUnclamped(_prevPos, _targetPos, te);

            var moveDir = newPos - transform.position;
            transform.position = newPos;

            if (animator != null)
            {
                bool moving = t < 0.95f && (_prevPos - _targetPos).sqrMagnitude > 0.01f;
                SafeSetBool(paramIsMoving, moving);
                if (_latest != null)
                    SafeSetBool(paramDead, !_latest.alive);
            }

            // 이동 중이면 이동 방향, 정지면 보스를 바라봄 (전투 중 자연스러움)
            Vector3 faceTarget;
            bool haveFace = false;
            if (moveDir.sqrMagnitude > 0.0004f)
            {
                faceTarget = transform.position + moveDir.normalized;
                haveFace = true;
            }
            else if (viewer != null && viewer.TryGetBossPosition(out var bossPos))
            {
                faceTarget = bossPos;
                haveFace = true;
            }
            else faceTarget = transform.position + transform.forward;

            if (haveFace)
            {
                var flat = new Vector3(faceTarget.x - transform.position.x, 0, faceTarget.z - transform.position.z);
                if (flat.sqrMagnitude > 0.0001f)
                    _targetRot = Quaternion.LookRotation(flat.normalized, Vector3.up);
            }
            transform.rotation = Quaternion.Slerp(transform.rotation, _targetRot, Time.deltaTime * rotateLerpSpeed);
        }

        private void LateUpdate()
        {
            // 표식이 꺼져야 하면 여기서 제어 (스냅샷의 marked 필드로)
            if (_markInstance != null && _hasData && !_latest.marked)
                _markInstance.SetActive(false);
        }
    }
}
