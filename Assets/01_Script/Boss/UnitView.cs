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
        public float moveLerpSpeed = 8f;
        public float rotateLerpSpeed = 10f;
        public Animator animator;
        public GameObject hpBarRoot;
        public Transform hpBarFill;
        public GameObject deathEffect;
        public GameObject shieldEffect;
        public GameObject buffAtkEffect;
        public GameObject guardEffect;

        private Vector3 _targetPos;
        private Quaternion _targetRot = Quaternion.identity;
        private GameObject _markInstance;
        private UnitData _latest;
        private bool _hasData;

        public void ApplySnapshot(UnitData u)
        {
            _latest = u;
            _hasData = true;

            var world = viewer.GridToWorld(u.x, u.y)
                        + new Vector3(0.5f * viewer.cellSize, 0f, 0.5f * viewer.cellSize);
            _targetPos = world;

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
            transform.position = Vector3.Lerp(transform.position, _targetPos, Time.deltaTime * moveLerpSpeed);
            var diff = _targetPos - transform.position;
            if (diff.sqrMagnitude > 0.0001f)
            {
                _targetRot = Quaternion.LookRotation(new Vector3(diff.x, 0, diff.z).normalized, Vector3.up);
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
