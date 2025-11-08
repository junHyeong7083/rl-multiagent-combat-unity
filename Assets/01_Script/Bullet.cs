using UnityEngine;

public class Bullet : MonoBehaviour
{
    [Header("Motion")]
    [Min(0.01f)] public float speed = 120f;   // ↑ 빠르게 (연출 싱크 개선)
    [Min(0.01f)] public float maxLife = 3f;   // ↑ 약간 여유
    public bool faceDirection = true;

    [Header("Arrival")]
    [Tooltip("목표점과 이 거리 이하이면 도달로 처리")]
    [Min(0f)] public float arriveDistance = 0.03f;

    [Header("Hit")]
    public bool destroyOnArrive = true;
    public GameObject hitEffectPrefab;

    [Header("Trail (optional)")]
    public TrailRenderer trail;

    [Header("Optional Homing")]
    [Tooltip("지정하면 매 프레임 목표를 이 트랜스폼 위치로 보정")]
    public Transform followTarget;
    [Range(0f, 1f)]
    [Tooltip("0이면 고정 목표, 1이면 완전 추적. 0.15~0.35 권장")]
    public float homingStrength = 0.25f;

    [HideInInspector] public Vector3 target;

    float life;
    bool fired;

    /// <summary>스폰 직후 목표 세팅 + 발사</summary>
    public void FireTo(Vector3 worldTarget, Transform follow = null)
    {
        target = worldTarget;
        followTarget = follow; // null이면 고정 목표 모드
        fired = true;

        if (faceDirection)
        {
            Vector3 dir = (target - transform.position);
            if (dir.sqrMagnitude > 1e-8f)
                transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
        }
    }

    /// <summary>런타임으로 속도/수명 보정할 때 사용</summary>
    public void SetMotion(float newSpeed, float newMaxLife)
    {
        if (newSpeed > 0f) speed = newSpeed;
        if (newMaxLife > 0f) maxLife = newMaxLife;
    }

    void OnEnable()
    {
        life = 0f;
        fired = false;
    }

    void Update()
    {
        life += Time.deltaTime;
        if (life >= maxLife)
        {
            DestroySelf();
            return;
        }
        if (!fired) return;

        // 선택적 호밍: 목표가 있으면 부드럽게 목표점 업데이트
        if (followTarget)
        {
            Vector3 desired = followTarget.position;
            target = Vector3.Lerp(target, desired, homingStrength);
        }

        // 타겟으로 직선 이동
        Vector3 to = target - transform.position;
        float dist = to.magnitude;

        // 근접 처리
        if (dist <= arriveDistance)
        {
            transform.position = target;
            Arrive();
            return;
        }

        float step = speed * Time.deltaTime;
        if (step >= dist)
        {
            transform.position = target;
            Arrive();
        }
        else
        {
            Vector3 move = to * (step / dist);
            transform.position += move;

            if (faceDirection && move.sqrMagnitude > 1e-14f)
                transform.rotation = Quaternion.LookRotation(move.normalized, Vector3.up);
        }
    }

    void Arrive()
    {
        if (hitEffectPrefab)
        {
            var fx = Instantiate(hitEffectPrefab, target, Quaternion.identity);
            Destroy(fx, 1.5f);
        }
        if (destroyOnArrive) DestroySelf();
    }

    void DestroySelf()
    {
        // 트레일 잔상 자연 소멸
        if (trail)
        {
            try
            {
                trail.transform.SetParent(null, true);
                if (!trail.gameObject.activeSelf) trail.gameObject.SetActive(true);
                Destroy(trail.gameObject, trail.time + 0.1f);
            }
            catch { /* ignored */ }
        }
        Destroy(gameObject);
    }
}
