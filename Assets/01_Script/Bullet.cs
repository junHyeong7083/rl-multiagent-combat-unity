using UnityEngine;

public class Bullet : MonoBehaviour
{
    [Header("Motion")]
    [Min(0.01f)] public float speed = 30f;   // units/sec
    [Min(0.01f)] public float maxLife = 2f;  // seconds
    public bool faceDirection = true;

    [Header("Hit")]
    public bool destroyOnArrive = true;
    public GameObject hitEffectPrefab;

    [Header("Trail (optional)")]
    public TrailRenderer trail;

    [HideInInspector] public Vector3 target;

    float life;
    bool fired;

    /// <summary>스폰 직후 목표 세팅 + 발사</summary>
    public void FireTo(Vector3 worldTarget)
    {
        target = worldTarget;
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

        // 타겟으로 직선 이동
        Vector3 to = target - transform.position;
        float sqr = to.sqrMagnitude;

        if (sqr <= 1e-10f)
        {
            // 거의 도달한 상태면 즉시 처리
            transform.position = target;
            Arrive();
            return;
        }

        float dist = Mathf.Sqrt(sqr);
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
                // TrailRenderer가 비활성화될 수 있으니 먼저 켜두기
                if (!trail.gameObject.activeSelf) trail.gameObject.SetActive(true);
                Destroy(trail.gameObject, trail.time + 0.1f);
            }
            catch { /* ignored */ }
        }
        Destroy(gameObject);
    }
}
