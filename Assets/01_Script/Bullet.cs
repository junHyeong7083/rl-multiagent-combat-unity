using UnityEngine;

public class Bullet : MonoBehaviour
{
    [Header("Motion")]
    public float speed = 30f;          // 이동 속도 (units/sec)
    public float maxLife = 2f;         // 자동 파괴 시간 (초)
    public bool faceDirection = true;  // 진행 방향으로 회전할지

    [Header("Hit")]
    public bool destroyOnArrive = true; // 목표 도달 시 파괴
    public GameObject hitEffectPrefab;  // 선택: 도착 시 이펙트

    [Header("Trail (optional)")]
    public TrailRenderer trail;        // 선택: 트레일 지정 시 파괴 타이밍에 잔상 자연 삭제

    [HideInInspector] public Vector3 target;

    float life;

    /// <summary>
    /// 코드에서 스폰 직후 호출해서 목표를 세팅하고 즉시 발사.
    /// </summary>
    public void FireTo(Vector3 worldTarget)
    {
        target = worldTarget;
        if (faceDirection)
        {
            Vector3 dir = (target - transform.position);
            if (dir.sqrMagnitude > 1e-6f)
                transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
        }
    }

    void Update()
    {
        life += Time.deltaTime;
        if (life >= maxLife)
        {
            DestroySelf();
            return;
        }

        // 타겟으로 직선 이동
        Vector3 dir = target - transform.position;
        float dist = dir.magnitude;

        if (dist <= Mathf.Epsilon)
        {
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
            Vector3 move = dir / dist * step;
            transform.position += move;

            if (faceDirection && move.sqrMagnitude > 1e-10f)
                transform.rotation = Quaternion.LookRotation(move.normalized, Vector3.up);
        }
    }

    void Arrive()
    {
        if (hitEffectPrefab != null)
        {
            var fx = Instantiate(hitEffectPrefab, target, Quaternion.identity);
            Destroy(fx, 1.5f);
        }
        if (destroyOnArrive) DestroySelf();
    }

    void DestroySelf()
    {
        // 트레일이 달려있으면 잔상 자연 소멸을 위해 분리 후 일정 시간 뒤 파괴
        if (trail != null)
        {
            // 트레일을 분리해서 잔상 유지
            trail.transform.parent = null;
            // 잔상 길이만큼 기다렸다가 파괴
            Destroy(trail.gameObject, trail.time + 0.1f);
        }
        Destroy(gameObject);
    }
}
