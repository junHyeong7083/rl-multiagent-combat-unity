using UnityEngine;

/// <summary>
/// 스프라이트가 항상 카메라를 바라보도록 회전
/// </summary>
public class Billboard : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("바라볼 카메라 (null이면 Main Camera 사용)")]
    public Camera targetCamera;

    [Tooltip("Y축만 회전 (수직 유지)")]
    public bool lockYAxis = false;

    private void Start()
    {
        if (targetCamera == null)
        {
            targetCamera = Camera.main;
        }
    }

    private void LateUpdate()
    {
        if (targetCamera == null) return;

        if (lockYAxis)
        {
            // Y축만 회전 (캐릭터가 기울어지지 않음)
            Vector3 lookDir = targetCamera.transform.position - transform.position;
            lookDir.y = 0;
            if (lookDir != Vector3.zero)
            {
                transform.rotation = Quaternion.LookRotation(-lookDir);
            }
        }
        else
        {
            // 완전히 카메라를 향함
            transform.LookAt(targetCamera.transform);
            transform.Rotate(0, 180, 0);
        }
    }
}