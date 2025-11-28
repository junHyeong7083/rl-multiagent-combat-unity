using UnityEngine;

/// <summary>
/// 3D 캡슐 유닛 컴포넌트
/// HP 바를 월드 스페이스에 표시
/// </summary>
public class Unit3D : MonoBehaviour
{
    [Header("Unit Info")]
    public bool isTeamA = true;
    public bool isPlayer = false;
    public Renderer capsuleRenderer;

    [Header("HP Bar Settings")]
    public float hpBarWidth = 0.6f;
    public float hpBarHeight = 0.08f;
    public float hpBarOffsetY = 1.0f;
    public Color hpBarBackgroundColor = new Color(0.2f, 0.2f, 0.2f, 0.8f);
    public Color hpBarHealthyColor = new Color(0.2f, 0.9f, 0.2f);
    public Color hpBarWarnColor = new Color(0.9f, 0.9f, 0.2f);
    public Color hpBarCriticalColor = new Color(0.9f, 0.2f, 0.2f);

    [Header("Player Indicator")]
    public Color playerIndicatorColor = new Color(1f, 1f, 0f, 0.8f);
    public float playerIndicatorRadius = 0.5f;

    // HP 바 오브젝트
    private GameObject _hpBarBg;
    private GameObject _hpBarFill;
    private Material _hpBarBgMat;
    private Material _hpBarFillMat;

    // 플레이어 표시기
    private GameObject _playerIndicator;
    private Material _playerIndicatorMat;

    private void Start()
    {
        CreateHpBar();
        CreatePlayerIndicator();
    }

    private void CreateHpBar()
    {
        // HP 바 부모
        GameObject hpBarParent = new GameObject("HpBar");
        hpBarParent.transform.SetParent(transform);
        hpBarParent.transform.localPosition = new Vector3(0f, hpBarOffsetY, 0f);

        // 배경
        _hpBarBg = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _hpBarBg.name = "Background";
        _hpBarBg.transform.SetParent(hpBarParent.transform);
        _hpBarBg.transform.localPosition = Vector3.zero;
        _hpBarBg.transform.localScale = new Vector3(hpBarWidth, hpBarHeight, 1f);

        // 배경 콜라이더 제거
        var bgCollider = _hpBarBg.GetComponent<Collider>();
        if (bgCollider != null) Destroy(bgCollider);

        _hpBarBgMat = CreateUnlitMaterial(hpBarBackgroundColor);
        _hpBarBg.GetComponent<Renderer>().material = _hpBarBgMat;

        // Fill
        _hpBarFill = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _hpBarFill.name = "Fill";
        _hpBarFill.transform.SetParent(hpBarParent.transform);
        _hpBarFill.transform.localPosition = new Vector3(0f, 0f, -0.01f);
        _hpBarFill.transform.localScale = new Vector3(hpBarWidth, hpBarHeight, 1f);

        // Fill 콜라이더 제거
        var fillCollider = _hpBarFill.GetComponent<Collider>();
        if (fillCollider != null) Destroy(fillCollider);

        _hpBarFillMat = CreateUnlitMaterial(hpBarHealthyColor);
        _hpBarFill.GetComponent<Renderer>().material = _hpBarFillMat;

        // Billboard 컴포넌트 추가
        hpBarParent.AddComponent<Billboard>();
    }

    private void CreatePlayerIndicator()
    {
        // 바닥에 노란색 링 표시
        _playerIndicator = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        _playerIndicator.name = "PlayerIndicator";
        _playerIndicator.transform.SetParent(transform);
        _playerIndicator.transform.localPosition = new Vector3(0f, 0.02f, 0f);
        _playerIndicator.transform.localScale = new Vector3(playerIndicatorRadius * 2f, 0.02f, playerIndicatorRadius * 2f);

        // 콜라이더 제거
        var collider = _playerIndicator.GetComponent<Collider>();
        if (collider != null) Destroy(collider);

        _playerIndicatorMat = CreateUnlitMaterial(playerIndicatorColor);
        _playerIndicator.GetComponent<Renderer>().material = _playerIndicatorMat;

        // 초기 상태: 비활성화
        _playerIndicator.SetActive(isPlayer);
    }

    /// <summary>
    /// 플레이어 표시 설정
    /// </summary>
    public void SetIsPlayer(bool playerControlled)
    {
        isPlayer = playerControlled;
        if (_playerIndicator != null)
        {
            _playerIndicator.SetActive(isPlayer);
        }
    }

    /// <summary>
    /// HP 업데이트
    /// </summary>
    public void SetHp(int hp, int maxHp)
    {
        if (_hpBarFill == null || maxHp <= 0) return;

        float ratio = Mathf.Clamp01((float)hp / maxHp);

        // 스케일 조정 (왼쪽 정렬)
        Vector3 scale = _hpBarFill.transform.localScale;
        scale.x = hpBarWidth * ratio;
        _hpBarFill.transform.localScale = scale;

        // 왼쪽 정렬
        Vector3 pos = _hpBarFill.transform.localPosition;
        pos.x = -(hpBarWidth - scale.x) / 2f;
        _hpBarFill.transform.localPosition = pos;

        // 색상 변경
        if (_hpBarFillMat != null)
        {
            if (ratio > 0.5f)
                _hpBarFillMat.color = hpBarHealthyColor;
            else if (ratio > 0.25f)
                _hpBarFillMat.color = hpBarWarnColor;
            else
                _hpBarFillMat.color = hpBarCriticalColor;
        }
    }

    /// <summary>
    /// 팀 색상 변경
    /// </summary>
    public void SetTeam(bool teamA)
    {
        isTeamA = teamA;
        if (capsuleRenderer != null)
        {
            capsuleRenderer.material.color = teamA
                ? new Color(0.2f, 0.4f, 1.0f)
                : new Color(1.0f, 0.2f, 0.2f);
        }
    }

    /// <summary>
    /// URP/Built-in 호환 Unlit Material 생성
    /// </summary>
    private Material CreateUnlitMaterial(Color color)
    {
        // URP Unlit 시도
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Unlit/Color");
        if (shader == null) shader = Shader.Find("Legacy Shaders/Diffuse");

        Material mat = new Material(shader);

        // URP용
        if (mat.HasProperty("_BaseColor"))
        {
            mat.SetColor("_BaseColor", color);
        }
        // Built-in용
        if (mat.HasProperty("_Color"))
        {
            mat.SetColor("_Color", color);
        }

        return mat;
    }
}
