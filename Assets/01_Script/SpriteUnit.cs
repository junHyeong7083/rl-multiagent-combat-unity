using UnityEngine;

/// <summary>
/// 3D 환경에서 스프라이트 유닛을 표시
/// 프리팹 루트에 이 컴포넌트를 추가하면 자동으로 구조 생성
/// </summary>
public class SpriteUnit : MonoBehaviour
{
    [Header("Unit Info")]
    public RoleType role = RoleType.Tank;
    public bool isTeamA = true;
    public bool isPlayer = false;  // 플레이어가 조종하는 유닛

    [Header("Sprite Settings")]
    [Tooltip("null이면 역할에 맞는 스프라이트 자동 생성")]
    public Sprite unitSprite;
    public float spriteScale = 1f;  // 타일 크기와 동일

    [Header("HP Bar Settings")]
    public bool showHpBar = true;
    public float hpBarWidth = 0.8f;
    public float hpBarHeight = 0.08f;
    public float hpBarOffsetY = 0.6f;
    public Color hpBarBackgroundColor = new Color(0.2f, 0.2f, 0.2f, 0.8f);
    public Color hpBarHealthyColor = new Color(0.2f, 0.9f, 0.2f);
    public Color hpBarWarnColor = new Color(0.9f, 0.9f, 0.2f);
    public Color hpBarCriticalColor = new Color(0.9f, 0.2f, 0.2f);

    [Header("Player Indicator")]
    public Color playerIndicatorColor = new Color(1f, 1f, 0f, 0.8f);  // 노란색
    public float playerIndicatorSize = 1.2f;

    [Header("Billboard")]
    public bool lockYAxis = true;

    // 런타임 참조
    private SpriteRenderer _spriteRenderer;
    private SpriteRenderer _hpBarBg;
    private SpriteRenderer _hpBarFill;
    private SpriteRenderer _playerIndicator;
    private Billboard _billboard;
    private float _currentHpBarWidth;

    private bool _initialized = false;

    private void Start()
    {
        // Start에서 초기화 (role, isTeamA가 설정된 후 실행됨)
        Initialize();
    }

    /// <summary>
    /// 수동 초기화 (외부에서 role, isTeamA 설정 후 호출 가능)
    /// </summary>
    public void Initialize()
    {
        if (_initialized) return;
        _initialized = true;

        SetupSprite();
        if (showHpBar) SetupHpBar();
        SetupPlayerIndicator();
        SetupBillboard();
    }

    private void SetupSprite()
    {
        // Sprite 자식 오브젝트 찾기 또는 생성
        Transform spriteChild = transform.Find("Sprite");
        if (spriteChild == null)
        {
            GameObject spriteObj = new GameObject("Sprite");
            spriteObj.transform.SetParent(transform);
            spriteObj.transform.localPosition = Vector3.zero;
            spriteChild = spriteObj.transform;
        }

        _spriteRenderer = spriteChild.GetComponent<SpriteRenderer>();
        if (_spriteRenderer == null)
        {
            _spriteRenderer = spriteChild.gameObject.AddComponent<SpriteRenderer>();
        }

        // 스프라이트가 없으면 자동 생성
        if (unitSprite == null)
        {
            unitSprite = SpriteGenerator.GetRoleSprite(role, isTeamA);
        }

        _spriteRenderer.sprite = unitSprite;
        _spriteRenderer.sortingOrder = 5;
        spriteChild.localScale = Vector3.one * spriteScale;

        // 탑다운 뷰: 스프라이트를 바닥에 눕히기 (XZ 평면)
        spriteChild.localRotation = Quaternion.Euler(90, 0, 0);
    }

    private void SetupHpBar()
    {
        _currentHpBarWidth = hpBarWidth;

        // HP Bar 부모 오브젝트
        Transform hpBarParent = transform.Find("HpBar");
        if (hpBarParent == null)
        {
            GameObject hpBarObj = new GameObject("HpBar");
            hpBarObj.transform.SetParent(transform);
            hpBarObj.transform.localPosition = new Vector3(0, hpBarOffsetY, 0);
            hpBarParent = hpBarObj.transform;
        }
        else
        {
            hpBarParent.localPosition = new Vector3(0, hpBarOffsetY, 0);
        }

        // HP Bar 배경
        Transform bgChild = hpBarParent.Find("Background");
        if (bgChild == null)
        {
            GameObject bgObj = new GameObject("Background");
            bgObj.transform.SetParent(hpBarParent);
            bgObj.transform.localPosition = Vector3.zero;
            bgChild = bgObj.transform;
        }

        _hpBarBg = bgChild.GetComponent<SpriteRenderer>();
        if (_hpBarBg == null)
        {
            _hpBarBg = bgChild.gameObject.AddComponent<SpriteRenderer>();
        }
        _hpBarBg.sprite = CreateSquareSprite();
        _hpBarBg.color = hpBarBackgroundColor;
        _hpBarBg.sortingOrder = 10;
        bgChild.localScale = new Vector3(hpBarWidth, hpBarHeight, 1);

        // HP Bar Fill
        Transform fillChild = hpBarParent.Find("Fill");
        if (fillChild == null)
        {
            GameObject fillObj = new GameObject("Fill");
            fillObj.transform.SetParent(hpBarParent);
            fillObj.transform.localPosition = new Vector3(0, 0, -0.01f);
            fillChild = fillObj.transform;
        }

        _hpBarFill = fillChild.GetComponent<SpriteRenderer>();
        if (_hpBarFill == null)
        {
            _hpBarFill = fillChild.gameObject.AddComponent<SpriteRenderer>();
        }
        _hpBarFill.sprite = CreateSquareSprite();
        _hpBarFill.color = hpBarHealthyColor;
        _hpBarFill.sortingOrder = 11;
        fillChild.localScale = new Vector3(hpBarWidth, hpBarHeight, 1);

        // 탑다운 뷰: HP바도 바닥에 눕히기
        hpBarParent.localRotation = Quaternion.Euler(90, 0, 0);
        hpBarParent.localPosition = new Vector3(0, 0.01f, hpBarOffsetY);  // Y는 약간 위, Z로 오프셋
    }

    private void SetupPlayerIndicator()
    {
        // 플레이어 표시기 (링 형태)
        Transform indicatorChild = transform.Find("PlayerIndicator");
        if (indicatorChild == null)
        {
            GameObject indicatorObj = new GameObject("PlayerIndicator");
            indicatorObj.transform.SetParent(transform);
            indicatorObj.transform.localPosition = new Vector3(0, 0.02f, 0);
            indicatorChild = indicatorObj.transform;
        }

        _playerIndicator = indicatorChild.GetComponent<SpriteRenderer>();
        if (_playerIndicator == null)
        {
            _playerIndicator = indicatorChild.gameObject.AddComponent<SpriteRenderer>();
        }

        _playerIndicator.sprite = CreateRingSprite();
        _playerIndicator.color = playerIndicatorColor;
        _playerIndicator.sortingOrder = 4;  // 스프라이트보다 아래
        indicatorChild.localScale = Vector3.one * playerIndicatorSize;
        indicatorChild.localRotation = Quaternion.Euler(90, 0, 0);

        // 초기 상태: 비활성화
        _playerIndicator.gameObject.SetActive(isPlayer);
    }

    private void SetupBillboard()
    {
        // 탑다운 뷰에서는 Billboard 불필요
        // _billboard = GetComponent<Billboard>();
        // if (_billboard == null)
        // {
        //     _billboard = gameObject.AddComponent<Billboard>();
        // }
        // _billboard.lockYAxis = lockYAxis;
    }

    /// <summary>
    /// 역할과 팀 설정 (런타임에 스프라이트 변경)
    /// </summary>
    public void SetRoleAndTeam(RoleType newRole, bool teamA)
    {
        role = newRole;
        isTeamA = teamA;

        if (_spriteRenderer != null)
        {
            _spriteRenderer.sprite = SpriteGenerator.GetRoleSprite(role, isTeamA);
        }
    }

    /// <summary>
    /// 플레이어 표시 설정
    /// </summary>
    public void SetIsPlayer(bool playerControlled)
    {
        isPlayer = playerControlled;
        if (_playerIndicator != null)
        {
            _playerIndicator.gameObject.SetActive(isPlayer);
        }
    }

    /// <summary>
    /// HP 업데이트
    /// </summary>
    public void SetHp(int hp, int maxHp)
    {
        if (_hpBarFill == null || maxHp <= 0) return;

        float ratio = Mathf.Clamp01((float)hp / maxHp);

        // 스케일 조정 (왼쪽 정렬을 위해 위치도 조정)
        Vector3 scale = _hpBarFill.transform.localScale;
        scale.x = _currentHpBarWidth * ratio;
        _hpBarFill.transform.localScale = scale;

        // 왼쪽 정렬
        Vector3 pos = _hpBarFill.transform.localPosition;
        pos.x = -(_currentHpBarWidth - scale.x) / 2f;
        pos.z = -0.01f;
        _hpBarFill.transform.localPosition = pos;

        // 색상 변경
        if (ratio > 0.5f)
            _hpBarFill.color = hpBarHealthyColor;
        else if (ratio > 0.25f)
            _hpBarFill.color = hpBarWarnColor;
        else
            _hpBarFill.color = hpBarCriticalColor;
    }

    /// <summary>
    /// 스프라이트 변경
    /// </summary>
    public void SetSprite(Sprite sprite)
    {
        unitSprite = sprite;
        if (_spriteRenderer != null)
        {
            _spriteRenderer.sprite = sprite;
        }
    }

    // 1x1 흰색 스프라이트 생성 (HP 바용)
    private static Sprite _squareSprite;
    private static Sprite CreateSquareSprite()
    {
        if (_squareSprite == null)
        {
            Texture2D tex = new Texture2D(1, 1);
            tex.SetPixel(0, 0, Color.white);
            tex.Apply();
            _squareSprite = Sprite.Create(tex, new Rect(0, 0, 1, 1), new Vector2(0.5f, 0.5f), 1);
        }
        return _squareSprite;
    }

    // 플레이어 표시용 링 스프라이트 생성
    private static Sprite _ringSprite;
    private static Sprite CreateRingSprite()
    {
        if (_ringSprite == null)
        {
            int size = 64;
            int thickness = 4;
            Texture2D tex = new Texture2D(size, size);

            // 투명하게 초기화
            Color[] colors = new Color[size * size];
            for (int i = 0; i < colors.Length; i++)
                colors[i] = Color.clear;
            tex.SetPixels(colors);

            // 링 그리기
            float center = size / 2f;
            float outerRadius = size / 2f - 2;
            float innerRadius = outerRadius - thickness;

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float dist = Mathf.Sqrt((x - center) * (x - center) + (y - center) * (y - center));
                    if (dist >= innerRadius && dist <= outerRadius)
                    {
                        tex.SetPixel(x, y, Color.white);
                    }
                }
            }

            tex.Apply();
            _ringSprite = Sprite.Create(tex, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), size);
        }
        return _ringSprite;
    }
}