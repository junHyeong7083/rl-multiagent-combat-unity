using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// 패턴 텔레그래프 Quad에 붙는 컴포넌트.
    /// BossRaid/Telegraph 쉐이더 파라미터를 제어해 원/부채꼴/빔/십자를 그린다.
    ///
    /// BossGameViewer.RenderShape()가 다음 순서로 호출:
    ///   1) ApplyShape(ShapeData)   — shape 종류, 크기, 회전을 세팅
    ///   2) SetTelegraph(...)       — 색·펄스·진행도
    /// </summary>
    public class TileMarker : MonoBehaviour
    {
        [Header("Renderer")]
        public Renderer rend;

        [Header("Shape Colors — 완전 구분되게 채도↑명도↑")]
        public Color slashColor    = new Color(1.0f,  0.55f, 0.1f,  0.9f);   // Slash: 주황
        public Color chargeColor   = new Color(1.0f,  0.15f, 0.15f, 0.9f);   // Charge: 빨강
        public Color eruptionColor = new Color(1.0f,  0.35f, 0.05f, 0.9f);   // Eruption: 용암 주황빨강
        public Color tailColor     = new Color(0.2f,  1.0f,  0.3f,  0.9f);   // TailSwipe: 독 초록 (꼬리=독)
        public Color markColor     = new Color(0.9f,  0.3f,  1.0f,  0.9f);   // Mark: 밝은 보라
        public Color staggerColor  = new Color(0.2f,  0.85f, 1.0f,  0.85f);  // Stagger: 시안
        public Color crossColor    = new Color(1.0f,  0.1f,  0.5f,  0.95f);  // Cross: 핫핑크
        public Color chainColor    = new Color(1.0f,  0.2f,  1.0f,  0.85f);  // Chain: 마젠타
        public Color sealColor     = new Color(0.2f,  1.0f,  0.4f,  0.9f);   // SealBreak: 밝은 초록
        public Color rimColor      = new Color(1.0f,  1.0f,  0.7f,  1.0f);   // 림 글로우: 밝은 노랑

        // URP SRP Batcher는 MaterialPropertyBlock을 UnityPerMaterial CBUFFER에 대해 무시함.
        // → 인스턴스 머티리얼(.material)을 직접 수정하는 방식 사용.
        [Tooltip("Debug 로그로 색 세팅 확인")]
        public bool debugLog = false;

        private Material _mat;

        private Material EnsureMat()
        {
            if (rend == null) rend = GetComponentInChildren<Renderer>();
            if (rend == null)
            {
                if (debugLog) Debug.LogWarning($"[TileMarker] {name}: Renderer not found!");
                return null;
            }
            // rend.material은 필요 시 인스턴스 생성 (sharedMaterial 복제).
            // 이후 호출은 동일 인스턴스 반환.
            if (_mat == null) _mat = rend.material;
            return _mat;
        }

        private void Awake()
        {
            EnsureMat();
        }

        public void ApplyShape(ShapeData shape)
        {
            var m = EnsureMat();
            if (m == null) return;

            int shapeType = 0;
            float fanHalf = 0.785f;
            float safeMask = 0f;

            switch (shape.kind)
            {
                case "circle": shapeType = 0; break;
                case "fan":    shapeType = 1; fanHalf = Mathf.Max(0.01f, shape.width * 0.5f); break;
                case "line":   shapeType = 2; break;
                case "cross":  shapeType = 3; safeMask = shape.safe_mask; break;
            }

            m.SetInt("_ShapeType", shapeType);
            m.SetFloat("_FanWidthRad", fanHalf);
            m.SetFloat("_SafeMask", safeMask);

            if (debugLog)
                Debug.Log($"[TileMarker] ApplyShape kind={shape.kind} type={shapeType}");
        }

        public void SetTelegraph(int pattern, int turnsRemaining, int totalWindUp)
        {
            var m = EnsureMat();
            if (m == null) return;

            Color baseCol = ColorFor((BossPatternId)pattern);
            float progress = totalWindUp <= 0 ? 1f : 1f - (float)turnsRemaining / totalWindUp;

            m.SetColor("_Color", baseCol);
            m.SetColor("_RimColor", rimColor);
            m.SetFloat("_Progress", Mathf.Clamp01(progress));

            if (debugLog)
                Debug.Log($"[TileMarker] SetTelegraph pattern={pattern} color={baseCol} progress={progress:F2}");
        }

        private Color ColorFor(BossPatternId p)
        {
            switch (p)
            {
                case BossPatternId.Slash: return slashColor;
                case BossPatternId.Charge: return chargeColor;
                case BossPatternId.Eruption: return eruptionColor;
                case BossPatternId.TailSwipe: return tailColor;
                case BossPatternId.Mark: return markColor;
                case BossPatternId.Stagger: return staggerColor;
                case BossPatternId.CrossInferno: return crossColor;
                case BossPatternId.CursedChain: return chainColor;
                case BossPatternId.SealBreak: return sealColor;
                default: return Color.white;
            }
        }
    }
}
