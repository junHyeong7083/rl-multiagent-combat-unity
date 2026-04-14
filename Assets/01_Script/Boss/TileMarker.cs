using UnityEngine;

namespace BossRaid
{
    /// <summary>
    /// 위험 타일(텔레그래프) 데칼.
    /// 패턴별 색상 + 남은 턴 비율에 따른 알파/채도 제어.
    /// </summary>
    public class TileMarker : MonoBehaviour
    {
        [Header("Renderer")]
        public Renderer rend;
        public string colorProperty = "_BaseColor";

        [Header("Colors by Pattern")]
        public Color slashColor = new Color(1f, 0.6f, 0.1f, 0.6f);
        public Color chargeColor = new Color(1f, 0.2f, 0.2f, 0.6f);
        public Color eruptionColor = new Color(1f, 0.1f, 0.1f, 0.7f);
        public Color tailColor = new Color(0.9f, 0.4f, 0.1f, 0.6f);
        public Color markColor = new Color(0.7f, 0.2f, 1f, 0.7f);
        public Color staggerColor = new Color(0.2f, 0.8f, 1f, 0.5f);
        public Color crossColor = new Color(1f, 0.2f, 0.4f, 0.8f);
        public Color chainColor = new Color(0.9f, 0.1f, 0.9f, 0.6f);

        public void SetTelegraph(int pattern, int turnsRemaining, int totalWindUp)
        {
            if (rend == null) rend = GetComponentInChildren<Renderer>();
            if (rend == null) return;

            var c = ColorFor((BossPatternId)pattern);
            // 남은 턴 비율: 1 → 0 으로 갈수록 알파 강해짐
            float progress = totalWindUp <= 0 ? 1f : 1f - (float)turnsRemaining / totalWindUp;
            c.a = Mathf.Lerp(0.3f, 1.0f, progress);

            if (rend.material.HasProperty(colorProperty))
                rend.material.SetColor(colorProperty, c);
            else
                rend.material.color = c;
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
                default: return Color.white;
            }
        }
    }
}
