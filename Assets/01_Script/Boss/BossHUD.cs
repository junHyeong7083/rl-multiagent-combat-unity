using UnityEngine;
using UnityEngine.UI;

namespace BossRaid
{
    /// <summary>
    /// 보스 HP 바, 페이즈, 스태거 게이지, 그로기 표시.
    /// TextMeshPro 대신 기본 Text로 구성 (에셋 의존성 최소화).
    /// </summary>
    public class BossHUD : MonoBehaviour
    {
        [Header("Boss HP")]
        public Slider bossHpSlider;
        public Text bossHpText;

        [Header("Stagger")]
        public GameObject staggerRoot;
        public Slider staggerSlider;

        [Header("Phase / Status")]
        public Text phaseText;
        public Text statusText;     // 무적 / 그로기 / 일반

        [Header("Result")]
        public GameObject resultPanel;
        public Text resultText;

        public void ApplySnapshot(BossSnapshot snap)
        {
            if (snap.boss != null)
            {
                float r = Mathf.Clamp01((float)snap.boss.hp / Mathf.Max(1, snap.boss.max_hp));
                if (bossHpSlider) bossHpSlider.value = r;
                if (bossHpText) bossHpText.text = $"{snap.boss.hp} / {snap.boss.max_hp}";

                if (staggerRoot) staggerRoot.SetActive(snap.boss.stagger_active);
                if (staggerSlider)
                {
                    const float max = 300f;
                    staggerSlider.value = Mathf.Clamp01(snap.boss.stagger_gauge / max);
                }

                if (phaseText) phaseText.text = $"PHASE {snap.boss.phase + 1}";

                if (statusText)
                {
                    if (snap.boss.invuln > 0) statusText.text = $"INVULN ({snap.boss.invuln})";
                    else if (snap.boss.grog > 0) statusText.text = $"GROGGY ({snap.boss.grog})";
                    else statusText.text = "";
                }
            }

            if (resultPanel)
            {
                resultPanel.SetActive(snap.done);
                if (resultText && snap.done)
                {
                    resultText.text = snap.victory ? "CLEAR!" : (snap.wipe ? "WIPE" : "TIMEOUT");
                }
            }
        }
    }
}
