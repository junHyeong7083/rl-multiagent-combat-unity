Shader "BossRaid/Telegraph"
{
    // 패턴 텔레그래프 전용 쉐이더.
    // Quad(1x1, 바닥에 평평히 회전된 상태) 위에 shape 종류에 따라 실제 도형을 그림.
    // Quad의 localScale과 rotation은 C#에서 shape에 맞게 맞춰주면 됨.
    //
    // _ShapeType:
    //   0 = Circle        (UV 중심 기준 원)
    //   1 = Fan           (UV 중심 오른쪽 방향 부채꼴, _FanWidthRad 반각)
    //   2 = Line (beam)   (가로 전체, 세로는 중앙 띠)
    //   3 = Cross         (가로+세로 십자 + 안전 사분면)
    //
    // _Progress: 0(wind-up 시작) → 1(발동 직전), 펄스 속도에 반영
    Properties
    {
        _Color("Base Color", Color) = (1, 0.3, 0.3, 0.55)
        _RimColor("Rim/Glow Color", Color) = (1, 1, 0.4, 1)
        _ShapeType("Shape (0=Circle 1=Fan 2=Line 3=Cross)", Int) = 0
        _Progress("Progress 0-1", Range(0, 1)) = 0
        _FanWidthRad("Fan Half-Angle (rad)", Range(0, 3.15)) = 0.785
        _EdgeSoftness("Edge Softness", Range(0.001, 0.2)) = 0.04
        _PulseSpeed("Pulse Speed", Range(0, 12)) = 5
        _LineWidth("Line Width (UV)", Range(0, 1)) = 0.15
        _CrossBandWidth("Cross Band Half Width (UV)", Range(0, 0.5)) = 0.08
        _SafeMask("Safe Quad Mask (bits 0-3)", Float) = 0
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent+10" "RenderPipeline"="UniversalPipeline" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attr { float4 positionOS: POSITION; float2 uv: TEXCOORD0; };
            struct V2F  { float4 positionCS: SV_POSITION; float2 uv: TEXCOORD0; };

            CBUFFER_START(UnityPerMaterial)
                float4 _Color;
                float4 _RimColor;
                int _ShapeType;
                float _Progress;
                float _FanWidthRad;
                float _EdgeSoftness;
                float _PulseSpeed;
                float _LineWidth;
                float _CrossBandWidth;
                float _SafeMask;
            CBUFFER_END

            V2F vert(Attr IN)
            {
                V2F OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = IN.uv;
                return OUT;
            }

            // UV(0~1) 중심 기준 좌표로 변환
            float2 centered(float2 uv) { return uv - 0.5; }

            float maskCircle(float2 p)
            {
                float r = length(p) * 2.0;  // 0(center) ~ 1(edge)
                return smoothstep(1.0, 1.0 - _EdgeSoftness, r);
            }

            float maskFan(float2 p)
            {
                float r = length(p) * 2.0;
                if (r > 1.0) return 0;
                float ang = atan2(p.y, p.x);        // -PI ~ PI, +X축 기준
                float diff = abs(ang);              // 부채꼴 중앙이 +X(오른쪽) 기준
                float inside = smoothstep(_FanWidthRad + _EdgeSoftness, _FanWidthRad - _EdgeSoftness, diff);
                float edge = smoothstep(1.0, 1.0 - _EdgeSoftness, r);
                return inside * edge;
            }

            float maskLine(float2 p)
            {
                // 가로 전체, 세로는 중앙 띠 (Quad를 빔 방향으로 회전시킨 가정)
                float d = abs(p.y);
                return smoothstep(_LineWidth, _LineWidth - _EdgeSoftness, d);
            }

            float maskCross(float2 p)
            {
                float hBand = smoothstep(_CrossBandWidth, _CrossBandWidth - _EdgeSoftness, abs(p.y));
                float vBand = smoothstep(_CrossBandWidth, _CrossBandWidth - _EdgeSoftness, abs(p.x));
                float cross = max(hBand, vBand);

                // 십자에 맞았으면 무조건 위험
                if (cross > 0.001) return cross;

                // 그 외 사분면: safe_mask에 걸리면 안전(0 반환), 아니면 약한 경고색
                int q = 0;
                if (p.x >= 0) q |= 1;
                if (p.y >= 0) q |= 2;
                int mask = (int)_SafeMask;
                bool isSafe = ((mask >> q) & 1) == 1;
                return isSafe ? 0.0 : 0.25;
            }

            half4 frag(V2F IN) : SV_Target
            {
                float2 p = centered(IN.uv);

                float mask = 0;
                if (_ShapeType == 0) mask = maskCircle(p);
                else if (_ShapeType == 1) mask = maskFan(p);
                else if (_ShapeType == 2) mask = maskLine(p);
                else if (_ShapeType == 3) mask = maskCross(p);

                if (mask <= 0.001) discard;

                // 펄스: wind-up 끝날수록 빨라지고 강해짐
                float pulseHz = lerp(2.0, _PulseSpeed, _Progress);
                float pulse = 0.5 + 0.5 * sin(_Time.y * pulseHz * 6.283);
                float intensity = lerp(0.3, 1.0, _Progress);

                // 림 영향은 최대 30%만 (색 정체성 유지). 전환은 프로그레스 후반부에만.
                float rimMix = pulse * smoothstep(0.5, 1.0, _Progress) * 0.3;
                float3 col = lerp(_Color.rgb, _RimColor.rgb, rimMix);
                // 추가로 펄스에 따라 명도 살짝 증폭 (색은 유지)
                col *= lerp(1.0, 1.3, pulse * _Progress);

                float a = _Color.a * mask * intensity * lerp(0.75, 1.0, pulse);
                return half4(col, a);
            }
            ENDHLSL
        }
    }
}
