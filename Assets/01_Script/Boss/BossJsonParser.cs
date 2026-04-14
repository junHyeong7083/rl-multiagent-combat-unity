using System.Collections.Generic;
using System.Text;

namespace BossRaid
{
    /// <summary>
    /// Python snapshot(JSON)용 경량 파서.
    /// JsonUtility가 int[][]를 못 읽어서 직접 구현.
    /// 외부 의존성(Newtonsoft 등) 없이 MiniJSON 스타일로 구현.
    /// </summary>
    internal static class BossJsonParser
    {
        public static BossSnapshot Parse(string json)
        {
            int i = 0;
            object root = ParseValue(json, ref i);
            if (!(root is Dictionary<string, object> d)) return null;

            var snap = new BossSnapshot
            {
                step = GetInt(d, "step"),
                done = GetBool(d, "done"),
                victory = GetBool(d, "victory"),
                wipe = GetBool(d, "wipe"),
                boss = ParseBoss(GetDict(d, "boss")),
                units = ParseUnits(GetList(d, "units")),
                telegraphs = ParseTelegraphs(GetList(d, "telegraphs")),
            };
            return snap;
        }

        private static BossData ParseBoss(Dictionary<string, object> d)
        {
            if (d == null) return null;
            return new BossData
            {
                x = GetInt(d, "x"),
                y = GetInt(d, "y"),
                hp = GetInt(d, "hp"),
                max_hp = GetInt(d, "max_hp"),
                phase = GetInt(d, "phase"),
                invuln = GetInt(d, "invuln"),
                grog = GetInt(d, "grog"),
                stagger_active = GetBool(d, "stagger_active"),
                stagger_gauge = GetFloat(d, "stagger_gauge"),
            };
        }

        private static UnitData[] ParseUnits(List<object> list)
        {
            if (list == null) return new UnitData[0];
            var arr = new UnitData[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                var d = list[i] as Dictionary<string, object>;
                arr[i] = new UnitData
                {
                    uid = GetInt(d, "uid"),
                    role = GetInt(d, "role"),
                    x = GetInt(d, "x"),
                    y = GetInt(d, "y"),
                    hp = GetInt(d, "hp"),
                    max_hp = GetInt(d, "max_hp"),
                    alive = GetBool(d, "alive"),
                    marked = GetBool(d, "marked"),
                    chained_with = GetIntOrDefault(d, "chained_with", -1),
                    buff_atk = GetInt(d, "buff_atk"),
                    buff_shield = GetInt(d, "buff_shield"),
                };
            }
            return arr;
        }

        private static TelegraphData[] ParseTelegraphs(List<object> list)
        {
            if (list == null) return new TelegraphData[0];
            var arr = new TelegraphData[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                var d = list[i] as Dictionary<string, object>;
                var tilesRaw = GetList(d, "danger_tiles");
                int[][] tiles = new int[tilesRaw?.Count ?? 0][];
                for (int t = 0; t < tiles.Length; t++)
                {
                    var tileList = tilesRaw[t] as List<object>;
                    tiles[t] = new int[] {
                        (int)(long)tileList[0],
                        (int)(long)tileList[1]
                    };
                }
                var targetsRaw = GetList(d, "target_uids");
                int[] targets = new int[targetsRaw?.Count ?? 0];
                for (int t = 0; t < targets.Length; t++)
                    targets[t] = (int)(long)targetsRaw[t];

                arr[i] = new TelegraphData
                {
                    pattern = GetInt(d, "pattern"),
                    turns_remaining = GetInt(d, "turns_remaining"),
                    total_wind_up = GetInt(d, "total_wind_up"),
                    danger_tiles = tiles,
                    target_uids = targets,
                };
            }
            return arr;
        }

        // ── Helpers ──
        private static int GetInt(Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v) && v != null ? (int)(long)v : 0;

        private static int GetIntOrDefault(Dictionary<string, object> d, string k, int def)
            => d != null && d.TryGetValue(k, out var v) && v != null ? (int)(long)v : def;

        private static float GetFloat(Dictionary<string, object> d, string k)
        {
            if (d == null || !d.TryGetValue(k, out var v) || v == null) return 0f;
            if (v is double dd) return (float)dd;
            if (v is long ll) return ll;
            return 0f;
        }

        private static bool GetBool(Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v) && v is bool b && b;

        private static Dictionary<string, object> GetDict(Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v) ? v as Dictionary<string, object> : null;

        private static List<object> GetList(Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v) ? v as List<object> : null;

        // ── Minimal JSON tokenizer ──
        private static object ParseValue(string s, ref int i)
        {
            SkipWhite(s, ref i);
            if (i >= s.Length) return null;
            char c = s[i];
            if (c == '{') return ParseObject(s, ref i);
            if (c == '[') return ParseArray(s, ref i);
            if (c == '"') return ParseString(s, ref i);
            if (c == 't' || c == 'f') return ParseBool(s, ref i);
            if (c == 'n') { i += 4; return null; }
            return ParseNumber(s, ref i);
        }

        private static Dictionary<string, object> ParseObject(string s, ref int i)
        {
            var d = new Dictionary<string, object>();
            i++; // {
            SkipWhite(s, ref i);
            if (i < s.Length && s[i] == '}') { i++; return d; }
            while (i < s.Length)
            {
                SkipWhite(s, ref i);
                var key = ParseString(s, ref i);
                SkipWhite(s, ref i);
                if (i < s.Length && s[i] == ':') i++;
                var val = ParseValue(s, ref i);
                d[key] = val;
                SkipWhite(s, ref i);
                if (i < s.Length && s[i] == ',') { i++; continue; }
                if (i < s.Length && s[i] == '}') { i++; break; }
            }
            return d;
        }

        private static List<object> ParseArray(string s, ref int i)
        {
            var l = new List<object>();
            i++; // [
            SkipWhite(s, ref i);
            if (i < s.Length && s[i] == ']') { i++; return l; }
            while (i < s.Length)
            {
                l.Add(ParseValue(s, ref i));
                SkipWhite(s, ref i);
                if (i < s.Length && s[i] == ',') { i++; continue; }
                if (i < s.Length && s[i] == ']') { i++; break; }
            }
            return l;
        }

        private static string ParseString(string s, ref int i)
        {
            i++; // opening "
            var sb = new StringBuilder();
            while (i < s.Length && s[i] != '"')
            {
                if (s[i] == '\\' && i + 1 < s.Length)
                {
                    char nx = s[i + 1];
                    if (nx == 'n') sb.Append('\n');
                    else if (nx == 't') sb.Append('\t');
                    else if (nx == 'r') sb.Append('\r');
                    else sb.Append(nx);
                    i += 2;
                    continue;
                }
                sb.Append(s[i]);
                i++;
            }
            i++; // closing "
            return sb.ToString();
        }

        private static object ParseNumber(string s, ref int i)
        {
            int start = i;
            bool isFloat = false;
            if (i < s.Length && (s[i] == '-' || s[i] == '+')) i++;
            while (i < s.Length && (char.IsDigit(s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+'))
            {
                if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') isFloat = true;
                i++;
            }
            string num = s.Substring(start, i - start);
            if (isFloat) return double.Parse(num, System.Globalization.CultureInfo.InvariantCulture);
            return long.Parse(num, System.Globalization.CultureInfo.InvariantCulture);
        }

        private static bool ParseBool(string s, ref int i)
        {
            if (s[i] == 't') { i += 4; return true; }
            i += 5; return false;
        }

        private static void SkipWhite(string s, ref int i)
        {
            while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
                i++;
        }
    }
}
