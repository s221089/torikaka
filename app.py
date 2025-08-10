# app.py
# Streamlit Pomodoro + Notes + AI Summary/Quiz (Gemini) + Obsidian風グラフ
# 使い方:
#   1) .env に GEMINI_API_KEY=... を入れる
#   2) pip install -r requirements.txt
#   3) streamlit run app.py

import os, time, sqlite3, json, re, tempfile
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
import tempfile, pathlib

# Gemini
import google.generativeai as genai

# Graph
from pyvis.network import Network

# ============== 基本設定 ==============
st.set_page_config(page_title="とりかか Note Pomodoro", page_icon="🍅", layout="wide")
st.markdown("""
<style>
.status-pill {display:inline-block; padding:6px 12px; border-radius:999px; background:#eee; font-weight:600;}
.status-work {background:#ffe5e5; color:#8a1f1f;}
.status-break{background:#e6f6ff; color:#134b7d;}
.status-idle {background:#efefef; color:#333;}
.small-muted {color:#6b7280; font-size:12px;}
blockquote {border-left: 4px solid #ddd; margin: .5rem 0; padding:.25rem .75rem; color:#374151; background:#fafafa;}
</style>
""", unsafe_allow_html=True)

# ============== API鍵 ==============
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY が未設定です（.env に GEMINI_API_KEY=...）。")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")

# ============== DB ==============
DB = "torikaka_note_pomo.db"
conn = sqlite3.connect(DB, check_same_thread=False)
cur = conn.cursor()
cur.execute("PRAGMA journal_mode=WAL;")
cur.execute("PRAGMA synchronous=NORMAL;")
cur.execute("PRAGMA foreign_keys=ON;")

cur.execute("""
CREATE TABLE IF NOT EXISTS cycles(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  d TEXT,            -- YYYY-MM-DD
  start_at TEXT,     -- ISO
  work_min INTEGER,  -- 実作業分
  note TEXT,         -- ノート
  title TEXT         -- ノートタイトル（Obsidian風）
)""")
cur.execute("""
CREATE TABLE IF NOT EXISTS quizzes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  d TEXT,
  summary TEXT,
  quiz_json TEXT,
  score REAL
)""")
conn.commit()

# 既存DBに title 列が無い場合だけ追加（例外で判定）
try:
    cur.execute("ALTER TABLE cycles ADD COLUMN title TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass

# ============== ヘルパ ==============
def _make_title_fallback(text: str) -> str:
    head = (text or "").strip().splitlines()[0] if text else ""
    head = head[:30] if head else ""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return head if head else f"Note {ts}"

def save_cycle(worked_min: int, today: str):
    note_text = (st.session_state.get("note_draft","") or "").strip()
    title_txt = (st.session_state.get("note_title","") or "").strip() or _make_title_fallback(note_text)
    cur.execute(
        "INSERT INTO cycles(d, start_at, work_min, note, title) VALUES(?,?,?,?,?)",
        (today, datetime.now().isoformat(timespec="seconds"), worked_min, note_text, title_txt)
    )
    conn.commit()
    st.session_state.completed_sessions += 1
    st.session_state["note_draft"] = ""
    st.session_state["note_title"] = ""
    st.session_state.work_minutes_snap = 0

# ============== 状態 ==============
def init_state():
    st.session_state.setdefault("phase", "idle")          # idle | work | break
    st.session_state.setdefault("t_start", None)
    st.session_state.setdefault("work_minutes_snap", 0)
    st.session_state.setdefault("note_draft", "")
    st.session_state.setdefault("note_title", "")
    st.session_state.setdefault("completed_sessions", 0)
    st.session_state.setdefault("current_break_min", 5)
    st.session_state.setdefault("day_summary", "")
    st.session_state.setdefault("quiz_data", None)
    st.session_state.setdefault("quiz_answers", {})
    st.session_state.setdefault("settings", {
        "WORK_MIN": 25,
        "SHORT_BREAK_MIN": 5,
        "LONG_BREAK_MIN": 15,
        "LONG_BREAK_EVERY": 4
    })
init_state()

S = st.session_state["settings"]
WORK_MIN = int(S["WORK_MIN"])
SHORT_BREAK_MIN = int(S["SHORT_BREAK_MIN"])
LONG_BREAK_MIN = int(S["LONG_BREAK_MIN"])
LONG_BREAK_EVERY = int(S["LONG_BREAK_EVERY"])

# 作業/休憩中は毎秒更新
if st.session_state.phase in ("work", "break") and st.session_state.t_start:
    st_autorefresh(interval=1000, key=f"tick-{st.session_state.phase}")

# ============== ヘッダ ==============
today = date.today().isoformat()
df_today = pd.read_sql_query("SELECT id, start_at, work_min, note FROM cycles WHERE d=? ORDER BY id", conn, params=(today,))
total_min_today = int(df_today["work_min"].sum()) if len(df_today) else 0

left, right = st.columns([0.62, 0.38], vertical_alignment="center")
with left:
    st.title("🍅 とりかか：25分→ノート→休憩")
    phase = st.session_state.phase
    pill = "status-idle" if phase=="idle" else ("status-work" if phase=="work" else "status-break")
    st.markdown(f'<span class="status-pill {pill}">状態: {phase.upper()}</span>', unsafe_allow_html=True)
with right:
    st.subheader("今日のダッシュボード")
    c1, c2 = st.columns(2)
    with c1: st.metric("本数", max(len(df_today), st.session_state.completed_sessions))
    with c2: st.metric("合計", f"{total_min_today} 分")
    next_count = (st.session_state.completed_sessions + 1)
    next_is_long = (next_count % LONG_BREAK_EVERY == 0)
    st.caption(f"次の休憩: {'長休憩 '+str(LONG_BREAK_MIN)+'分' if next_is_long else '短休憩 '+str(SHORT_BREAK_MIN)+'分'}（{LONG_BREAK_EVERY}セットごとに長休憩）")

# ============== サイドバー設定 ==============
with st.sidebar:
    st.header("⚙️ 設定")
    st.caption("各値は次のサイクルから反映。")
    n_work = st.number_input("作業(分)", min_value=10, max_value=120, value=WORK_MIN, step=5)
    n_sbrk = st.number_input("短休憩(分)", min_value=1, max_value=60, value=SHORT_BREAK_MIN, step=1)
    n_lbrk = st.number_input("長休憩(分)", min_value=5, max_value=120, value=LONG_BREAK_MIN, step=5)
    n_every = st.number_input("何本ごとに長休憩？", min_value=2, max_value=10, value=LONG_BREAK_EVERY, step=1)
    if st.button("保存して反映"):
        st.session_state["settings"] = {
            "WORK_MIN": int(n_work),
            "SHORT_BREAK_MIN": int(n_sbrk),
            "LONG_BREAK_MIN": int(n_lbrk),
            "LONG_BREAK_EVERY": int(n_every),
        }
        st.success("設定を保存しました。")

    st.markdown("---")
    st.subheader("📤 エクスポート")
    df_export = pd.read_sql_query("SELECT * FROM cycles ORDER BY id", conn)
    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⤵ 作業ログ CSV をダウンロード", data=csv_bytes, file_name="cycles.csv", mime="text/csv")

    df_today_full_export = pd.read_sql_query("SELECT start_at, work_min, note FROM cycles WHERE d=? ORDER BY id", conn, params=(today,))
    md = ["# 今日のノート", f"- 日付: {today}", ""]
    for _, r in df_today_full_export.iterrows():
        md.append(f"## {r['start_at']}（{r['work_min']}分）")
        md.append(r["note"] or "(ノートなし)")
        md.append("")
    md_text = "\n".join(md)
    st.download_button("⤵ 今日のノート Markdown", data=md_text.encode("utf-8"), file_name=f"notes_{today}.md", mime="text/markdown")

# ============== レイアウト ==============
main_col, side_col = st.columns([0.66, 0.34])

# ---------- メイン ----------
with main_col:
    st.subheader("コントロール")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        start_btn = st.button(f"▶ {WORK_MIN}分スタート", use_container_width=True, disabled=(st.session_state.phase!="idle"))
    with c2:
        end_early_btn = st.button("⏹ 早期終了→保存", use_container_width=True, disabled=(st.session_state.phase!="work"))
    with c3:
        save_btn = st.button("💾 ノート保存（休憩中）", use_container_width=True, disabled=(st.session_state.phase!="break"))
    with c4:
        skip_break = st.button("⏭ 休憩スキップ", use_container_width=True, disabled=(st.session_state.phase!="break"))
    with c5:
        reset_btn = st.button("■ 停止/リセット", use_container_width=True, disabled=(st.session_state.phase=="idle"))

    # フェーズ開始
    if start_btn:
        st.session_state.phase = "work"
        st.session_state.t_start = time.time()
        st.session_state.work_minutes_snap = 0

    # 早期終了→保存→休憩へ
    if end_early_btn and st.session_state.phase == "work" and st.session_state.t_start:
        elapsed = int(time.time() - st.session_state.t_start)
        worked_min = max(1, min(WORK_MIN, max(1, elapsed // 60)))
        save_cycle(worked_min, today)
        next_is_long = (st.session_state.completed_sessions % LONG_BREAK_EVERY == 0)
        st.session_state.current_break_min = LONG_BREAK_MIN if next_is_long else SHORT_BREAK_MIN
        st.success(f"⏹ 早期終了で保存（{worked_min}分）。休憩 {st.session_state.current_break_min} 分へ。")
        st.session_state.phase = "break"
        st.session_state.t_start = time.time()

    # 休憩中ノート保存
    if save_btn and st.session_state.phase == "break":
        worked_min = min(WORK_MIN, st.session_state.work_minutes_snap or WORK_MIN)
        save_cycle(worked_min, today)
        st.success("💾 ノートを保存しました（休憩は継続）。")

    # 休憩スキップ
    if skip_break and st.session_state.phase == "break":
        st.session_state.phase = "idle"
        st.session_state.t_start = None

    # リセット
    if reset_btn:
        st.session_state.phase = "idle"
        st.session_state.t_start = None
        st.session_state.work_minutes_snap = 0
        st.session_state["note_draft"] = ""
        st.session_state["note_title"] = ""

    # タイマー可視化
    def countdown(title: str, total_min: int) -> Tuple[int, int]:
        elapsed = int(time.time() - st.session_state.t_start) if st.session_state.t_start else 0
        remain = max(0, total_min*60 - elapsed)
        st.metric(title, f"{remain//60:02d}:{remain%60:02d}")
        st.progress(1 - (remain/(total_min*60)) if total_min>0 else 0)
        return elapsed, remain

    # 自動遷移
    if st.session_state.phase == "work" and st.session_state.t_start:
        elapsed, remain = countdown("作業 残り", WORK_MIN)
        st.session_state.work_minutes_snap = max(st.session_state.work_minutes_snap, elapsed//60)
        if remain == 0:
            worked_min = WORK_MIN
            save_cycle(worked_min, today)
            next_is_long = (st.session_state.completed_sessions % LONG_BREAK_EVERY == 0)
            st.session_state.current_break_min = LONG_BREAK_MIN if next_is_long else SHORT_BREAK_MIN
            st.success(f"✅ {WORK_MIN}分完了 → 休憩 {st.session_state.current_break_min} 分に入ります。")
            st.session_state.phase = "break"
            st.session_state.t_start = time.time()

    elif st.session_state.phase == "break" and st.session_state.t_start:
        _, remain_b = countdown("休憩 残り", st.session_state.current_break_min)
        if remain_b == 0:
            st.session_state.phase = "idle"
            st.session_state.t_start = None
            st.balloons()
            st.success("🍵 休憩おわり。いつでも次のサイクルへ。")

    # ノート入力
    st.subheader("📝 ノート（下書き）")
    st.text_input(
        "ノートタイトル（例：分散の定義メモ / 2025-08-10-01）",
        key="note_title",
        value=st.session_state.get("note_title", ""),
        placeholder="未入力なら自動で日時＋先頭文を使用"
    )
    st.caption("作業中も下書きOK。保存は休憩中に「💾 ノート保存」。")
    _ = st.text_area(
        "・何をやった？ / ・発見 / ・次やること（任意）",
        value=st.session_state.get("note_draft",""),
        height=160,
        key="note_draft",
        placeholder="例）分散の定義を確認。numpy.varで実験。次は標準化の実装。"
    )

# ---------- サイド：週次 & 履歴 ----------
with side_col:
    st.subheader("🗓 週次サマリ（7日間・作業時間のみ）")
    default_start = date.today() - timedelta(days=6)
    start_day = st.date_input("開始日（この日から7日間）", value=default_start, key="week_start")
    end_day = start_day + timedelta(days=6)

    df_all = pd.read_sql_query("SELECT d, work_min FROM cycles", conn)
    if len(df_all):
        df_all["d"] = pd.to_datetime(df_all["d"], errors="coerce").dt.date
        mask = (df_all["d"] >= start_day) & (df_all["d"] <= end_day)
        wk = df_all.loc[mask].copy()
    else:
        wk = pd.DataFrame(columns=["d","work_min"])

    index_days = pd.date_range(start_day, end_day, freq="D").date
    wk_agg = wk.groupby("d", dropna=True)["work_min"].sum().reset_index(name="total_min")
    wk_full = pd.DataFrame({"d": index_days}).merge(wk_agg, how="left", on="d").fillna({"total_min": 0})
    wk_full["total_min"] = wk_full["total_min"].astype(int)

    chart_min = (
        alt.Chart(wk_full)
        .mark_bar()
        .encode(
            x=alt.X("d:T", title="日付"),
            y=alt.Y("total_min:Q", title="分", scale=alt.Scale(nice=True)),
            tooltip=[alt.Tooltip("d:T", title="日付"), alt.Tooltip("total_min:Q", title="分")]
        )
        .properties(height=220)
    )
    st.altair_chart(chart_min, use_container_width=True)
    st.caption(f"期間: {start_day} 〜 {end_day}｜合計 {int(wk_full['total_min'].sum())} 分")

    st.subheader("📒 ノート履歴")
    dates = pd.read_sql_query("SELECT DISTINCT d FROM cycles ORDER BY d DESC", conn)["d"].tolist()
    if not dates: dates = [today]
    sel = st.selectbox("日付", options=dates, index=0)
    df_sel = pd.read_sql_query("SELECT start_at, work_min, note, title FROM cycles WHERE d=? ORDER BY id", conn, params=(sel,))
    if len(df_sel):
        for _, r in df_sel.iterrows():
            title = r.get("title") or _make_title_fallback(r["note"] or "")
            with st.expander(f"{r['start_at']} / {r['work_min']}分 / {title}"):
                st.write(r["note"] or "(ノートなし)")
    else:
        st.info("この日のノートはありません。")

# ============== AIユーティリティ ==============
def _clean_json_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"^```.*?```", "", s, flags=re.DOTALL | re.MULTILINE)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return s[start:end+1]

def _validate_quiz_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict): return False
    if "quiz" not in obj or not isinstance(obj["quiz"], list): return False
    for item in obj["quiz"]:
        if not all(k in item for k in ("q", "choices", "a", "explain")):
            return False
        if not isinstance(item["choices"], list) or len(item["choices"]) != 4:
            return False
        if not isinstance(item["a"], str) or item["a"][:1] not in "ABCD":
            return False
    return True

def ai_day_summary(notes_text: str) -> str:
    if not notes_text.strip():
        return "(ノートがありません)"
    try:
        prompt = "以下は今日の学習ノート。専門用語は崩さず、100字以内で要点要約：\n" + notes_text
        res = MODEL.generate_content(prompt)
        return (res.text or "").strip()
    except Exception as e:
        return f"(要約失敗: {e})"

def ai_quiz_from_notes(notes_text: str) -> Dict[str, Any]:
    fallback = {
        "quiz":[
            {"q":"今日の重要語は？","choices":["A. 分散","B. 回帰","C. API","D. Unicode"],"a":"A","explain":"ノート頻出語"}
        ]
    }
    if not notes_text.strip():
        return fallback
    prompt = f"""
以下の学習ノートから、理解確認用の4択クイズを3問作成。
- 正答は1つ（A/B/C/D のいずれか）
- 各問に簡潔な解説を付ける
- choices は "A. ～", "B. ～", "C. ～", "D. ～" の4要素
ノート:
{notes_text}

出力JSON:
{{
 "quiz": [
   {{"q":"", "choices":["A. ","B. ","C. ","D. "], "a":"A", "explain":""}},
   {{"q":"", "choices":["A. ","B. ","C. ","D. "], "a":"B", "explain":""}},
   {{"q":"", "choices":["A. ","B. ","C. ","D. "], "a":"C", "explain":""}}
 ]
}}
"""
    try:
        raw = MODEL.generate_content(prompt).text or ""
        jtxt = _clean_json_text(raw)
        obj = json.loads(jtxt) if jtxt else None
        if obj and _validate_quiz_schema(obj):
            return obj
    except Exception:
        pass
    return fallback

# ============== 今日の復習（要約＋小テスト） ==============
st.markdown("---")
st.subheader("🧠 今日の復習（要約 & 小テスト）")

df_today_full = pd.read_sql_query("SELECT note FROM cycles WHERE d=? ORDER BY id", conn, params=(today,))
all_notes = "\n\n".join(df_today_full["note"].fillna("").tolist()) if len(df_today_full) else ""

cS1, cS2 = st.columns(2)
with cS1:
    if st.button("🧾 要約を作る", use_container_width=True, disabled=(len(df_today_full)==0)):
        st.session_state.day_summary = ai_day_summary(all_notes)
with cS2:
    if st.button("📝 小テストを作る（3問）", use_container_width=True, disabled=(len(df_today_full)==0)):
        if len(df_today_full):
            st.session_state.quiz_data = ai_quiz_from_notes(all_notes)
            st.session_state.quiz_answers = {}
        else:
            st.warning("ノートがないためクイズを作成できません。")

if st.session_state.day_summary:
    st.success(st.session_state.day_summary)

if st.session_state.quiz_data:
    st.markdown("---")
    st.subheader("小テスト")
    for i, q in enumerate(st.session_state.quiz_data.get("quiz", [])):
        st.write(f"Q{i+1}. {q.get('q','')}")
        choices = q.get("choices", [])
        choice_idx = st.radio("選択", list(range(len(choices))), format_func=lambda k: choices[k], key=f"q_{i}")
        if st.button(f"回答 Q{i+1}", key=f"btn_{i}"):
            selected_label = choices[choice_idx] if 0 <= choice_idx < len(choices) else ""
            selected_letter = selected_label[:1].strip()
            correct_letter = q.get("a","").strip()[:1]
            ok = (selected_letter == correct_letter)
            st.session_state.quiz_answers[i] = {
                "selected": selected_label,
                "correct": correct_letter,
                "ok": ok
            }
            st.info(f"あなたの答え: {selected_label} / 正解: {correct_letter}")
            st.caption(q.get("explain",""))

    if st.button("📌 結果を保存"):
        score = sum(1 for v in st.session_state.quiz_answers.values() if v.get("ok"))
        try:
            cur.execute(
                "INSERT INTO quizzes(d, summary, quiz_json, score) VALUES(?,?,?,?)",
                (today, st.session_state.day_summary, json.dumps(st.session_state.quiz_data, ensure_ascii=False), float(score))
            )
            conn.commit()
            st.success(f"保存しました（スコア: {score} / {len(st.session_state.quiz_data.get('quiz', []))}）")
        except Exception as e:
            st.error(f"保存に失敗しました: {e}")

# ============== Obsidian風：ノートグラフ & バックリンク ==============
st.markdown("---")
st.subheader("🕸 ノートグラフ（Obsidian風）")

range_opt = st.selectbox("対象", ["今日", "直近7日", "すべて"])
if range_opt == "今日":
    df_notes = pd.read_sql_query(
        "SELECT id, d, start_at, title, note FROM cycles WHERE d=? ORDER BY id",
        conn, params=(today,)
    )
elif range_opt == "直近7日":
    d0 = (date.today() - timedelta(days=6)).isoformat()
    df_notes = pd.read_sql_query(
        "SELECT id, d, start_at, title, note FROM cycles WHERE d>=? ORDER BY id",
        conn, params=(d0,)
    )
else:
    df_notes = pd.read_sql_query(
        "SELECT id, d, start_at, title, note FROM cycles ORDER BY id", conn
    )

def extract_links(text: str):
    if not text: return []
    return re.findall(r"\[\[([^\]]+)\]\]", text)

def extract_tags(text: str):
    if not text: return []
    return re.findall(r"#([A-Za-z0-9_\-ぁ-んァ-ヶ一-龠ー]+)", text)

# タイトル辞書
titles = {}
for _, r in df_notes.iterrows():
    t = (r["title"] or "").strip()
    if not t:
        t = _make_title_fallback(r["note"] or "")
    titles[r["id"]] = t

# 逆引き
title_to_id = {}
for nid, name in titles.items():
    title_to_id.setdefault(name, nid)

# PyVisネットワーク
net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#111")
net.set_options("""
var options = {
  "physics": {
    "barnesHut": { "gravitationalConstant": -4000, "centralGravity": 0.3, "springLength": 120, "springConstant": 0.04 },
    "minVelocity": 0.75
  },
  "nodes": { "shape": "dot", "scaling": { "min": 5, "max": 30 } },
  "edges": { "smooth": { "type": "dynamic" } }
}
""")

# ノートノード & タグ
for _, r in df_notes.iterrows():
    nid = int(r["id"])
    title = titles[nid]
    note = r["note"] or ""
    tags = extract_tags(note)
    net.add_node(nid, label=title, title=r["start_at"], size=18)
    for tg in tags:
        tag_label = "#"+tg
        if not any(n['id']==tag_label for n in net.nodes):
            net.add_node(tag_label, label=tag_label, color="#A3A3A3", size=10)
        net.add_edge(nid, tag_label, color="#BBBBBB", width=1)

# [[リンク]] -> エッジ
ghost_targets = set()
for _, r in df_notes.iterrows():
    src = int(r["id"])
    links = extract_links(r["note"] or "")
    for name in links:
        name = name.strip()
        if not name:
            continue
        if name in title_to_id:
            dst = int(title_to_id[name])
            if src != dst:
                net.add_edge(src, dst)
        else:
            ghost_targets.add(name)

for name in ghost_targets:
    if not any(n['id']==name for n in net.nodes):
        net.add_node(name, label=name, color="#DDDDDD", size=12)

# HTMLを書き出して埋め込み（テンポラリ）


# Jinja2がないとpyvisが描画できないのでチェック（保険）
try:
    import jinja2  # noqa: F401
except ImportError:
    st.error("pyvisの表示には Jinja2 が必要です。`pip install jinja2 markupsafe` を実行してください。")
else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        # show() ではなく write_html() を使う
        net.write_html(tmp.name, notebook=False)  # open_browser=False相当
        html = pathlib.Path(tmp.name).read_text(encoding="utf-8")
    components.html(html, height=620, scrolling=True)


# バックリンクビュー
st.markdown("### 🔁 バックリンク / 内容")
if len(df_notes):
    id_list = df_notes["id"].tolist()
    label_list = [titles[int(i)] for i in id_list]
    sel_idx = st.selectbox("ノートを選択", range(len(id_list)),
                           format_func=lambda i: f"{label_list[i]}", index=0)
    sel_id = int(id_list[sel_idx])
    row = df_notes[df_notes["id"]==sel_id].iloc[0]
    st.write(f"**{titles[sel_id]}**  —  {row['start_at']}")
    if row["note"]:
        st.markdown(f"> {row['note'].replace(chr(10), '<br>')}", unsafe_allow_html=True)
    else:
        st.caption("(ノートなし)")

    # backlinks
    target_name = titles[sel_id]
    backlinks = []
    for _, r in df_notes.iterrows():
        if r["id"] == sel_id: 
            continue
        links = extract_links(r["note"] or "")
        if any(link.strip() == target_name for link in links):
            backlinks.append((r["id"], titles[int(r["id"])], r["start_at"]))
    if backlinks:
        st.write("**Backlinks**")
        for (bid, btitle, btime) in backlinks:
            st.markdown(f"- **{btitle}**  ({btime})")
    else:
        st.caption("Backlinks はありません。")

# 画面下にヘルプ
st.markdown("""
---
**ヒント**
- ノート本文に `[[別ノートのタイトル]]` と書くとリンク、 `#tag` はタグノードになります。  
- 週次集計は開始日を変えると7日間のバーが更新されます。  
- サイドバーからCSVとMarkdownをエクスポートできます。
""")
