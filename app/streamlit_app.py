"""MetVision front end: describe, pick, or upload -> the ten most similar artworks,
hung salon-style on a gallery wall.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import base64
import html
import io
import sys
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metsearch import db  # noqa: E402
from metsearch.config import get_settings  # noqa: E402
from metsearch.embed import get_embedder  # noqa: E402
from metsearch.nl_filter import parse_query  # noqa: E402
from metsearch.search import Filters, search_by_image, search_by_text, similar_to_object  # noqa: E402

st.set_page_config(page_title="MetVision", page_icon="🖼️", layout="wide")
settings = get_settings()

MODES = ["Describe", "Similar to an artwork", "Upload a photo"]
EXAMPLES = ["Japanese prints of birds", "woman reading by a window", "stormy sea at night", "misty mountains"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption, button, input { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif !important; font-weight: 600; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebarCollapsedControl"] { display: none; }

/* ---------- the whole page is the gallery wall ---------- */
.stApp {
  background:
    radial-gradient(ellipse 26% 520px at 15% 430px, rgba(255,255,255,.55), transparent 75%),
    radial-gradient(ellipse 26% 520px at 50% 430px, rgba(255,255,255,.55), transparent 75%),
    radial-gradient(ellipse 26% 520px at 85% 430px, rgba(255,255,255,.55), transparent 75%),
    linear-gradient(180deg, #DEDEE0 0%, #D7D7D9 50%, #CACACE 100%);
}
.block-container { padding-top: 0; padding-bottom: 0; max-width: 1480px; }

/* hero: a full-bleed Deep Water banner that holds the search itself */
.st-key-hero { position: relative; padding: 1.6rem 0 2.4rem; margin-bottom: .6rem; }
.st-key-hero::before { content: ""; position: absolute; z-index: 0; top: -40px; bottom: 0; left: calc(50% - 50vw); width: 100vw;
  background:
    radial-gradient(ellipse 50% 110% at 50% -10%, rgba(255,255,255,.22), transparent 70%),
    linear-gradient(160deg, #919E86 0%, #829076 50%, #75836A 100%);
  box-shadow: 0 6px 14px rgba(47,64,61,.18); }
.st-key-hero > * { position: relative; z-index: 1; }
.mv-hero { text-align: center; padding-bottom: 1.2rem; }
.mv-kicker { font-size: .7rem; letter-spacing: .34em; text-transform: uppercase; color: #2F403D; }
.mv-wordmark { font-family: 'Cormorant Garamond', Georgia, serif !important; font-size: 5.4rem !important;
               font-weight: 600; line-height: 1; color: #2F403D; margin: .35rem 0 .25rem;
               text-shadow: 0 1px 0 rgba(255,255,255,.25); }
.mv-wordmark em { color: #425854; text-shadow: none; font-style: italic; font-weight: 500; }
.mv-tagline { font-family: 'Cormorant Garamond', Georgia, serif; font-style: italic; font-size: 1.35rem; color: #2F403D; }
.mv-stats { color: #425854; font-size: .68rem; letter-spacing: .1em; margin-top: .5rem; }
/* widgets on the banner: light text, no boxes */
.st-key-hero p, .st-key-hero label, .st-key-hero [data-testid="stWidgetLabel"] p,
.st-key-hero [data-testid="stMarkdownContainer"] p { color: #2F403D !important; }
.st-key-hero [data-testid="stTextInputRootElement"], .st-key-hero [data-baseweb="input"],
.st-key-hero [data-baseweb="base-input"] { background: transparent !important; border: none !important;
  border-radius: 0 !important; }
.st-key-hero [data-testid="stTextInputRootElement"] { border-bottom: 1.5px solid #425854 !important; }
.st-key-hero input { background: transparent !important; color: #1F2D2A !important; text-align: center;
  font-family: 'Cormorant Garamond', Georgia, serif !important; font-size: 1.6rem !important; font-style: italic;
  caret-color: #425854; padding: .4rem 0 !important; }
.st-key-hero input::placeholder { color: rgba(47,64,61,.6) !important; }
.st-key-hero [data-testid="stButtonGroup"] { justify-content: center; }
.st-key-hero button[data-variant] { background: transparent !important; color: #2F403D !important;
  border: 1px solid rgba(47,64,61,.4) !important; box-shadow: none !important; }
.st-key-hero button[data-variant] p { color: inherit !important; }
.st-key-hero button[data-variant="segmented_control"] { border: none !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important; }
.st-key-hero button[data-variant="segmented_control"] p { letter-spacing: .16em; text-transform: uppercase; font-size: .74rem; }
.st-key-hero button[data-variant="segmented_control"][aria-checked="true"] { border-bottom-color: #425854 !important;
  color: #1F2D2A !important; font-weight: 600; }
.st-key-hero button[data-variant="pills"] { border-radius: 999px !important; }
.st-key-hero button[data-variant]:hover { background: rgba(255,255,255,.22) !important; }
.st-key-hero [data-testid="stButtonGroup"] div:has(> button) { justify-content: center; }
.st-key-hero [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]),
.st-key-hero [data-testid="stElementContainer"]:has([data-testid="stCheckbox"]),
.st-key-hero [data-testid="stElementContainer"]:has([data-testid="stRadio"]) { align-self: center; }
.st-key-hero [data-testid="stPopoverButton"] { background: transparent !important; border: none !important;
  box-shadow: none !important; }
.st-key-hero [data-testid="stPopoverButton"] p { letter-spacing: .16em; text-transform: uppercase; font-size: .74rem;
  color: #2F403D !important; }
.st-key-hero [data-testid^="stBaseButton-secondary"] { background: rgba(255,255,255,.18) !important;
  border: 1px solid rgba(47,64,61,.45) !important; color: #2F403D !important; }
.st-key-hero [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.2); border: 1px dashed rgba(47,64,61,.5); }
.st-key-hero [data-testid="stFileUploaderDropzone"] span, .st-key-hero small { color: #2F403D !important; }
.st-key-hero [data-testid="stCheckbox"], .st-key-hero [data-testid="stRadio"] > div { display: flex; justify-content: center; }
.st-key-hero [data-testid="stElementContainer"]:has([data-testid="stCheckbox"]) { display: flex; justify-content: center; }
.mv-note { color: #425854; font-size: .8rem; text-align: center; margin-top: .3rem; }
[data-testid="stHeader"] button, [data-testid="stHeader"] span, [data-testid="stToolbar"] * { color: #2F403D !important; }

/* exhibition room title */
.mv-roomtitle { text-align: center; margin: 2.6rem 0 .4rem; font-family: 'Cormorant Garamond', Georgia, serif;
                font-size: 1rem; letter-spacing: .42em; text-transform: uppercase; color: #425854; }
.mv-roomtitle::after { content: ""; display: block; width: 46px; height: 1px; background: #C9A44A; margin: .7rem auto 0; }

/* salon hang: rows of differently sized frames, centred like a hung wall */
.mv-hang { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 3rem 3.4rem;
           padding: 2.2rem 0 1rem; }
.mv-piece { display: flex; flex-direction: column; align-items: center; width: var(--w, 260px); max-width: 42vw; }
@media (max-width: 760px) { .mv-hang { gap: 2rem 1.2rem; } .mv-wordmark { font-size: 3.2rem !important; } }

/* ---------- ornate gilt frames ---------- */
.mv-frame { --fw: 24px; --slip: 10px; position: relative; display: block; width: 100%; padding: var(--fw);
  background:
    radial-gradient(circle at 13px 13px, #FFF0B8 0 3px, #8A6420 4px 6px, transparent 7px),
    radial-gradient(circle at calc(100% - 13px) 13px, #FFF0B8 0 3px, #8A6420 4px 6px, transparent 7px),
    radial-gradient(circle at 13px calc(100% - 13px), #FFF0B8 0 3px, #8A6420 4px 6px, transparent 7px),
    radial-gradient(circle at calc(100% - 13px) calc(100% - 13px), #FFF0B8 0 3px, #8A6420 4px 6px, transparent 7px),
    repeating-linear-gradient(45deg, rgba(255,244,200,.20) 0 2px, transparent 2px 5px),
    repeating-linear-gradient(-45deg, rgba(90,60,10,.14) 0 1px, transparent 1px 6px),
    linear-gradient(135deg, #7A5A1C 0%, #E8C96A 17%, #B08530 33%, #F7E09A 50%, #A57A27 67%, #E6C567 83%, #6C4E15 100%);
  box-shadow:
    inset 0 0 0 1px #4A3510, inset 0 0 0 4px #DDBE62, inset 0 0 0 5px #6B4D16,
    inset 0 0 calc(var(--fw) * .7) rgba(70,45,5,.5),
    0 18px 26px rgba(45,38,30,.32), 0 4px 7px rgba(45,38,30,.22);
  transition: transform .2s ease, box-shadow .2s ease; }
.mv-frame:hover { transform: translateY(-3px);
  box-shadow: inset 0 0 0 1px #4A3510, inset 0 0 0 4px #DDBE62, inset 0 0 0 5px #6B4D16,
    inset 0 0 calc(var(--fw) * .7) rgba(70,45,5,.5), 0 26px 36px rgba(45,38,30,.36), 0 6px 10px rgba(45,38,30,.22); }
/* bead-and-reel moulding just inside the outer edge */
.mv-frame::before { content: ""; position: absolute; inset: 7px; border: 3px dotted rgba(255,238,178,.85);
                    pointer-events: none; }
/* bevelled inner lip where the frame meets the liner */
.mv-frame::after { content: ""; position: absolute; inset: calc(var(--fw) - 4px); border: 4px solid;
                   border-color: #5E4312 #F4DC8E #F4DC8E #5E4312; pointer-events: none; }
/* flat gilt liner (slip) around the picture */
.mv-mat { display: block; padding: var(--slip);
          background: linear-gradient(135deg, #D2B464, #EEDA9C 45%, #C6A452);
          box-shadow: inset 1px 1px 3px rgba(80,55,10,.6); }
.mv-frame img { display: block; width: 100%; height: auto; object-fit: cover; max-width: none; border: 1px solid rgba(50,35,5,.55); }
/* little brass title plaque on the bottom rail */
.mv-plaque { position: absolute; left: 50%; bottom: calc(var(--fw) / 2 - 5px); transform: translateX(-50%);
             width: 38px; height: 10px; border-radius: 1px; z-index: 1;
             background: linear-gradient(180deg, #F3E2A6, #B99446); box-shadow: 0 1px 2px rgba(0,0,0,.4); }
/* variations, so the wall doesn't look like one frame repeated */
.f-wide  { --fw: 32px; --slip: 14px; }
.f-slim  { --fw: 16px; --slip: 0px; }
.f-slim::before { inset: 4px; border-width: 2px; }
.f-antique { background:
    radial-gradient(circle at 13px 13px, #F1DC98 0 3px, #5E4312 4px 6px, transparent 7px),
    radial-gradient(circle at calc(100% - 13px) 13px, #F1DC98 0 3px, #5E4312 4px 6px, transparent 7px),
    radial-gradient(circle at 13px calc(100% - 13px), #F1DC98 0 3px, #5E4312 4px 6px, transparent 7px),
    radial-gradient(circle at calc(100% - 13px) calc(100% - 13px), #F1DC98 0 3px, #5E4312 4px 6px, transparent 7px),
    repeating-linear-gradient(45deg, rgba(255,240,190,.16) 0 2px, transparent 2px 5px),
    linear-gradient(135deg, #5C4214 0%, #C9A44E 20%, #8A6A28 40%, #DCC07A 55%, #7E5E20 75%, #B8923F 90%, #4E370E 100%); }
.f-selected { }

/* museum wall label, printed straight on the wall */
.mv-label { margin-top: 1.05rem; text-align: center; max-width: 100%; }
.mv-lt { font-family: 'Cormorant Garamond', Georgia, serif; font-style: italic; font-size: 1.04rem; line-height: 1.15;
         color: #2F403D; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.mv-lm { font-size: .68rem; color: #5C6B60; margin-top: .2rem; line-height: 1.4; }
.mv-la { margin-top: .35rem; font-size: .6rem; letter-spacing: .16em; text-transform: uppercase; }
.mv-la a { color: #553C4B !important; text-decoration: none; margin: 0 .5rem; }
.mv-la a:hover { text-decoration: underline; }
.mv-eyebrow { font-size: .6rem; letter-spacing: .22em; text-transform: uppercase; color: #553C4B; margin-bottom: .15rem; }

/* featured (selected / uploaded) piece gets its own spotlight */
.mv-featured { display: flex; justify-content: center; padding: 1.4rem 0 .6rem;
               background: radial-gradient(ellipse 380px 300px at 50% 45%, rgba(255,255,255,.75), transparent 70%); }
.mv-featured .mv-piece { --w: 400px; }
.mv-empty { text-align: center; color: #425854; font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 1.4rem; font-style: italic; padding: 3rem 0 2rem; }

/* skirting board and floor across the bottom of the page */
.mv-baseboard { width: 100vw; margin: 4rem 0 0 calc(50% - 50vw); height: 22px;
                background: linear-gradient(#EEEEF0, #C4C4C9); border-top: 1px solid #B5B5BA; }
.mv-floor { width: 100vw; margin-left: calc(50% - 50vw); height: 90px;
  background: linear-gradient(180deg, rgba(0,0,0,.18), rgba(255,255,255,.06) 40%, rgba(0,0,0,.2)),
    repeating-linear-gradient(90deg, #9A6A3E 0 140px, #7E522B 140px 142px, #A8764A 142px 290px, #7E522B 290px 292px); }
.mv-footer { color: #E9DCC8; font-size: .72rem; text-align: center; margin-top: -58px; padding-bottom: 30px;
             position: relative; }

/* detail dialog */
.mv-detail-img { width: 100%; height: 62vh; background: #E6E6E8 center / contain no-repeat; }
.mv-detail dt { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: #829076; margin-top: .6rem; }
.mv-detail dd { margin: 0; color: #2F403D; }
</style>
"""

FRAMES = ["", "f-wide", "f-antique", "f-slim", "f-wide f-antique", "", "f-slim f-antique"]
WIDTHS = [210, 260, 180, 240, 290, 200, 250]   # px: the salon hang's varied sizes


# ---------------------------------------------------------------- data access
@st.cache_resource
def conn():
    c = db.connect(settings.database_url, vector=False)
    c.autocommit = True
    from pgvector.psycopg import register_vector
    register_vector(c)
    return c


@st.cache_resource
def embedder():
    return get_embedder(settings.embedder, settings.clip_model, settings.clip_pretrained)


@st.cache_data(ttl=600)
def facets():
    c = conn()
    depts = [r[0] for r in c.execute(
        "SELECT DISTINCT d.name FROM objects o JOIN departments d USING (department_id) ORDER BY 1")]
    stats = c.execute("SELECT count(*), (SELECT count(*) FROM object_embeddings) FROM objects").fetchone()
    return depts, stats


@st.cache_data(ttl=3600, show_spinner=False)
def image_data_uri(object_id: int, fallback_url: str) -> str:
    """Inline the cached JPEG (works offline and avoids hot-linking); else the Met's URL."""
    row = conn().execute("SELECT local_path FROM object_images WHERE object_id = %s AND status = 'ok'",
                         (object_id,)).fetchone()
    if row and row[0] and Path(row[0]).exists():
        return "data:image/jpeg;base64," + base64.b64encode(Path(row[0]).read_bytes()).decode()
    return fallback_url


@st.cache_data(ttl=600)
def artwork(object_id: int) -> dict | None:
    cur = conn().execute(
        """SELECT o.object_id, o.title, o.artist, d.name AS department, o.object_date, o.culture, o.period,
                  o.medium, o.classification, o.image_small_url, o.object_url,
                  coalesce(array_agg(t.term ORDER BY t.term) FILTER (WHERE t.term IS NOT NULL), '{}') AS tags
           FROM objects o LEFT JOIN departments d USING (department_id)
           LEFT JOIN object_tags ot USING (object_id) LEFT JOIN tags t USING (tag_id)
           WHERE o.object_id = %s GROUP BY o.object_id, d.name""", (object_id,))
    row = cur.fetchone()
    return dict(zip([c.name for c in cur.description], row)) if row else None


@st.cache_data(ttl=3600)
def todays_hang(n: int = 12) -> list[dict]:
    """A different random wall each day, for the landing page."""
    from datetime import date

    from metsearch.search import fetch_rows
    ids = [r[0] for r in conn().execute(
        "SELECT object_id FROM object_embeddings ORDER BY md5(object_id::text || %s) LIMIT %s",
        (date.today().isoformat(), n))]
    return fetch_rows(conn(), [(i, 0.0) for i in ids])


# ---------------------------------------------------------------- routing
# Frames on the wall are plain HTML links, so clicks travel through the URL:
#   ?m=<mode index>&q=<text>&pick=<object id>&detail=<object id>
def href(**changes) -> str:
    state = {"m": MODES.index(st.session_state.get("mode") or MODES[0]),
             "q": st.session_state.get("query", ""), "pick": st.session_state.get("picked")}
    state.update(changes)
    return "?" + urlencode({k: v for k, v in state.items() if v not in (None, "")})


def apply_url():
    """Copy URL parameters into session state once per distinct URL; return a detail id to open."""
    qp = st.query_params
    sig = urlencode(sorted(qp.to_dict().items()))
    if st.session_state.get("_applied") == sig:
        return None
    st.session_state["_applied"] = sig
    if qp.get("m", "").isdigit() and int(qp["m"]) < len(MODES):
        st.session_state["mode"] = MODES[int(qp["m"])]
    if "q" in qp:
        st.session_state["query"] = qp["q"]
    if qp.get("pick", "").isdigit():
        st.session_state["picked"] = int(qp["pick"])
    return int(qp["detail"]) if qp.get("detail", "").isdigit() else None


def pick(object_id: int):
    st.session_state["picked"] = object_id
    st.session_state["mode"] = MODES[1]


def use_example():
    if st.session_state.get("example"):
        st.session_state["query"] = st.session_state["example"]
    st.session_state["example"] = None


# ---------------------------------------------------------------- rendering
def esc(v) -> str:
    return html.escape(str(v)) if v not in (None, "") else ""


def framed(src: str, frame: str, link: str | None, extra: str = "") -> str:
    tag, attrs = ("a", f"href='{link}' target='_self'") if link else ("span", "")
    return (f"<{tag} {attrs} class='mv-frame {frame} {extra}'>"
            f"<span class='mv-mat'><img src='{src}' alt='' loading='lazy'></span>"
            f"<span class='mv-plaque'></span></{tag}>")


def wall_label(r: dict, actions: bool = True, eyebrow: str = "") -> str:
    meta = "<br>".join(x for x in [esc(r.get("artist")),
                                   ", ".join(esc(m) for m in [r.get("culture"), r.get("object_date")] if m),
                                   esc(r.get("classification"))] if x)
    acts = ""
    if actions:
        acts = (f"<div class='mv-la'><a href='{href(m=1, pick=r['object_id'])}' target='_self'>Similar</a>"
                f"<a href='{href(detail=r['object_id'])}' target='_self'>Details</a></div>")
    eb = f"<div class='mv-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    return (f"<div class='mv-label'>{eb}<div class='mv-lt'>{esc(r.get('title')) or 'Untitled'}</div>"
            f"<div class='mv-lm'>{meta}</div>{acts}</div>")


def gallery_wall(results: list[dict], room: str, featured_html: str = "", empty: str = ""):
    """Hang results straight on the page: the whole page is the wall."""
    pieces = []
    for r in results:
        oid = r["object_id"]
        frame, width = FRAMES[oid % len(FRAMES)], WIDTHS[(oid // 7) % len(WIDTHS)]
        src = image_data_uri(oid, r["image_small_url"])
        pieces.append(f"<div class='mv-piece' style='--w:{width}px'>"
                      f"{framed(src, frame, href(detail=oid))}{wall_label(r)}</div>")
    body = (f"<div class='mv-hang'>{''.join(pieces)}</div>" if pieces
            else f"<div class='mv-empty'>{empty or 'Nothing on this wall yet.'}</div>")
    st.markdown(f"{featured_html}<div class='mv-roomtitle'>{esc(room)}</div>{body}", unsafe_allow_html=True)


def featured(src: str, label_html: str) -> str:
    return (f"<div class='mv-featured'><div class='mv-piece'>"
            f"{framed(src, 'f-wide', None, 'f-selected')}{label_html}</div></div>")


@st.dialog("On view", width="large")
def detail(object_id: int):
    a = artwork(object_id)
    if not a:
        st.write("This artwork is no longer in the collection.")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"<div class='mv-detail-img' style=\"background-image:url('{image_data_uri(object_id, a['image_small_url'])}')\"></div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown(f"<h2 style='margin-top:0;font-style:italic'>{esc(a['title']) or 'Untitled'}</h2>",
                    unsafe_allow_html=True)
        fields = [("Artist", a["artist"]), ("Date", a["object_date"]), ("Culture", a["culture"]),
                  ("Period", a["period"]), ("Medium", a["medium"]), ("Classification", a["classification"]),
                  ("Department", a["department"]), ("Tags", ", ".join(a["tags"]))]
        st.markdown("<dl class='mv-detail'>" + "".join(
            f"<dt>{k}</dt><dd>{esc(v)}</dd>" for k, v in fields if v) + "</dl>", unsafe_allow_html=True)
        st.write("")
        if a["object_url"]:
            st.link_button("View on metmuseum.org ↗", a["object_url"], width="stretch")
        if st.button("Find similar artworks", type="primary", width="stretch",
                     on_click=pick, args=(object_id,)):
            st.rerun()


def filter_menu(departments: list[str]) -> Filters:
    """Filters live in a drop-down at the top of the page, so the wall gets the full width."""
    ss = st.session_state
    active = sum(bool(ss.get(k)) for k in ("f_dept", "f_cult", "f_class", "f_date"))
    with st.popover(f"Filters · {active}" if active else "Filters", width="stretch"):
        st.caption("Applied to every search.")
        f = Filters(
            departments=st.multiselect("Department", departments, key="f_dept"),
            cultures=[c.strip() for c in st.text_input("Culture", placeholder="e.g. Japan, French",
                                                       key="f_cult").split(",") if c.strip()],
            classifications=[c.strip() for c in st.text_input("Classification", placeholder="e.g. Prints",
                                                              key="f_class").split(",") if c.strip()],
        )
        if st.toggle("Limit by date", key="f_date"):
            y0, y1 = st.slider("Years (negative = BCE)", -3000, 2000, (1600, 1900), step=50)
            f.year_from, f.year_to = y0, y1
    return f


def why_similar(a_id: int, b_id: int) -> str:
    """Optional: Claude looks at both images and writes one sentence."""
    import anthropic

    def img_block(oid):
        uri = image_data_uri(oid, "")
        if not uri.startswith("data:"):
            raise RuntimeError("image not cached locally")
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": uri.split(",", 1)[1]}}

    resp = anthropic.Anthropic().messages.create(
        model=settings.anthropic_model, max_tokens=2000, output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            img_block(a_id), img_block(b_id),
            {"type": "text", "text": "In one sentence, say what visually links these two artworks "
                                     "(composition, palette, subject, technique)."}]}])
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- page
st.markdown(CSS, unsafe_allow_html=True)
open_detail = apply_url()
departments, (n_objects, n_embedded) = facets()
hero = st.container(key="hero")
hero.markdown(
    f"""<div class="mv-hero">
      <div class="mv-wordmark">Met<em>Vision</em></div>
      <div class="mv-tagline">Describe a picture, choose a work, or bring a photo. We'll hang its closest relatives.</div>
      <div class="mv-stats">{n_embedded:,} WORKS ON VIEW · CLIP EMBEDDINGS · PGVECTOR · CLAUDE-PARSED FILTERS</div>
    </div>""", unsafe_allow_html=True)

st.session_state.setdefault("mode", MODES[0])
_, rail_col, _ = hero.columns([1, 5, 1])
with rail_col:
    _, c_mode, c_filters = st.columns([1, 4, 1], vertical_alignment="center")
    with c_mode:
        mode = st.segmented_control("Search by", MODES, key="mode", label_visibility="collapsed") or MODES[0]
    with c_filters:
        filters = filter_menu(departments)

    if mode == MODES[0]:
        q = st.text_input("Describe what you're looking for", key="query", label_visibility="collapsed",
                          placeholder="Describe a picture you'd like to see…")
        st.pills("Try", EXAMPLES, key="example", on_change=use_example, label_visibility="collapsed")
        use_claude = st.toggle("Let Claude turn the sentence into filters", value=True,
                               help="Claude reads the sentence and fills in culture, classification and "
                                    "dates. The ranking itself is pure vector search.")
    elif mode == MODES[1]:
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        title_q = c1.text_input("Find an artwork by title", placeholder="Type part of a title",
                                label_visibility="collapsed")
        if c2.button("Surprise me", width="stretch"):
            st.session_state["picked"] = conn().execute(
                "SELECT object_id FROM object_embeddings ORDER BY random() LIMIT 1").fetchone()[0]
        if title_q:
            opts = conn().execute("SELECT o.object_id, o.title FROM objects o JOIN object_embeddings USING (object_id) "
                                  "WHERE o.title ILIKE %s ORDER BY o.title LIMIT 50", (f"%{title_q}%",)).fetchall()
            if opts:
                c3, c4 = st.columns([4, 1], vertical_alignment="bottom")
                choice = c3.selectbox("Matches", opts, format_func=lambda r: f"{r[1]} ({r[0]})")
                c4.button("Hang this one", on_click=pick, args=(choice[0],), width="stretch")
            else:
                st.caption("No titles match.")
        sim_mode = st.radio("Similarity", ["Visual", "Visual + description"], horizontal=True,
                            help="Visual: image embeddings only. Visual + description: also fuses "
                                 "title, medium and tags, ranked together.")
    else:
        up = st.file_uploader("Bring a photo of anything", type=["jpg", "jpeg", "png", "webp"])

if mode == MODES[0]:
    if q:
        text, f = q, filters
        note = ""
        if use_claude:
            parsed = parse_query(q, departments, model=settings.anthropic_model)
            if parsed.fallback:
                note = "Claude isn't available (no API key set), so this searches the raw text."
            else:
                text = parsed.free_text
                f = Filters(**{**parsed.filters.__dict__, **{k: v for k, v in filters.__dict__.items() if v}})
                applied = [f"{k.replace('_', ' ')}: {v}" for k, v in parsed.raw.items()
                           if v not in (None, "", []) and k != "free_text"]
                if applied:
                    note = "Filters from Claude → " + " · ".join(applied) + (f" · looking for “{text}”" if text else "")
        if note:
            st.markdown(f"<div class='mv-note'>{esc(note)}</div>", unsafe_allow_html=True)
        gallery_wall(search_by_text(conn(), text, embedder(), k=10, filters=f), f"“{q}”")
    else:
        gallery_wall(todays_hang(), "On view today")

elif mode == MODES[1]:
    picked = st.session_state.get("picked")
    a = artwork(picked) if picked else None
    if not a:
        gallery_wall([], "Gallery", empty="Pick a work by title, press Surprise me, or choose Similar on any label.")
    else:
        results = similar_to_object(conn(), picked, k=10, filters=filters,
                                    mode="image" if sim_mode == "Visual" else "hybrid")
        feat = featured(image_data_uri(picked, a["image_small_url"]), wall_label(a, eyebrow="Selected work"))
        gallery_wall(results, "Kindred works", featured_html=feat)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if results and st.button("Why is the closest match similar?", width="stretch",
                                     help="Claude looks at both images (needs an API key)."):
                try:
                    st.info(why_similar(picked, results[0]["object_id"]))
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Claude isn't available: {exc}")

else:
    if up:
        img = Image.open(up).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((600, 600))
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=85)
        src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        vec = embedder().embed_images([img])[0]
        label = ("<div class='mv-label'><div class='mv-eyebrow'>Your photo</div>"
                 f"<div class='mv-lt'>{esc(up.name)}</div></div>")
        gallery_wall(search_by_image(conn(), vec, k=10, filters=filters), "Kindred works",
                     featured_html=featured(src, label))
    else:
        gallery_wall([], "Gallery", empty="Upload a photo to see which artworks it resembles.")

if open_detail:
    detail(open_detail)

st.markdown(
    "<div class='mv-baseboard'></div><div class='mv-floor'></div>"
    "<div class='mv-footer'>Images and data: The Metropolitan Museum of Art Open Access (CC0). "
    "MetVision is an independent portfolio project, not affiliated with the museum.</div>",
    unsafe_allow_html=True)
