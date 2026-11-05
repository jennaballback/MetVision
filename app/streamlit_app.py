"""MetVision front end: describe, pick, or upload -> the ten most similar artworks.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

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
EXAMPLES = ["Japanese prints of birds", "woman reading by a window", "Egyptian cat sculpture",
            "stormy sea at night", "misty mountains"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption, button, input { font-family: 'Inter', sans-serif; }
h1, h2, h3, .mv-serif { font-family: 'Cormorant Garamond', Georgia, serif !important; font-weight: 600; }
.block-container { padding-top: 4.5rem; max-width: 1400px; }
[data-testid="stHeader"] { background: transparent; }

.mv-hero { border-bottom: 1px solid #DDD5C8; padding-bottom: 1.1rem; margin-bottom: 1.2rem; }
.mv-wordmark { font-family: 'Cormorant Garamond', Georgia, serif !important; font-size: 3.6rem !important; line-height: 1; font-weight: 600;
               letter-spacing: .01em; margin: 0; color: #1F1B16; }
.mv-wordmark span { color: #8C2F39; }
.mv-tagline { color: #5E554A; font-size: 1.05rem; margin: .45rem 0 0; }
.mv-stats { color: #8A8073; font-size: .8rem; margin-top: .35rem; letter-spacing: .02em; }

/* gallery cards */
.st-key-grid [data-testid="stImage"] img, .st-key-query [data-testid="stImage"] img {
    height: 230px; width: 100%; object-fit: contain; background: #EFEAE1; border-radius: 6px; }
.st-key-query [data-testid="stImage"] img { height: 340px; }
.mv-title { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 1.15rem; font-weight: 600;
            line-height: 1.2; margin: .45rem 0 .15rem; color: #1F1B16;
            display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
            min-height: 2.4em; }
.mv-meta { color: #6F665B; font-size: .78rem; line-height: 1.35; height: 2.7em; overflow: hidden;
           display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.mv-chips { height: 1.9rem; }
.mv-chip { display: inline-block; font-size: .68rem; letter-spacing: .06em; text-transform: uppercase;
           color: #8C2F39; border: 1px solid #D9C3C0; border-radius: 999px; padding: .05rem .5rem;
           margin: .25rem 0 .35rem; }
.mv-rank { color: #A39888; font-size: .72rem; }
.mv-label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: #8A8073; }
.mv-detail dt { font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: #8A8073; margin-top: .6rem; }
.mv-detail dd { margin: 0; color: #1F1B16; }
[data-testid="stDialog"] img, div[role="dialog"] img { max-height: 65vh !important; width: 100%; object-fit: contain;
    background: #EFEAE1; border-radius: 6px; }
.mv-footer { color: #8A8073; font-size: .75rem; border-top: 1px solid #DDD5C8; margin-top: 2.5rem; padding-top: .8rem; }
</style>
"""


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


@st.cache_data(ttl=600)
def local_paths(ids: tuple[int, ...]) -> dict[int, str]:
    rows = conn().execute("SELECT object_id, local_path FROM object_images WHERE object_id = ANY(%s) "
                          "AND status = 'ok'", (list(ids),))
    return {r[0]: r[1] for r in rows if r[1] and Path(r[1]).exists()}


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


def image_src(r: dict) -> str:
    return local_paths((r["object_id"],)).get(r["object_id"]) or r["image_small_url"]


# ---------------------------------------------------------------- actions
def pick(object_id: int):
    """Switch to the similarity view for this artwork."""
    st.session_state["picked"] = object_id
    st.session_state["mode"] = MODES[1]


def set_query(text: str):
    st.session_state["query"] = text


@st.dialog("Artwork", width="large")
def detail(object_id: int):
    a = artwork(object_id)
    if not a:
        st.write("This artwork is no longer in the collection.")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(image_src(a), use_container_width=True)
    with right:
        st.markdown(f"<h2 style='margin-top:0'>{a['title'] or 'Untitled'}</h2>", unsafe_allow_html=True)
        fields = [("Artist", a["artist"]), ("Date", a["object_date"]), ("Culture", a["culture"]),
                  ("Period", a["period"]), ("Medium", a["medium"]), ("Classification", a["classification"]),
                  ("Department", a["department"]), ("Tags", ", ".join(a["tags"]))]
        st.markdown("<dl class='mv-detail'>" + "".join(
            f"<dt>{k}</dt><dd>{v}</dd>" for k, v in fields if v) + "</dl>", unsafe_allow_html=True)
        st.write("")
        if a["object_url"]:
            st.link_button("View on metmuseum.org ↗", a["object_url"], use_container_width=True)
        if st.button("Find similar artworks", type="primary", use_container_width=True,
                     on_click=pick, args=(object_id,)):
            st.rerun()


def card(r: dict, key: str):
    st.image(image_src(r), use_container_width=True)
    st.markdown(f"<div class='mv-title' title=\"{r['title'] or ''}\">{r['title'] or 'Untitled'}</div>",
                unsafe_allow_html=True)
    meta = " · ".join(m for m in [r.get("artist"), r.get("culture"), r.get("object_date")] if m)
    st.markdown(f"<div class='mv-meta'>{meta or '&nbsp;'}</div>"
                + "<div class='mv-chips'>"
                + (f"<span class='mv-chip'>{r['classification']}</span>" if r.get("classification") else "")
                + "</div>",
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.button("Similar", key=f"{key}-sim-{r['object_id']}", on_click=pick, args=(r["object_id"],),
              use_container_width=True)
    if b2.button("Details", key=f"{key}-det-{r['object_id']}", use_container_width=True):
        detail(r["object_id"])


def gallery(results: list[dict], key: str, columns: int = 5):
    if not results:
        st.info("No matches. Try loosening the filters in the sidebar.")
        return
    with st.container(key="grid"):
        for start in range(0, len(results), columns):
            cols = st.columns(columns, gap="medium")
            for i, r in enumerate(results[start:start + columns]):
                with cols[i]:
                    card(r, key)


def sidebar_filters(departments: list[str]) -> Filters:
    with st.sidebar:
        st.markdown("<h3 style='margin-bottom:0'>Filters</h3>", unsafe_allow_html=True)
        st.caption("Applied to every search.")
        f = Filters(
            departments=st.multiselect("Department", departments),
            cultures=[c.strip() for c in st.text_input("Culture", placeholder="e.g. Japan, French").split(",")
                      if c.strip()],
            classifications=[c.strip() for c in st.text_input("Classification", placeholder="e.g. Prints")
                             .split(",") if c.strip()],
        )
        if st.toggle("Limit by date"):
            y0, y1 = st.slider("Years (negative = BCE)", -3000, 2000, (1600, 1900), step=50)
            f.year_from, f.year_to = y0, y1
    return f


def why_similar(a_id: int, b_id: int) -> str:
    """Optional: Claude looks at both images and writes one sentence."""
    import anthropic

    paths = local_paths((a_id, b_id))

    def img_block(oid):
        data = Path(paths[oid]).read_bytes()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(data).decode()}}

    resp = anthropic.Anthropic().messages.create(
        model=settings.anthropic_model, max_tokens=2000, output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            img_block(a_id), img_block(b_id),
            {"type": "text", "text": "In one sentence, say what visually links these two artworks "
                                     "(composition, palette, subject, technique)."}]}])
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- page
st.markdown(CSS, unsafe_allow_html=True)
departments, (n_objects, n_embedded) = facets()
st.markdown(
    f"""<div class="mv-hero">
      <div class="mv-wordmark">Met<span>Vision</span></div>
      <div class="mv-tagline">Find artworks that look alike across the Metropolitan Museum's open-access collection.</div>
      <div class="mv-stats">{n_embedded:,} artworks searchable · ranked by CLIP image embeddings in pgvector ·
      Claude only turns sentences into filters</div>
    </div>""", unsafe_allow_html=True)
filters = sidebar_filters(departments)

st.session_state.setdefault("mode", MODES[0])
mode = st.segmented_control("Search by", MODES, key="mode", label_visibility="collapsed") or MODES[0]

if mode == MODES[0]:
    q = st.text_input("Describe what you're looking for", key="query",
                      placeholder="e.g. Japanese woodblock prints of birds from the 1800s")
    chips = st.columns(len(EXAMPLES))
    for c, ex in zip(chips, EXAMPLES):
        c.button(ex, key=f"ex-{ex}", on_click=set_query, args=(ex,), use_container_width=True)
    use_claude = st.toggle("Let Claude turn the sentence into filters", value=True,
                           help="Claude reads the sentence and fills in culture, classification and dates. "
                                "The ranking itself is pure vector search.")
    if q:
        text, f = q, filters
        if use_claude:
            parsed = parse_query(q, departments, model=settings.anthropic_model)
            if parsed.fallback:
                st.caption("Claude isn't available (no API key set), so this searches the raw text.")
            else:
                text = parsed.free_text
                f = Filters(**{**parsed.filters.__dict__, **{k: v for k, v in filters.__dict__.items() if v}})
                applied = [f"{k.replace('_', ' ')}: {v}" for k, v in parsed.raw.items()
                           if v not in (None, "", []) and k != "free_text"]
                if applied:
                    st.caption("Filters from Claude → " + " · ".join(applied)
                               + (f" · looking for “{text}”" if text else ""))
        st.write("")
        gallery(search_by_text(conn(), text, embedder(), k=10, filters=f), "text")

elif mode == MODES[1]:
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    title_q = c1.text_input("Find an artwork by title", placeholder="Type part of a title")
    if c2.button("Surprise me", use_container_width=True):
        st.session_state["picked"] = conn().execute(
            "SELECT object_id FROM object_embeddings ORDER BY random() LIMIT 1").fetchone()[0]
    if title_q:
        opts = conn().execute("SELECT o.object_id, o.title FROM objects o JOIN object_embeddings USING (object_id) "
                              "WHERE o.title ILIKE %s ORDER BY o.title LIMIT 50", (f"%{title_q}%",)).fetchall()
        if opts:
            choice = st.selectbox("Matches", opts, format_func=lambda r: f"{r[1]} ({r[0]})")
            st.button("Use this artwork", on_click=pick, args=(choice[0],))
        else:
            st.caption("No titles match.")
    picked = st.session_state.get("picked")
    if not picked:
        st.info("Pick an artwork by title, press Surprise me, or press Similar on any result.")
    else:
        a = artwork(picked)
        sim_mode = st.radio("Similarity", ["Visual", "Visual + description"], horizontal=True,
                            help="Visual: image embeddings only. Visual + description: also fuses "
                                 "title, medium and tags, ranked together.")
        results = similar_to_object(conn(), picked, k=10, filters=filters,
                                    mode="image" if sim_mode == "Visual" else "hybrid")
        left, right = st.columns([1, 4], gap="large")
        with left, st.container(key="query"):
            st.markdown("<div class='mv-label'>Selected</div>", unsafe_allow_html=True)
            st.image(image_src(a), use_container_width=True)
            st.markdown(f"<div class='mv-title'>{a['title'] or 'Untitled'}</div>"
                        f"<div class='mv-meta'>{' · '.join(m for m in [a['culture'], a['object_date']] if m)}</div>",
                        unsafe_allow_html=True)
            if st.button("Details", key="query-details", use_container_width=True):
                detail(picked)
            if results and st.button("Why is the top match similar?", use_container_width=True,
                                     help="Claude looks at both images (needs an API key)."):
                try:
                    st.info(why_similar(picked, results[0]["object_id"]))
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Claude isn't available: {exc}")
        with right:
            st.markdown("<div class='mv-label'>Most similar</div>", unsafe_allow_html=True)
            gallery(results, "pick", columns=4)

else:
    up = st.file_uploader("Upload a photo of anything", type=["jpg", "jpeg", "png", "webp"])
    if up:
        img = Image.open(up).convert("RGB")
        left, right = st.columns([1, 4], gap="large")
        with left, st.container(key="query"):
            st.markdown("<div class='mv-label'>Your photo</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with right:
            st.markdown("<div class='mv-label'>Most similar</div>", unsafe_allow_html=True)
            vec = embedder().embed_images([img])[0]
            gallery(search_by_image(conn(), vec, k=10, filters=filters), "upload", columns=4)

st.markdown(
    "<div class='mv-footer'>Images and data: The Metropolitan Museum of Art Open Access (CC0). "
    "MetVision is an independent portfolio project, not affiliated with the museum.</div>",
    unsafe_allow_html=True)
