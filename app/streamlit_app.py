"""Streamlit front end: describe, pick, or upload -> ten nearest artworks.

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

st.set_page_config(page_title="Met visual search", layout="wide")
settings = get_settings()


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


def show_grid(results: list[dict], key: str):
    if not results:
        st.info("No matches. Try loosening the filters.")
        return
    paths = local_paths(tuple(r["object_id"] for r in results))
    cols = st.columns(5)
    for i, r in enumerate(results):
        with cols[i % 5]:
            src = paths.get(r["object_id"]) or r["image_small_url"]
            st.image(src, use_container_width=True)
            st.markdown(f"**[{r['title'] or 'Untitled'}]({r['object_url']})**")
            meta = [r.get("artist"), r.get("culture"), r.get("object_date"), r.get("classification"),
                    r.get("department")]
            st.caption(" · ".join(m for m in meta if m))
            if st.button("More like this", key=f"{key}-{r['object_id']}"):
                st.session_state["picked"] = r["object_id"]
                st.toast("Open the 'Pick an artwork' tab to see its neighbours")


def sidebar_filters(departments: list[str]) -> Filters:
    st.sidebar.header("Filters")
    f = Filters(
        departments=st.sidebar.multiselect("Department", departments),
        cultures=[c.strip() for c in st.sidebar.text_input("Culture contains", "").split(",") if c.strip()],
        classifications=[c.strip() for c in st.sidebar.text_input("Classification contains", "").split(",")
                         if c.strip()],
    )
    if st.sidebar.checkbox("Limit by date"):
        y0, y1 = st.sidebar.slider("Years", -3000, 2000, (1600, 1900), step=50)
        f.year_from, f.year_to = y0, y1
    return f


def why_similar(a: dict, b: dict) -> str:
    """Optional: Claude looks at both images and writes one sentence."""
    import anthropic

    paths = local_paths((a["object_id"], b["object_id"]))

    def img_block(r):
        data = Path(paths[r["object_id"]]).read_bytes()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(data).decode()}}

    resp = anthropic.Anthropic().messages.create(
        model=settings.anthropic_model, max_tokens=2000, output_config={"effort": "low"},
        messages=[{"role": "user", "content": [
            img_block(a), img_block(b),
            {"type": "text", "text": "In one sentence, say what visually links these two artworks "
                                     "(composition, palette, subject, technique)."}]}])
    return "".join(b.text for b in resp.content if b.type == "text")


departments, (n_objects, n_embedded) = facets()
st.title("Met Museum visual search")
st.caption(f"{n_embedded:,} of {n_objects:,} open-access artworks embedded with `{embedder().name}`. "
           "Rankings come from CLIP vectors in pgvector; Claude only parses filters.")
filters = sidebar_filters(departments)

tab_text, tab_pick, tab_upload = st.tabs(["Describe it", "Pick an artwork", "Upload a photo"])

with tab_text:
    q = st.text_input("What are you looking for?", placeholder="Japanese woodblock prints of birds from the 1800s")
    use_claude = st.toggle("Let Claude turn the sentence into filters", value=True)
    if q:
        text, f = q, filters
        if use_claude:
            parsed = parse_query(q, departments, model=settings.anthropic_model)
            if parsed.fallback:
                st.warning("Claude unavailable (no API key?); searching the raw text instead.")
            else:
                text = parsed.free_text
                f = Filters(**{**parsed.filters.__dict__,
                               **{k: v for k, v in filters.__dict__.items() if v}})
                with st.expander("Parsed filters"):
                    st.json(parsed.raw)
        show_grid(search_by_text(conn(), text, embedder(), k=10, filters=f), "text")

with tab_pick:
    c1, c2 = st.columns([3, 1])
    title_q = c1.text_input("Find an artwork by title", key="title_q")
    if c2.button("Random artwork"):
        st.session_state["picked"] = conn().execute(
            "SELECT object_id FROM object_embeddings ORDER BY random() LIMIT 1").fetchone()[0]
    if title_q:
        opts = conn().execute("SELECT o.object_id, o.title FROM objects o JOIN object_embeddings USING (object_id) "
                              "WHERE o.title ILIKE %s ORDER BY o.title LIMIT 50", (f"%{title_q}%",)).fetchall()
        if opts:
            label = st.selectbox("Matches", opts, format_func=lambda r: f"{r[1]} ({r[0]})")
            if st.button("Use this one"):
                st.session_state["picked"] = label[0]
    picked = st.session_state.get("picked")
    if picked:
        mode = st.radio("Similarity", ["image", "hybrid"], horizontal=True,
                        help="image: pure visual. hybrid: visual + title/medium/tags text, fused by rank.")
        q_row = conn().execute("SELECT title FROM objects WHERE object_id = %s", (picked,)).fetchone()
        left, right = st.columns([1, 4])
        with left:
            src = local_paths((picked,)).get(picked) or conn().execute(
                "SELECT image_small_url FROM objects WHERE object_id = %s", (picked,)).fetchone()[0]
            st.image(src, caption=q_row[0] if q_row else picked)
        with right:
            results = similar_to_object(conn(), picked, k=10, filters=filters, mode=mode)
            show_grid(results, "pick")
        if results and st.button("Why is the top match similar? (Claude vision)"):
            try:
                st.info(why_similar({"object_id": picked}, results[0]))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Claude unavailable: {exc}")

with tab_upload:
    up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, width=240)
        vec = embedder().embed_images([img])[0]
        show_grid(search_by_image(conn(), vec, k=10, filters=filters), "upload")
