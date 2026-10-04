"""Claude turns a natural-language query into structured filters via tool use.

"Japanese woodblock prints of birds from the 1800s" ->
    Filters(cultures=["Japan"], classifications=["Prints"], year_from=1800, year_to=1899)
    + free_text "birds"  (which becomes a CLIP text embedding)

Claude never ranks or names artworks; it only fills in the filter schema. That
keeps the step testable (assert on the parsed filters) and means it cannot
hallucinate an object.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .search import Filters

log = logging.getLogger(__name__)

TOOL_NAME = "apply_search_filters"

SYSTEM = """You translate art-collection search requests into structured filters for the \
Metropolitan Museum of Art collection. Always answer by calling the {tool} tool exactly once.

Rules:
- Put visual subject matter (what the artwork depicts or looks like) in free_text; it is \
matched by an image-similarity model. Example: "birds", "a woman reading by a window".
- Only set a filter when the request clearly implies it. Leave lists empty and years null otherwise.
- Culture values are the Met's own wording: a country or people such as "Japan", "China", \
"Korea", "India", "Egypt", "French", "Dutch", "Italian". Use the nouns the Met uses for Asian \
cultures ("Japan") and adjectives for European ones ("French").
- Classification values look like: Prints, Paintings, Drawings, Ceramics, Sculpture, \
Textiles, Calligraphy, Metalwork, Photographs, Jewelry.
- Centuries: "1800s" or "19th century" is 1800-1899. "Edo period" is 1615-1868. \
"Ming dynasty" is 1368-1644. Use negative years for BCE.
- departments must come from this list, or be empty: {departments}."""


def _tool(departments: list[str]) -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Apply structured filters and a free-text visual query to the collection search.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "free_text": {"type": "string",
                              "description": "Visual subject to search for; empty string if none."},
                "departments": {"type": "array", "items": {"type": "string", "enum": departments}},
                "cultures": {"type": "array", "items": {"type": "string"}},
                "classifications": {"type": "array", "items": {"type": "string"}},
                "year_from": {"type": ["integer", "null"]},
                "year_to": {"type": ["integer", "null"]},
            },
            "required": ["free_text", "departments", "cultures", "classifications", "year_from", "year_to"],
            "additionalProperties": False,
        },
    }


@dataclass
class ParsedQuery:
    free_text: str
    filters: Filters
    raw: dict | None = None      # the tool input, for display/debugging
    fallback: bool = False       # True when Claude was unavailable and we used the raw text


def parse_tool_input(data: dict, departments: list[str]) -> ParsedQuery:
    filters = Filters(
        departments=[d for d in data.get("departments") or [] if d in departments],
        cultures=[c for c in data.get("cultures") or [] if c],
        classifications=[c for c in data.get("classifications") or [] if c],
        year_from=data.get("year_from"),
        year_to=data.get("year_to"),
    )
    if filters.year_from is not None and filters.year_to is not None and filters.year_from > filters.year_to:
        filters.year_from, filters.year_to = filters.year_to, filters.year_from
    return ParsedQuery(free_text=(data.get("free_text") or "").strip(), filters=filters, raw=data)


def parse_query(query: str, departments: list[str], client=None, model: str = "claude-opus-5-5") -> ParsedQuery:
    """Ask Claude for filters. Falls back to plain free-text search on any failure."""
    try:
        import anthropic

        client = client or anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM.format(tool=TOOL_NAME, departments=", ".join(departments)),
            tools=[_tool(departments)],
            # Forced tool_choice is not accepted on this model; the system prompt asks for
            # the call and strict=True guarantees a schema-valid input when it comes.
            tool_choice={"type": "auto"},
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": query}],
        )
    except Exception as exc:  # noqa: BLE001 - no key, network, API error: degrade gracefully
        log.warning("filter parsing unavailable (%s); using raw text", exc)
        return ParsedQuery(free_text=query, filters=Filters(), fallback=True)

    if response.stop_reason == "refusal":
        return ParsedQuery(free_text=query, filters=Filters(), fallback=True)
    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            data = block.input if isinstance(block.input, dict) else json.loads(block.input)
            return parse_tool_input(data, departments)
    log.warning("Claude answered without calling the tool; using raw text")
    return ParsedQuery(free_text=query, filters=Filters(), fallback=True)
