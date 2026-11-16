"""Shared retrieve+answer wrapper (used by the benchmark now, by the backend in M9).

ask(kb, question) -> {"answer", "citations": [{doc, revision, page_start, page_end, key}], "figures": [...], "refs"}
A figure is returned only when the answer cites it: by [ref_id:N] of a figure chunk, or by a [DOC p.N]
citation whose page equals a retrieved figure's page of the same document.
"""
import re
import time

import requests

from common import config as C

CIT = re.compile(r"\[([A-Z]+-\d+)(?:\s+Rev\s+\w+)?\s+pp?\.?\s*(\d+(?:\s*[-–,;]\s*(?:pp?\.?\s*)?\d+)*)\]")


def cited_spans(answer):
    """[(doc, first_page, last_page)] from citations like [DOC Rev X p.4], [DOC pp.4-6] or [DOC pp.4, 8]."""
    out = []
    for doc, pages in CIT.findall(answer):
        for part in re.split(r"[,;]", pages):
            nums = [int(n) for n in re.findall(r"\d+", part)]
            out.append((doc, nums[0], nums[-1]))
    return out
REF = re.compile(r"ref_id:(\d+)")
FIELDS = "chunk_id,kind,platform,revision,caption,image_path,chunk_page_start,chunk_page_end"
_meta = {}


def chunk_meta(key):
    if key not in _meta:
        r = requests.get(f"{C.SEARCH_ENDPOINT}/indexes/{C.INDEX_NAME}/docs/{key}"
                         f"?api-version={C.SEARCH_API_VERSION}&$select={FIELDS}", headers=C.search_headers())
        _meta[key] = r.json() if r.ok else {}
    return _meta[key]


def retrieve(kb, question):
    url = f"{C.SEARCH_ENDPOINT}/knowledgebases/{kb}/retrieve?api-version={C.SEARCH_API_VERSION}"
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}]}
    for attempt in range(4):
        r = requests.post(url, headers=C.search_headers(), json=body, timeout=300)
        if r.ok:
            return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"retrieve failed {r.status_code}: {r.text[:300]}")


def ask(kb, question):
    try:
        j = retrieve(kb, question)
    except (RuntimeError, requests.RequestException) as e:
        return {"answer": f"ERROR {e}", "citations": [], "text_cites": [], "figures": [], "refs": {}}
    answer = "".join(c.get("text", "") for m in j.get("response", []) for c in m.get("content", []))
    refs = {str(x.get("id")): x.get("docKey") for x in j.get("references", [])}

    citations, seen = [], set()
    for k in REF.findall(answer):
        key = refs.get(k)
        if key and key not in seen:
            seen.add(key)
            m = chunk_meta(key)
            citations.append({"key": key, "kind": m.get("kind"), "doc": m.get("platform"),
                              "revision": m.get("revision"), "page_start": m.get("chunk_page_start"),
                              "page_end": m.get("chunk_page_end"), "image_path": m.get("image_path"),
                              "caption": m.get("caption")})
    figures = [{"key": c["key"], "doc": c["doc"], "revision": c["revision"], "page": c["page_start"],
                "caption": c["caption"], "image_path": c["image_path"]} for c in citations if c["kind"] == "figure"]

    # cited by page only: match retrieved figure chunks of that document/page
    cited_pages = {(d, p) for d, a, b in cited_spans(answer) for p in range(a, b + 1)}
    have = {f["key"] for f in figures}
    for key in refs.values():
        if key in have or "-fig" not in key:
            continue
        m = chunk_meta(key)
        if (m.get("platform"), m.get("chunk_page_start")) in cited_pages:
            figures.append({"key": key, "doc": m["platform"], "revision": m["revision"],
                            "page": m["chunk_page_start"], "caption": m["caption"], "image_path": m["image_path"]})
            have.add(key)
    return {"answer": answer, "citations": citations, "figures": figures, "refs": refs,
            "text_cites": [{"doc": d, "page_start": a, "page_end": b} for d, a, b in cited_spans(answer)]}


_all_figures = []


def all_figures():
    if not _all_figures:
        r = requests.post(f"{C.SEARCH_ENDPOINT}/indexes/{C.INDEX_NAME}/docs/search?api-version={C.SEARCH_API_VERSION}",
                          headers=C.search_headers(), json={"search": "*", "filter": "kind eq 'figure'", "top": 200,
                                                           "select": FIELDS})
        r.raise_for_status()
        _all_figures.extend(r.json()["value"])
    return _all_figures


def figures_from_text(answer):
    """Figures an agent answer relies on: a cited [DOC p.N] page that holds a figure of that document."""
    cited = {(d, p) for d, a, b in cited_spans(answer) for p in range(a, b + 1)}
    seen, out = set(), []
    for f in all_figures():
        if (f["platform"], f["chunk_page_start"]) in cited and f["chunk_id"] not in seen:
            seen.add(f["chunk_id"])
            out.append({"key": f["chunk_id"], "doc": f["platform"], "revision": f["revision"],
                        "page": f["chunk_page_start"], "caption": f["caption"], "image_path": f["image_path"]})
    return out
