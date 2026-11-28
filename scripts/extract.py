"""Stage 1: OGMP PDFs -> clean section chunks (data/chunks.jsonl) + readable per-document text (data/extracted/).

What is kept and dropped (see the study guide, chapter 4):
  kept    body text, headings (as section names), figure/exhibit/table captions, tables (one sentence per row)
  dropped running header (y < 80), "Page N of M" footer, footnotes (font < 9.5 pt), superscript markers,
          labels scattered inside vector diagrams and images, photos
usage: python scripts/extract.py
"""
import json
import re
import unicodedata
from pathlib import Path

import pymupdf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "OGMP docs", ROOT / "data"
HEADER_Y, FOOTER_Y = 80, 735          # page is 792 pt tall; measured on every document
MIN_BODY_SIZE = 9.5                    # footnotes are 9 pt, their numbers 6-6.5 pt
MAX_TOKENS, MIN_TOKENS = 500, 120      # chunk size in Qwen tokens
CAPTION = re.compile(r"^(Figure|Exhibit|Table)\s+[A-Z0-9][\w.]*", re.I)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
ntok = lambda s: len(tok(s)["input_ids"])


def body_spans(spans):
    """Body-size spans plus small subscripts (CO2, H2S, m3). Small spans raised above the baseline are
    footnote markers, and a line that is all small text is a footnote: both are dropped."""
    out = []
    for s in spans:
        if not s["text"].strip():
            continue
        if s["size"] >= MIN_BODY_SIZE:
            out.append(s)
        elif not out:
            continue                                                     # leading small number = footnote line
        elif s["origin"][1] >= out[-1]["origin"][1] - 0.5:                # subscript: on or below the baseline
            out.append(s)
        elif s["text"].strip() in ("2", "3") and re.search(r"\b(m|ft|cm|km)$", out[-1]["text"]):
            out.append(s)                                                # raised unit exponent: m3, ft3
    return out


def clean(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("", "-").replace("", "-").replace("•", "-")   # Symbol-font bullets
    return re.sub(r"[ \t]+", " ", s).strip()


def overlaps(r, rects, frac=0.5):
    return r.get_area() > 0 and any((r & q).get_area() > frac * r.get_area() for q in rects)


def cell_text(page, cell):
    # read the cell's box from the page: find_tables' own text splits "4.9" into "49\n." and drops spaces
    return clean(page.get_textbox(pymupdf.Rect(cell)).replace("\n", " ")) if cell else ""


def drop_markers(cell, header=False):
    """Footnote markers glued to words: 'FactorA' -> 'Factor'; in headers also 'Factor14' -> 'Factor'."""
    cell = re.sub(r"(?<=[a-z)”\"])[A-C](?=[\s;:)]|$)", "", cell)          # also after a closing bracket/quote
    if header:
        cell = re.sub(r"\)[a-z](?=\s|$)", ")", cell)                     # '(scf THC/h/source)b' -> '...)'
        return re.sub(r"(?<=[a-z])\d{1,2}(?=[\s(]|$)", "", cell)
    cell = re.sub(r"(?<=[a-z])\d{1,2}$", "", cell)                         # 'Processing19' -> 'Processing'
    return re.sub(r"(?<=[”\")])\s*\d{1,2}$", "", cell).strip()             # '(“Mitigated”) 1' -> '(“Mitigated”)'


def is_number(v):
    return bool(re.fullmatch(r"[$<>~±-]?\s*[\d.,]+\s*%?", v))


def table_sentences(rows, header, caption, state=None):
    """One sentence per data row: 'Caption: col: value; col: value.'
    - Empty LABEL cells inherit from the row above (merged cells, e.g. 'All Components - Gas Service' spanning
      rows); empty NUMBER cells never do.
    - A row with only its first cell filled is a section heading ('Gas Transmission & Storage'): it is not a
      data row, and the rows under it are named 'Section – Row'.
    state = (previous row, current section), carried across pages for tables split over a page break."""
    header = [drop_markers(h, header=True) for h in header]
    prev, section = state or ([""] * len(header), "")
    out = []
    for row in rows:
        row = [drop_markers(c) for c in row]
        filled = [i for i, c in enumerate(row) if c]
        if filled == [0] and len(row) > 1:
            section = row[0]
            prev = [""] * len(header)
            continue
        row = [c if c else (prev[i] if i < len(prev) and not is_number(prev[i]) else "") for i, c in enumerate(row)]
        prev = row
        cells = list(row)
        if section and cells and cells[0] and cells[0] != section:
            cells[0] = f"{section} – {cells[0]}"
        pairs = [f"{h}: {v}" if h else v for h, v in zip(header, cells) if v]
        if len(pairs) > 1:                                       # a lone value is a spanning label, not a row
            out.append((f"{caption}: " if caption else "") + "; ".join(pairs) + ".")
    return out, (prev, section)


def split_header(rows):
    """Header row(s) and data rows. A header with blank (merged) cells spanning a second row of sub-headings,
    e.g. 'Methane Emission Factor' over '(scm/hour)', '(scf/hour)', becomes one row: 'Methane Emission Factor (scm/hour)'."""
    head = rows[0]
    numeric = lambda c: bool(re.fullmatch(r"[\d.,%$\s()-]+", c))
    if "" in head and len(rows) > 2 and not any(numeric(c) for c in rows[1] if c):
        top, last = [], ""
        for c in head:
            last = c or last
            top.append(last)
        return [f"{t} {b}".strip() if b and b != t else t for t, b in zip(top, rows[1])], rows[2:]
    return head, rows[1:]


def diagram_regions(page, t_rects):
    """Padded boxes around vector drawings that are not tables; labels sit just outside the strokes."""
    big = [r for r in page.cluster_drawings() if r.width > 100 and r.height > 60 and not overlaps(r, t_rects, 0.3)]
    return [r + (-35, -20, 35, 20) for r in big]


def page_items(page, carry):
    """Return the page's content in reading order as (kind, text), kind in heading/para/caption/table.
    carry = (header, caption, ncols) of the last table on the previous page, for tables split across pages."""
    tabs = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    t_rects = [pymupdf.Rect(t.bbox) for t in tabs]
    img_rects = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info() if i["bbox"][3] > HEADER_Y]
    figures = diagram_regions(page, t_rects) + img_rects

    items = []                                                   # (y, kind, text)
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        para, para_y = "", None
        for ln in b["lines"]:
            r = pymupdf.Rect(ln["bbox"])
            if r.y1 < HEADER_Y or r.y0 > FOOTER_Y or overlaps(r, t_rects):
                continue
            spans = body_spans(ln["spans"])
            if not spans:
                continue
            text = clean("".join(s["text"] for s in spans))
            bold = all("Bold" in s["font"] for s in spans)
            is_cap = bold and bool(CAPTION.match(text))
            if not is_cap and overlaps(r, figures) and (len(text) < 45 or spans[0]["size"] > 12):
                continue                                         # diagram label, or text drawn over a photo
            if bold and not is_cap and spans[0]["size"] > 11.5 and page.number > 0:
                continue                                         # headings are 10-11 pt; bigger bold = diagram label
            if text.startswith("Mitigation Option") or is_cap or (
                    bold and len(text) < 110 and "=" not in text and text != "-"):
                if para:
                    items.append((para_y, "para", para, para_y1)); para = ""
                prev = items[-1] if items else None
                if (prev and r.y0 - prev[3] < 20 and not is_cap and not text.startswith("Mitigation Option")
                        and (prev[1] != "para" or text[0].islower())):
                    # second line of a wrapped caption/heading (or a heading whose first line read as text)
                    kind = "heading" if prev[2].startswith("Mitigation Option") else prev[1]
                    items[-1] = (prev[0], kind, prev[2] + " " + text, r.y1)
                else:
                    items.append((r.y0, "caption" if is_cap else "heading", text, r.y1))
                continue
            para_y1 = r.y1
            if not para:
                para, para_y = text, r.y0
            elif text == "-" or text.startswith("- "):
                para += "\n" + text
            elif para.endswith("\n-") or para == "-":
                para += " " + text
            elif para.endswith("-") and not para.endswith(" -"):
                para = para[:-1] + text                          # word hyphenated at line end
            else:
                para += " " + text
        if para:
            items.append((para_y, "para", para, para_y1))
    items = [it[:3] for it in items]

    for t, r in zip(tabs, t_rects):
        rows = [[cell_text(page, c) for c in row.cells] for row in t.rows]
        caps = [it for it in items if it[1] == "caption" and it[2].lower().startswith("table") and it[0] < r.y0]
        caption = re.sub(r"\d+$", "", caps[-1][2]) if caps and r.y0 - caps[-1][0] < 80 else ""
        caption = re.sub(r"^(Table \d+\.\d+):(?=\S)", r"\1: ", caption)         # 'Table 4.3:Typical' -> 'Table 4.3: Typical'
        last = None
        if not caption and carry and carry[2] == t.col_count and r.y0 < 160:
            header, caption, _, last = carry                     # continuation of last page's table
            if rows[0] == header:
                rows = rows[1:]
        else:
            header, rows = split_header(rows)
        sentences, last = table_sentences(rows, header, caption, last)
        items.append((r.y0, "table", "\n".join(sentences)))
        carry = (header, caption, t.col_count, last)
    items.sort(key=lambda it: it[0])
    if not tabs and any(k == "para" for _, k, _ in items):
        carry = None                                             # a page of prose ends any running table
    return [(k, t) for _, k, t in items if re.search(r"[A-Za-z]", t)], carry   # drop stray "." etc.


def doc_title(doc):
    text = doc[0].get_text()
    m = re.search(r"(Number\s+\d+:?.*?|APPENDIX A:.*?)\s*\n\s*(Modified|May)", text, re.S)
    assert m, f"no title found in {doc.name}"
    return re.sub(r"^Number (\d+):?\s*", r"Number \1: ", re.sub(r"\s+", " ", m.group(1)).strip(" :"))


def chunk_sections(sections):
    """sections: list of (heading, pages, text). Merge small sections forward, split big ones on paragraphs."""
    merged = []
    for h, pages, text in sections:
        if merged and ntok(merged[-1][2]) < MIN_TOKENS:
            ph, pp, pt = merged[-1]
            merged[-1] = (ph, pp | pages, f"{pt}\n\n{h}\n{text}")
        else:
            merged.append((h, set(pages), text))
    chunks = []
    for h, pages, text in merged:
        cur = ""
        pieces = []
        for para in text.split("\n\n"):
            if ntok(para) <= MAX_TOKENS:
                pieces.append(para)
            else:   # big table -> its row sentences; long paragraph -> its sentences
                pieces += para.split("\n") if "\n" in para else re.split(r"(?<=[.;])\s+(?=[A-Z(])", para)
        for para in pieces:
            sep = "\n" if para.startswith(cur.split(":")[0]) and cur.startswith("Table") else "\n\n"
            if cur and ntok(cur + sep + para) > MAX_TOKENS:
                chunks.append((h, pages, cur))
                cur = para
            else:
                cur = f"{cur}{sep}{para}" if cur else para
        if cur:
            chunks.append((h, pages, cur))
    return chunks


def drop_redundant(chunks, max_overlap=0.5, min_words=40):
    """Drop (1) chunks that repeat another document's text, e.g. the shared OGMP introduction that opens every
    TGD (kept once, from the first document), and (2) chunks that are mostly figure captions."""
    shingles = lambda t: {tuple(w[i:i + 5]) for w in [re.findall(r"\w+", t.lower())] for i in range(max(1, len(w) - 4))}
    kept, kept_sh = [], []
    for c in chunks:
        if len(re.sub(r"\[[^\]]*\]", "", c["text"]).split()) < min_words:
            print(f"  dropped {c['id']}: mostly captions"); continue
        sh = shingles(c["text"])
        twin = next((k for k, ks in zip(kept, kept_sh)
                     if k["doc"] != c["doc"] and len(sh & ks) > max_overlap * len(sh | ks)), None)
        if twin:
            print(f"  dropped {c['id']}: repeats {twin['id']}"); continue
        kept.append(c); kept_sh.append(sh)
    return kept


def main():
    (OUT / "extracted").mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for pdf in sorted(SRC.glob("*.pdf")):
        doc = pymupdf.open(pdf)
        title = doc_title(doc)
        short = re.search(r"OGMP-(.+?)_CCAC", pdf.name).group(1)          # e.g. TGD3-Centrifugal-Compressors
        sections, heading, pages, buf, carry = [], "Introduction", set(), [], None
        md = [f"# {title}\n"]
        for page in doc:
            items, carry = page_items(page, carry)
            for kind, text in items:
                if kind == "heading" and not re.match(r"^(TECHNICAL GUIDANCE DOCUMENT|APPENDIX A)", text):
                    if buf:
                        sections.append((heading, pages, "\n\n".join(buf)))
                    heading, pages, buf = text, set(), []
                    md.append(f"\n## {text}\n")
                    continue
                if kind == "heading":
                    continue                                              # document title block
                piece = f"[{text}]" if kind == "caption" else text
                buf.append(piece)
                pages.add(page.number + 1)
                md.append(piece + "\n")
        if buf:
            sections.append((heading, pages, "\n\n".join(buf)))
        (OUT / "extracted" / f"{short}.md").write_text("\n".join(md), encoding="utf-8")
        for i, (h, pg, text) in enumerate(chunk_sections(sections)):
            all_chunks.append({"id": f"{short}-{i:02d}", "doc": short, "doc_title": title, "section": h,
                               "pages": sorted(pg), "n_tokens": ntok(text), "text": text})
    all_chunks = drop_redundant(all_chunks)
    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # self-check: nothing we meant to drop leaked into the chunks
    joined = "\n".join(c["text"] for c in all_chunks)
    assert not re.search(r"Page \d+ of \d+", joined), "page footer leaked"
    assert "Technical Guidance Document\nNumber" not in joined, "running header leaked"
    assert all(c["n_tokens"] <= MAX_TOKENS * 1.5 for c in all_chunks), "oversized chunk"
    docs = sorted({c["doc"] for c in all_chunks})
    print(f"{len(all_chunks)} chunks from {len(docs)} documents, "
          f"{sum(c['n_tokens'] for c in all_chunks):,} tokens; median "
          f"{sorted(c['n_tokens'] for c in all_chunks)[len(all_chunks)//2]} tokens/chunk")
    for d in docs:
        cs = [c for c in all_chunks if c["doc"] == d]
        print(f"  {d:40s} {len(cs):3d} chunks  {sum(c['n_tokens'] for c in cs):6,} tokens")


assert drop_markers("Methane Emission Factor14 (scf)", header=True) == "Methane Emission Factor (scf)"
assert drop_markers("Capital CostA") == "Capital Cost" and drop_markers("CH4 vents") == "CH4 vents"
assert drop_markers("Diaphragm (“Unmitigated”)A") == "Diaphragm (“Unmitigated”)" and drop_markers("Option A") == "Option A"
assert table_sentences([["", "Pump Seal", "3.7"]], ["Service", "Component", "EF"], "T", (["Gas", "Valve", "4.9"], ""))[0] == [
    "T: Service: Gas; Component: Pump Seal; EF: 3.7."]
assert table_sentences([["Transmission", "", ""], ["High bleed", "0.5", "18.2"], ["Low bleed", "", "1.4"]],
                       ["Source", "EF (scm)", "EF (scf)"], "T")[0] == [
    "T: Source: Transmission – High bleed; EF (scm): 0.5; EF (scf): 18.2.",
    "T: Source: Transmission – Low bleed; EF (scf): 1.4."]          # section row named, number not copied down
assert drop_markers("Processing19") == "Processing" and drop_markers("Fisher 546") == "Fisher 546"
assert drop_markers("vents (“Mitigated”)A  1") == "vents (“Mitigated”)" and drop_markers("Williams P125") == "Williams P125"
assert [x["text"] for x in body_spans([{"text": "CO", "size": 11, "origin": (0, 100)}, {"text": "2", "size": 7, "origin": (9, 102)},
        {"text": "3", "size": 7, "origin": (12, 96)}])] == ["CO", "2"]                     # subscript kept, superscript dropped
assert [x["text"] for x in body_spans([{"text": "17 m", "size": 11, "origin": (0, 100)}, {"text": "3", "size": 7, "origin": (9, 96)},
        {"text": "/hour.", "size": 11, "origin": (12, 100)}, {"text": "21", "size": 7, "origin": (40, 96)}])] == ["17 m", "3", "/hour."]
assert split_header([["Source", "EF", ""], ["", "(scm/h)", "(scf/h)"], ["Wet seal", "60.6", "2,140"]]) == (
    ["Source", "EF (scm/h)", "EF (scf/h)"], [["Wet seal", "60.6", "2,140"]])

if __name__ == "__main__":
    main()
