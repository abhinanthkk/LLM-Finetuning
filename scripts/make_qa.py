"""Stage 2: data/chunks.jsonl -> data/qa/qa_raw.jsonl (question-answer pairs for fine-tuning).

Two sources, one output:
  A. table templates  every numeric table row becomes a look-up question; answers copy the verified
                      numbers, so no model can get them wrong
  B. LLM (Gemini)     3-5 pairs per chunk from the prose (table rows are removed from its input).
                      Each pair must quote its evidence; a pair is dropped unless the quote is in the
                      chunk and every number in the answer is in the quote.

The LLM is any OpenAI-compatible chat endpoint. Defaults are Gemini's:
  put GEMINI_API_KEY=... in the project's .env file (never in this script)
  python scripts/make_qa.py --list-models        see which models your key can use
  python scripts/make_qa.py --model MODEL --limit 5    try 5 chunks first
  python scripts/make_qa.py --model MODEL              all chunks (resumes; answers are cached)
  python scripts/make_qa.py --templates-only           part A only, no key needed
Other providers: set QA_BASE_URL and QA_API_KEY instead.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=ROOT / ".env"):
    """Read KEY=value lines from the project's private .env file into the environment (if not already set)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env()
CHUNKS, OUT_DIR = ROOT / "data" / "chunks.jsonl", ROOT / "data" / "qa"
CACHE = OUT_DIR / "cache"
BASE_URL = os.environ.get("QA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
TYPES = {"definition", "lookup", "quantification", "mitigation", "why", "comparison"}
TABLE_ROW = re.compile(r"^(Table \d+\.\d+): (.+?): (.+)$")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def norm(s):
    """Whitespace and quote-style insensitive form, for checking that a quote is verbatim."""
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def numbers(s):
    return {n.replace(",", "") for n in NUMBER.findall(s)}


# ---------------------------------------------------------------- A. table templates
def split_caption(rest):
    """'Caption (Source: API): Service: X; ...' -> ('Caption (Source: API)', 'Service: X; ...'): the caption
    ends at the first ': ' outside parentheses (captions can contain colons)."""
    depth = 0
    for i, ch in enumerate(rest):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and rest.startswith(": ", i):
            return rest[:i], rest[i + 2:]
    return None, None


def parse_row(line):
    """'Table 2.3: Caption: Service: X; Component: Y; EF (scf/h): 4.9.' -> ('Table 2.3', 'Caption', [(field, value)])"""
    m = re.match(r"^(Table \d+\.\d+): (.+)$", line.rstrip("."))
    if not m:
        return None
    caption, body = split_caption(m.group(2))
    if not caption:
        return None
    fields = []
    for part in body.split("; "):
        if ": " not in part:
            return None                                # unlabelled values (multi-level header): don't guess
        k, v = part.split(": ", 1)
        fields.append((k.strip(), v.strip()))
    return m.group(1), caption, fields


def is_value(v):
    return bool(re.fullmatch(r"[$<>~≈±-]?\s*\d[\d,]*(?:\.\d+)?\s*%?(?:\s*(?:-|to)\s*\d[\d,]*(?:\.\d+)?%?)?", v))


def stem(field):
    return re.sub(r"\s*\(.*\)\s*$", "", field).strip()


# shortcut: tables whose headers can't be phrased as one question (2.5: correlation formulas; 2.7: unlabelled
# multi-level values; 3.3: three-level cost header). Their content still reaches the model via the LLM pairs.
SKIP_TABLES = {"Table 2.5", "Table 2.7", "Table 3.3"}


def drop_ambiguous(pairs):
    """Drop template pairs whose table + row label occurs more than once with different values (Table 1.5 lists
    'Invalco CTU, Pump' five times): the same question would have contradictory answers."""
    key = lambda p: p["answer"].rsplit(" is ", 1)[0]          # 'According to OGMP Table 1.5 (...), the rate for X
    answers = {}
    for p in pairs:
        answers.setdefault(key(p), set()).add(p["answer"])
    return [p for p in pairs if len(answers[key(p)]) == 1], sum(len(a) for a in answers.values() if len(a) > 1)


def template_pairs(chunk):
    pairs, seen = [], set()
    topic = re.sub(r"^Number \d+:\s*", "", chunk["doc_title"])
    for line in chunk["text"].split("\n"):
        row = parse_row(line)
        if not row:
            continue
        table, caption, fields = row
        keys = [(k, v) for k, v in fields if not is_value(v) and re.search(r"[A-Za-z]", v)]
        vals = [(k, v) for k, v in fields if is_value(v)]
        if not keys or not vals or any(len(v) > 80 for _, v in keys):
            continue                                   # descriptive rows (configuration tables) go to the LLM
        if table in SKIP_TABLES or any(re.search(r"E-0|x \(SV\)", v) for _, v in keys):
            continue
        # footnote letter left after a label ('controller vents A'); not after 'Option'/'Type'/'Level'
        subject = ", ".join(re.sub(r"(?<!Option)(?<!Type)(?<!Level)(?<=[a-z]) [A-C]$", "", v) for _, v in keys)
        words = [k.split() for k, _ in vals]
        common = [w[0] for w in zip(*words) if len(set(w)) == 1] if len(vals) > 1 else []
        stems = {stem(k) for k, _ in vals}
        if len(common) >= 2 and len(stems) > 1:   # "Methane Emissions Factor scm/event", "... scf/event"
            what = " ".join(common)
            answer_vals = "; ".join(f"{v} {' '.join(k.split()[len(common):]).strip('()')}" for k, v in vals)
        elif len(stems) == 1:          # one quantity in several units: "4.9 scf/hour/component; 0.14 scm/hour/component"
            what = stems.pop()
            unit = lambda k: (re.search(r"\((.*)\)", k) or [None, ""])[1]
            answer_vals = "; ".join(f"{v} {unit(k)}".strip() for k, v in vals)
        else:                        # different quantities: "Implementation Costs: $12,500; Annual Costs: ..."
            what = " and ".join(stem(k) for k, _ in vals)
            answer_vals = "; ".join(f"{k}: {v}" for k, v in vals)
        what = re.sub(r"\b(.{4,}?) \1\b", r"\1", what)            # 'emission factor emission factor'
        what = " ".join(w if re.search(r"[A-Z].*[A-Z0-9]|\d", w) else w.lower() for w in what.split())  # keep CH4, THC
        # rotate phrasings so the model doesn't learn a single question pattern; every phrasing names the table,
        # because the same row label can appear in two tables with different units (e.g. Tables 7.2 and 7.3)
        h = int(hashlib.md5(line.encode()).hexdigest(), 16) % 3
        question = [f"What is the {what} for {subject} in OGMP {table} ({topic})?",
                    f"According to OGMP {table} ({caption}), what is the {what} for {subject}?",
                    f"For {subject}, what {what} does OGMP {table} of the {topic} guidance give?"][h]
        answer_vals = answer_vals.replace("/ ", "/")              # 'scm/minute/ compressor' -> 'scm/minute/compressor'
        answer = f"According to OGMP {table} ({caption}), the {what} for {subject} is {answer_vals}."
        if question in seen:
            continue
        seen.add(question)
        pairs.append({"chunk_id": chunk["id"], "doc": chunk["doc"], "section": chunk["section"], "type": "lookup",
                      "source": "template", "question": question, "answer": answer, "evidence": line.strip()})
    return pairs


# ---------------------------------------------------------------- B. LLM pairs
RULES = """You are writing training data for a question-answering assistant for oil and gas engineers.
The assistant answers questions about methane emissions using the OGMP (Oil & Gas Methane Partnership)
technical guidance documents.

Rules:
1. Every answer must be fully supported by its PASSAGE. No outside knowledge.
2. Each question must make sense on its own, without the passage: name the equipment or emission source
   (for example "for glycol dehydrators", "for centrifugal compressors with wet seals").
3. Mix question types: definition, quantification (how emissions are estimated), mitigation, why, comparison.
   Only make a comparison question when the passage itself states the comparison.
4. Answers: 1 to 4 sentences, specific, in your own words, but copy every number and unit exactly.
5. "evidence" must be copied VERBATIM from the passage: one sentence or clause (at most 50 words)
   that supports the answer. Every number in the answer must appear in the evidence.
6. Do not ask about document codes, dates, footnotes, citations or webinars.
7. Do not ask two questions about the same fact.
"""
PAIR = '{"type": "definition|quantification|mitigation|why|comparison|lookup", "question": "...", "answer": "...", "evidence": "..."}'
PROMPT = RULES + """
Write {n} question-answer pairs from the PASSAGE below.
Return only a JSON array, no other text:
[""" + PAIR.replace("{", "{{").replace("}", "}}") + """]

DOCUMENT: {doc_title}
SECTION: {section}
PASSAGE:
{text}"""
BATCH_PROMPT = RULES + """
Below are several PASSAGES, each with an ID. For EACH passage, write the number of question-answer pairs
it asks for, using only that passage. Return only a JSON object mapping each passage ID to its array:
{{"<ID>": [""" + PAIR.replace("{", "{{").replace("}", "}}") + """, ...], ...}}

{passages}"""


class DailyQuota(Exception):
    pass


def prose_of(chunk):
    """Chunk text without table-row sentences (those are covered by templates and are where models err)."""
    return "\n".join(l for l in chunk["text"].split("\n") if not TABLE_ROW.match(l)).strip()


def call_llm(client, model, prompt, retries=6):
    for attempt in range(retries):
        try:
            r = client.post("/chat/completions", json={"model": model, "temperature": 0.2,
                                                       "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 429 and "PerDay" in r.text:
                raise DailyQuota(r.text)
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(60, 5 * 2 ** attempt)
                print(f"    HTTP {r.status_code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except httpx.TransportError as e:
            time.sleep(5 * 2 ** attempt)
            print(f"    network error {e!r}, retrying")
    raise RuntimeError("LLM call failed after retries")


def parse_json_array(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("["), text.rfind("]")
    return json.loads(text[start:end + 1])


def check_pair(p, chunk_text):
    """Return None if the pair is usable, else the reason it is dropped."""
    if not isinstance(p, dict) or not all(isinstance(p.get(k), str) and p[k].strip()
                                          for k in ("type", "question", "answer", "evidence")):
        return "missing field"
    if p["type"] not in TYPES:
        return "bad type"
    if norm(p["evidence"]) not in norm(chunk_text):
        return "evidence not verbatim in chunk"
    if not numbers(p["answer"]) <= numbers(p["evidence"]):
        return "answer number not in evidence"
    if len(p["answer"].split()) > 120:
        return "answer too long"
    return None


def wants_llm(chunk):
    return len(prose_of(chunk).split()) >= 60          # mostly-table chunks are covered by templates


def n_pairs(chunk):
    return 3 if len(prose_of(chunk).split()) < 200 else 5


def fill_cache(client, model, chunks, per_call):
    """Ask the LLM for every chunk that has no cached answer yet, per_call chunks per request.
    Returns False if the daily quota ran out (run again later: finished chunks are kept)."""
    todo = [c for c in chunks if wants_llm(c) and not (CACHE / f"{c['id']}.json").exists()]
    print(f"LLM: {len(todo)} chunks to generate, {per_call} per request -> {-(-len(todo) // per_call)} requests")
    for k in range(0, len(todo), per_call):
        group = todo[k:k + per_call]
        try:
            if len(group) == 1:
                c = group[0]
                prompt = PROMPT.format(n=n_pairs(c), doc_title=c["doc_title"], section=c["section"], text=prose_of(c))
                raw = call_llm(client, model, prompt)
                try:
                    parse_json_array(raw)
                except (ValueError, json.JSONDecodeError):
                    raw = call_llm(client, model, prompt + "\n\nReturn ONLY the JSON array.")
                (CACHE / f"{c['id']}.json").write_text(raw, encoding="utf-8")
            else:
                passages = "\n\n".join(
                    f"=== ID: {c['id']} | write {n_pairs(c)} pairs | DOCUMENT: {c['doc_title']} | "
                    f"SECTION: {c['section']}\nPASSAGE:\n{prose_of(c)}" for c in group)
                prompt = BATCH_PROMPT.format(passages=passages)
                obj = None
                for extra in ("", "\n\nReturn ONLY the JSON object."):
                    raw = call_llm(client, model, prompt + extra)
                    try:
                        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
                        obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
                        break
                    except (ValueError, json.JSONDecodeError):
                        continue
                for c in group:                          # chunks missing from the reply are retried next run
                    if obj and isinstance(obj.get(c["id"]), list):
                        (CACHE / f"{c['id']}.json").write_text(json.dumps(obj[c["id"]], ensure_ascii=False), encoding="utf-8")
        except DailyQuota:
            left = len(todo) - k
            print(f"\nDaily quota reached for {model}: {left} chunks still to generate. "
                  f"Run the same command again after the quota resets (finished chunks are cached).")
            return False
        print(f"  request {k // per_call + 1}: {', '.join(c['id'] for c in group)}", flush=True)
    return True


def cached_llm_pairs(chunks, drops):
    """All cached LLM pairs, each attached to the chunk of its document that contains its evidence verbatim.
    Matching by evidence rather than by cache file name keeps pairs valid when re-extraction moves chunk
    boundaries (and with them the chunk ids)."""
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc"], []).append((c, norm(c["text"])))
    out, seen = [], set()
    for f in sorted(CACHE.glob("*.json")):
        doc = re.sub(r"-\d+$", "", f.stem)
        try:
            items = parse_json_array(f.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            drops["unparseable JSON (chunk)"] += 1
            continue
        for p in items:
            ev = norm(p.get("evidence", "")) if isinstance(p, dict) else ""
            home = next((c for c, t in by_doc.get(doc, []) if ev and ev in t), None)
            reason = check_pair(p, home["text"]) if home else "evidence not verbatim in chunk"
            if not reason and norm(p["question"]) in seen:
                reason = "duplicate question"
            if reason:
                drops[reason] += 1
                continue
            seen.add(norm(p["question"]))
            out.append({"chunk_id": home["id"], "doc": home["doc"], "section": home["section"], "type": p["type"],
                        "source": "llm", "generator": p.get("generator", "gemini"),
                        "question": p["question"].strip(), "answer": p["answer"].strip(),
                        "evidence": p["evidence"].strip()})
    covered = {p["chunk_id"] for p in out}
    drops["prose chunk without LLM pairs"] += sum(1 for c in chunks if wants_llm(c) and c["id"] not in covered)
    return out


# ---------------------------------------------------------------- main
def self_check():
    line = ("Table 2.3: Onshore Operations Methane Emission Factors: Service: All Components – Gas Service; "
            "Component: Valve; Leaker Emission Factor (scf/hour/component): 4.9; "
            "Leaker Emission Factor (scm/hour/component): 0.14.")
    chunk = {"id": "t-00", "doc": "TGD2", "section": "s", "doc_title": "Number 2: Fugitive Component and Equipment Leaks",
             "text": line}
    assert split_caption("Factors (Source: API): Source: Venting; EF: 2") == ("Factors (Source: API)", "Source: Venting; EF: 2")
    p = template_pairs(chunk)[0]
    assert "4.9 scf/hour/component; 0.14 scm/hour/component" in p["answer"], p["answer"]
    assert "All Components – Gas Service, Valve" in p["question"] and "Table 2.3" in p["question"], p["question"]
    a = dict(chunk_id="x", answer="According to OGMP Table 1.5 (T), the rate for Invalco CTU, Pump is 8.07 scm/hr.")
    b = dict(a, answer=a["answer"].replace("8.07", "15.28"))
    assert drop_ambiguous([a, b, dict(a, answer=a["answer"].replace("CTU", "CTS"))])[1] == 2
    ok = {"type": "why", "question": "q", "answer": "It is 4.9.", "evidence": "Valve; Leaker"}
    assert check_pair(dict(ok, evidence="Leaker Emission Factor (scf/hour/component): 4.9"), line) is None
    assert check_pair(dict(ok, evidence="Valve; Leaker"), line) == "answer number not in evidence"
    assert check_pair(dict(ok, evidence="not in the text"), line) == "evidence not verbatim in chunk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model id, e.g. from --list-models")
    ap.add_argument("--limit", type=int, help="only the first N chunks (for a test run)")
    ap.add_argument("--templates-only", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--per-call", type=int, default=1, help="chunks per LLM request (fewer requests, same checks)")
    ap.add_argument("--from-cache", action="store_true", help="no API calls: build pairs from cached answers only")
    args = ap.parse_args()
    self_check()

    client = None
    if not args.templates_only and not args.from_cache:
        key = os.environ.get("QA_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("No key found. Put GEMINI_API_KEY=... in /home/abee/usecase/.env, or use --templates-only.")
        client = httpx.Client(base_url=BASE_URL, timeout=180, headers={"Authorization": f"Bearer {key}"})
        if args.list_models:
            r = client.get("/models"); r.raise_for_status()
            print("\n".join(sorted(m["id"].removeprefix("models/") for m in r.json()["data"])))
            return
        if not args.model:
            sys.exit("Pass --model (see --list-models).")

    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")][:args.limit]
    CACHE.mkdir(parents=True, exist_ok=True)
    drops, pairs = Counter(), []
    complete = fill_cache(client, args.model, chunks, args.per_call) if client else True
    templates, n_ambiguous = drop_ambiguous([p for c in chunks for p in template_pairs(c)])
    drops["ambiguous table row (same label, different values)"] += n_ambiguous
    llm = cached_llm_pairs(chunks, drops) if not args.templates_only else []
    for i, c in enumerate(chunks, 1):
        got = [p for p in templates + llm if p["chunk_id"] == c["id"]]
        for j, p in enumerate(got, 1):
            pairs.append({"id": f"{c['id']}-q{j:02d}", **p})
        print(f"[{i}/{len(chunks)}] {c['id']}: {len(got)} pairs", flush=True)

    out = OUT_DIR / ("qa_raw.jsonl" if not args.limit else "qa_sample.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pairs)} pairs to {out.relative_to(ROOT)}")
    print("by source:", dict(Counter(p.get("generator", p["source"]) for p in pairs)))
    print("by type:  ", dict(Counter(p["type"] for p in pairs)))
    print("by doc:   ", dict(Counter(p["doc"] for p in pairs)))
    print("dropped:  ", dict(drops) or "none")
    if not complete:
        print("INCOMPLETE: some chunks have no LLM pairs yet (see 'chunk not generated yet').")


if __name__ == "__main__":
    main()
