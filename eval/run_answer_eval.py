"""Answer-quality evaluation: is the answer good, not just the retrieval.

`run_eval.py` measures whether the right passages reach Claude. This measures
what Claude then does with them, which is a different failure: the pipeline can
retrieve perfectly and still produce an answer that invents a visa rule or
confidently answers a question the archive knows nothing about.

Three checks, in increasing order of cost:

  citations   Do claims point at a real excerpt? Parses the [1]..[5] markers
              the system prompts ask for. Catches ungrounded answers without
              any model call. Free.
  refusal     Given a question the archive cannot answer (Australian visas,
              Japanese tuition), does the answer decline instead of inventing?
              Free, and covers the failure that damages a demo most.
  judge       Claude grades faithfulness - is every claim supported by the
              excerpts - and relevance. Costs money, so it is opt-in via
              --judge and defaults to Opus, a different model from the
              one answering, so it is not grading its own work.

    python eval/run_answer_eval.py                 # free checks only
    python eval/run_answer_eval.py --judge         # adds the LLM judge
    python eval/run_answer_eval.py --limit 5       # a quick subset

Answers are cached to disk keyed by the question, the models, the retrieval
settings, the system prompts and the index, so re-running to add the judge does
not pay for generation twice while any change that can alter an answer
invalidates the entry.

Reading the numbers
-------------------
The free checks are reliable; the judge is directional. Measured on the first
full run: citations 100%, unanswerable declined 100%, judge faithfulness 3.38,
relevance 4.31.

Faithfulness sits well below relevance, and spot-checking the transcripts shows
the gap is part real and part judge severity. The real part is a consistent
habit of adding interpretive caveats - a date, an inference - that the excerpts
do not state. The severity part: the lowest-scored answer (1/5) was in fact a
careful one that flagged its own missing data and labelled every figure as
UK-general rather than London-specific, and did not deserve a 1.

So treat faithfulness as a floor and a change detector, not a verdict, and read
the answer before acting on a low score. The same rule applies here as in
run_eval.py: when the metric looks surprising, the metric is the first suspect.

Known limit of the refusal check
--------------------------------
It over-reports. An answer that opens by narrowing scope - "there is no clear
consensus on the best city, but students mentioned these criteria [1]" - and
then answers at length is counted as a refusal. Two such answers are miscounted
today, so "answered" is a lower bound on the real figure.

Citation count cannot fix this: across real answers, refusals cited 0, 2, 2, 3
and 5 distinct excerpts while a substantive answer cited 2, so no threshold
separates them. Distinguishing a hedge from a decline needs the judge, not a
keyword rule. Always read the listed refusals rather than trusting the count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from murshid import config  # noqa: E402
from murshid.claude import ClaudeError, complete  # noqa: E402
from murshid.rag import _SYSTEM_AR, _SYSTEM_EN, RAGPipeline  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.json"
CACHE = Path(__file__).parent / ".answer_cache.json"

JUDGE_MODEL = "claude-opus-5-5"

_CITATION = re.compile(r"\[(\d+)\]")

# Sentence-ish segmentation for both scripts. Arabic uses ؟ and ۔ alongside
# Latin punctuation, and answers are often bulleted rather than punctuated.
_SENTENCE_SPLIT = re.compile(r"[.!?؟\n]+")

# Phrases either system prompt can use to decline. The prompts instruct Claude
# to say plainly when the excerpts do not cover a question; these are the forms
# that instruction actually produces. Collected by reading real answers - an
# earlier, shorter list scored a correct refusal as a failure because it
# matched "no information" but not "don't have any information". When a
# refusal is misclassified, widen this list rather than trusting the number.
_REFUSAL_MARKERS = (
    "لا تحتوي",
    "لا توجد",
    "لم أجد",
    "لا يوجد",
    "لا تغطي",
    "غير متوفر",
    "لا تتضمن",
    "ليس لدي",
    "لا أملك",
    "خارج نطاق",
    "لا أستطيع",
    "لا تشمل",
    "do not contain",
    "don't contain",
    "does not contain",
    "no information",
    "don't have any information",
    "do not have any information",
    "don't have information",
    "no mention",
    "not covered",
    "cannot answer",
    "can't answer",
    "could not find",
    "couldn't find",
    "not in the archive",
    "do not cover",
    "outside what",
    "outside the scope",
    "don't address",
    "do not address",
)


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def cache_key(question: str) -> str:
    """Identify a cached answer by everything that can change it.

    The retrieval settings belong in the key: an answer is a function of the
    passages that reached Claude, so a change to the candidate pool or the
    rerank cutoff produces a different answer from the same question and
    model. The system prompts count for the same reason. Keying on the models
    alone silently served answers built from a previous configuration and
    reported them as current.
    """
    settings = ":".join(
        str(part)
        for part in (
            config.CLAUDE_MODEL,
            config.VOYAGE_MODEL,
            config.RERANK_MODEL,
            config.RETRIEVE_CANDIDATES,
            config.TOP_K_RESULTS,
            config.RERANK_THRESHOLD,
        )
    )
    # The system prompts shape the answer as much as retrieval does, so they
    # are part of the identity too - hashed rather than inlined to keep the
    # key readable.
    prompts = hashlib.sha256((_SYSTEM_AR + _SYSTEM_EN).encode()).hexdigest()[:12]
    return f"{settings}:{prompts}:{index_fingerprint()}:{question}"


def index_fingerprint() -> str:
    """Identify the index answers were retrieved from.

    Same reasoning as the settings: a rebuilt index retrieves different
    passages, so answers cached against the old one are stale even when every
    setting matches. Keyed on content rather than modification time, which a
    checkout or a copy changes without changing a single vector.
    """
    path = config.INDEX_PATH
    if not path.exists():
        return "no-index"
    stat = path.stat()
    return _hash_file(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _hash_file(path: str, size: int, mtime_ns: int) -> str:
    """Hash a file once per version rather than once per question."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def answer_question(pipeline: RAGPipeline, question: str, cache: dict) -> dict:
    """Run the real pipeline end to end, reusing a cached answer when present."""
    key = cache_key(question)
    if key in cache:
        return cache[key]

    language, sources, error = pipeline.retrieve(question)
    if error:
        raise ClaudeError(f"retrieval failed: {error}")

    answer = "" if not sources else "".join(
        pipeline.stream_answer(question, language, sources)
    )

    record = {
        "question": question,
        "language": language,
        "answer": answer,
        "source_count": len(sources),
        "sources": [d.content for d in sources],
        # Exactly what the model saw, excerpt headers included, so the judge
        # grades against the same evidence rather than a subset of it.
        "prompt": pipeline._build_prompt(question, language=language, sources=sources)
        if sources
        else "",
    }
    cache[key] = record
    save_cache(cache)
    return record


# A refusal announces itself up front, within roughly the first paragraph.
# Later on, a marker is far more likely to be reported speech - good Arabic
# answers quote students saying "لا يوجد" or "ما صار شي" - than the answer
# itself declining.
#
# The window is a deliberate middle ground, arrived at by testing both ends:
# searching the whole answer scored four correct Arabic answers as refusals,
# while searching only the first sentence missed a real refusal that opened
# "ما أقدر أساعدك بهذا السؤال" and did not reach its "ولا تحتوي" until the
# sentence after.
_REFUSAL_WINDOW = 200


def is_refusal(answer: str) -> bool:
    """True when the answer declines rather than asserting an answer.

    Only the opening is searched. A plain substring match over the whole answer
    got four Arabic answers wrong: they were well-cited and correct, but quoted
    students saying "لا يوجد", which read as the assistant declining.

    Presence of citations deliberately does NOT rule out a refusal, and cannot
    be used to distinguish one: measured across real answers, refusals cited 0,
    2, 2, 3 and 5 distinct excerpts while a substantive answer cited 2. A good
    refusal cites the excerpts to show what they do cover, so no threshold on
    citation count separates the two.

    An empty answer counts: the app renders its own 'not in the archive'
    message when retrieval returns nothing, so that path is a refusal too.
    """
    if not answer.strip():
        return True

    opening = answer.strip()[:_REFUSAL_WINDOW].lower()
    return any(marker in opening for marker in _REFUSAL_MARKERS)


def check_citations(record: dict) -> dict:
    """Verify the answer cites, and that every citation resolves to an excerpt.

    A refusal is exempt - there is nothing to cite when the archive has no
    answer, and penalising that would reward confident invention.
    """
    answer = record["answer"]
    cited = {int(n) for n in _CITATION.findall(answer)}
    available = record["source_count"]

    dangling = sorted(n for n in cited if n < 1 or n > available)

    # A claim-bearing sentence is one long enough to assert something. Short
    # fragments (list headers, "Sources:") are not claims.
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if len(s.strip()) > 40]
    uncited = [s for s in sentences if not _CITATION.search(s)]

    return {
        "refusal": is_refusal(answer),
        "has_citations": bool(cited),
        "citations": len(cited),
        "dangling": dangling,
        "sentences": len(sentences),
        "uncited_sentences": len(uncited),
    }


_JUDGE_SYSTEM = """You grade a RAG assistant's answers. You are strict and terse.

You receive the prompt the assistant was given - numbered excerpts from Saudi
students' Telegram group discussions, each headed with its authors and date -
and the answer it wrote. The assistant was told the source is student chatter,
and asked to flag old excerpts and unofficial advice, so dates taken from the
headers and caveats of that kind are supported, not invented. Score two axes
from 1 to 5:

faithfulness - is every factual claim in the answer supported by the excerpts?
  5 = fully supported. 3 = mostly supported, some unsupported detail.
  1 = contains claims the excerpts do not support.
relevance - does the answer address the question asked?
  5 = directly answers it. 1 = off topic.

An answer that correctly says the excerpts do not cover the question is
faithful (5) - declining is not a failure.

Reply with only a JSON object: {"faithfulness": n, "relevance": n, "note": "<10 words"}"""


def judge(record: dict, model: str) -> dict | None:
    """Ask Claude to grade faithfulness and relevance. Returns None on failure."""
    # The judge must see the excerpt headers the assistant saw. Given only
    # the bare text, it marked every date the assistant correctly read from a
    # header as invented - the most common complaint across a full run, and
    # one that penalised following the prompt's instruction to flag old
    # advice.
    given = record.get("prompt") or "\n\n".join(
        f"[{i}] {content}" for i, content in enumerate(record["sources"], 1)
    ) + f"\n\nQuestion: {record['question']}"
    prompt = (
        f"Given to the assistant:\n\n{given}\n\n"
        f"Answer:\n{record['answer'] or '(no answer produced)'}"
    )

    # The budget is far above the reply's ~30 tokens on purpose. Models that
    # think before answering spend output tokens doing it, and at 200 an Opus
    # judge was cut off mid-JSON on every question - each grade silently
    # dropped, leaving a --judge run that printed no scores at all.
    try:
        raw = complete(prompt, _JUDGE_SYSTEM, model=model, max_tokens=2048)
    except ClaudeError as exc:
        print(f"  ! judge unavailable ({exc})", file=sys.stderr)
        return None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    print(f"  ! judge reply had no scores: {raw[:80]!r}", file=sys.stderr)
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true", help="add the LLM judge (costs money)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="model to grade with")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    args = parser.parse_args()

    # The app retries a question briefly because someone is waiting. Nobody
    # waits on an eval, and a rate-limited call that gives up early would
    # quietly score the vector-order fallback as if it were the reranker. Set
    # here rather than at import, which the tests do.
    config.QUERY_ATTEMPTS = 6
    config.QUERY_RATE_LIMIT_DELAY = 21.0

    data = json.loads(QUESTIONS.read_text())
    answerable = data["questions"]
    unanswerable = data.get("unanswerable", [])
    if args.limit:
        answerable = answerable[: args.limit]
        unanswerable = unanswerable[: args.limit]

    pipeline = RAGPipeline()
    cache = load_cache()

    print(f"Answering {len(answerable)} answerable + {len(unanswerable)} unanswerable "
          f"questions with {config.CLAUDE_MODEL}...\n")

    cited, dangling_total, uncited_ratios, refused_wrongly = 0, 0, [], []
    faithfulness, relevance = [], []

    for item in answerable:
        record = answer_question(pipeline, item["q"], cache)
        result = check_citations(record)

        if result["refusal"]:
            refused_wrongly.append(item["q"])
        else:
            cited += result["has_citations"]
            dangling_total += len(result["dangling"])
            if result["sentences"]:
                uncited_ratios.append(result["uncited_sentences"] / result["sentences"])

        flag = "REFUSED" if result["refusal"] else (
            "no citations" if not result["has_citations"] else
            f"{result['citations']} cites"
        )
        if result["dangling"]:
            flag += f", DANGLING {result['dangling']}"
        print(f"  [{item['lang']}] {item['q'][:48]:<50} {flag}")

        if args.judge:
            scores = judge(record, args.judge_model)
            if scores:
                faithfulness.append(scores.get("faithfulness", 0))
                relevance.append(scores.get("relevance", 0))
                print(f"       faithfulness {scores.get('faithfulness')}  "
                      f"relevance {scores.get('relevance')}  {scores.get('note','')}")

    correct_refusals = 0
    if unanswerable:
        print()
        for item in unanswerable:
            record = answer_question(pipeline, item["q"], cache)
            refused = is_refusal(record["answer"])
            correct_refusals += refused
            print(f"  [{item['lang']}] {item['q'][:48]:<50} "
                  f"{'declined' if refused else 'ANSWERED ANYWAY'}")

    n = len(answerable)
    answered = n - len(refused_wrongly)

    print("\n" + "=" * 62)
    print(f"{'answerable questions':<34}{n:>8}")
    print(f"{'  answered (not refused)':<34}{answered:>8}")
    print(f"{'  with citations':<34}{cited:>8}  {cited / answered if answered else 0:>6.0%}")
    print(f"{'  dangling citations':<34}{dangling_total:>8}")
    print(f"{'  mean uncited sentences':<34}{mean(uncited_ratios):>8.0%}")
    if unanswerable:
        print(f"{'unanswerable declined':<34}{correct_refusals:>8}  "
              f"{correct_refusals / len(unanswerable):>6.0%}")
    if faithfulness:
        print(f"{'judge faithfulness (1-5)':<34}{mean(faithfulness):>8.2f}")
        print(f"{'judge relevance (1-5)':<34}{mean(relevance):>8.2f}")
    print("=" * 62)

    if refused_wrongly:
        print("\nRefused questions the archive should cover:")
        for q in refused_wrongly:
            print(f"  - {q}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
