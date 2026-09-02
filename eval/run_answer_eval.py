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
              --judge and defaults to Haiku.

    python eval/run_answer_eval.py                 # free checks only
    python eval/run_answer_eval.py --judge         # adds the LLM judge
    python eval/run_answer_eval.py --limit 5       # a quick subset

Answers are cached to disk keyed by question and model, so re-running to add
the judge does not pay for generation twice. Delete eval/.answer_cache.json to
force regeneration after a prompt or retrieval change.

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
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from murshid import config  # noqa: E402
from murshid.claude import ClaudeError, complete  # noqa: E402
from murshid.rag import RAGPipeline  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.json"
CACHE = Path(__file__).parent / ".answer_cache.json"

JUDGE_MODEL = "claude-haiku-4-5-20251001"

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
)


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def answer_question(pipeline: RAGPipeline, question: str, cache: dict) -> dict:
    """Run the real pipeline end to end, reusing a cached answer when present."""
    key = f"{config.CLAUDE_MODEL}:{config.VOYAGE_MODEL}:{question}"
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

    Presence of citations deliberately does NOT rule out a refusal - a good
    refusal often cites the excerpts to show what they do cover instead.

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

You receive numbered excerpts and an answer written from them. Score two axes
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
    excerpts = "\n\n".join(
        f"[{i}] {content}" for i, content in enumerate(record["sources"], 1)
    )
    prompt = (
        f"Excerpts:\n\n{excerpts}\n\n"
        f"Question: {record['question']}\n\n"
        f"Answer:\n{record['answer'] or '(no answer produced)'}"
    )

    try:
        raw = complete(prompt, _JUDGE_SYSTEM, model=model, max_tokens=200)
    except ClaudeError as exc:
        print(f"  ! judge unavailable ({exc})", file=sys.stderr)
        return None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true", help="add the LLM judge (costs money)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="model to grade with")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    args = parser.parse_args()

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
