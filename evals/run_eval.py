"""
RAG Evaluation Pipeline
========================

Evaluates the RAG system on a test dataset using three metrics:

1. **Retrieval quality** — Did we retrieve chunks from the expected source?
2. **Faithfulness** — Is the answer grounded in the retrieved context (not hallucinated)?
3. **Answer relevancy** — Does the answer actually address the question?

Metrics 2 and 3 use LLM-as-judge: we ask the same LLM to score the RAG output.
This is standard practice for evaluating LLM applications.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --dataset evals/dataset.json --output evals/results.json
"""

import argparse
import json
from datetime import UTC, datetime

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client, observe
from openai import OpenAI

from rag_medical.config import MODEL
from rag_medical.rag_chain import format_context, generate_answer, get_collection, retrieve

langfuse = get_client()

JUDGE_PROMPT_FAITHFULNESS = """You are an evaluation judge. Given a context, a question, and an answer, determine if the answer is faithful to the context.

Faithful means: the answer ONLY contains information that can be verified from the context. It does not add facts, statistics, or claims not present in the context.

Context:
{context}

Question: {question}

Answer: {answer}

Score the faithfulness from 1 to 5:
1 = Answer contains major claims not in the context (hallucinated)
2 = Answer contains some claims not supported by context
3 = Answer is mostly faithful with minor unsupported details
4 = Answer is faithful with only trivial additions
5 = Answer is fully grounded in the context

Respond with ONLY a JSON object: {{"score": <int>, "reason": "<brief explanation>"}}"""

JUDGE_PROMPT_RELEVANCY = """You are an evaluation judge. Given a question, a ground truth answer, and the system's answer, determine if the system's answer is relevant and correct.

Question: {question}

Ground truth: {ground_truth}

System answer: {answer}

Score the answer relevancy from 1 to 5:
1 = Completely irrelevant or wrong
2 = Partially relevant but missing key information
3 = Relevant but incomplete or contains minor errors
4 = Relevant and mostly complete
5 = Highly relevant, complete, and accurate

Respond with ONLY a JSON object: {{"score": <int>, "reason": "<brief explanation>"}}"""


def judge(prompt: str) -> dict:
    """Ask the LLM to judge a RAG output. Returns {"score": int, "reason": str}."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    # Parse JSON from response, handling possible markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def eval_retrieval(retrieved: list[dict], expected_sources: list[str] | None) -> dict:
    """Check if any of the expected source documents were retrieved."""
    if expected_sources is None:
        return {
            "score": 1.0,
            "retrieved_sources": [c["source"] for c in retrieved],
            "note": "No expected source (question should be unanswerable)",
        }

    sources = [c["source"] for c in retrieved]
    hit = any(s in sources for s in expected_sources)
    return {
        "score": 1.0 if hit else 0.0,
        "retrieved_sources": sources,
        "expected_sources": expected_sources,
    }


@observe(name="eval_question")
def eval_single_question(collection, item: dict, index: int, total: int) -> dict:
    """Evaluate a single question and push scores to Langfuse."""
    question = item["question"]
    ground_truth = item["ground_truth"]
    expected_sources = item.get("expected_sources")

    print(f"\n[{index}/{total}] {question}")

    # Run RAG pipeline
    retrieved = retrieve(collection, question)
    context = format_context(retrieved)
    answer = generate_answer(question, context)

    # Eval 1: Retrieval quality
    retrieval_result = eval_retrieval(retrieved, expected_sources)
    print(
        f"  Retrieval: {'HIT' if retrieval_result['score'] == 1.0 else 'MISS'} "
        f"(sources: {retrieval_result['retrieved_sources']})"
    )

    # Eval 2: Faithfulness (LLM-as-judge)
    faithfulness = judge(
        JUDGE_PROMPT_FAITHFULNESS.format(
            context=context,
            question=question,
            answer=answer,
        )
    )
    print(f"  Faithfulness: {faithfulness['score']}/5 — {faithfulness['reason']}")

    # Eval 3: Answer relevancy (LLM-as-judge)
    relevancy = judge(
        JUDGE_PROMPT_RELEVANCY.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
        )
    )
    print(f"  Relevancy:   {relevancy['score']}/5 — {relevancy['reason']}")

    # Push scores to Langfuse on this trace
    trace_id = langfuse.get_current_trace_id()
    if trace_id:
        langfuse.create_score(trace_id=trace_id, name="retrieval", value=retrieval_result["score"])
        langfuse.create_score(trace_id=trace_id, name="faithfulness", value=faithfulness["score"] / 5)
        langfuse.create_score(trace_id=trace_id, name="relevancy", value=relevancy["score"] / 5)

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "retrieval": retrieval_result,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
    }


def run_eval(dataset_path: str, output_path: str | None = None) -> dict:
    """Run the full evaluation pipeline."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    collection = get_collection()
    run_name = f"eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    print(f"Running evaluation on {len(dataset)} questions...")
    print(f"Model: {MODEL}")
    print(f"Langfuse run: {run_name}")
    print("=" * 70)

    results = []
    for i, item in enumerate(dataset, 1):
        result = eval_single_question(collection, item, i, len(dataset))
        results.append(result)

    langfuse.flush()

    # Compute summary
    n = len(results)
    summary = {
        "model": MODEL,
        "num_questions": n,
        "retrieval_accuracy": sum(r["retrieval"]["score"] for r in results) / n,
        "avg_faithfulness": sum(r["faithfulness"]["score"] for r in results) / n,
        "avg_relevancy": sum(r["relevancy"]["score"] for r in results) / n,
        "langfuse_run": run_name,
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Retrieval accuracy:  {summary['retrieval_accuracy']:.0%}")
    print(f"  Avg faithfulness:    {summary['avg_faithfulness']:.1f}/5")
    print(f"  Avg relevancy:       {summary['avg_relevancy']:.1f}/5")
    print(f"  Langfuse run:        {run_name}")
    print("=" * 70)

    output = {"summary": summary, "results": results}

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--dataset", default="evals/dataset.json", help="Path to eval dataset")
    parser.add_argument("--output", default="evals/results.json", help="Path to save results")
    args = parser.parse_args()

    run_eval(args.dataset, args.output)
