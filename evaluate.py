import json
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipelines.rag_pipeline import query_rag
import ollama


DATASET_FILE  = "evaluation_dataset.json"
RESULTS_DIR   = "evaluation_results"
JUDGE_WORKERS = 2   # Ollama judge can still run in parallel

# Seconds to wait between RAG calls — tune based on your OpenAI tier:
# Tier 1 → 5.0, Tier 2 → 3.0, Tier 3+ → 1.0
RAG_DELAY_SECONDS = 5.0


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------
# RANDOM SAMPLING
# ---------------------------------------------------

def sample_questions(dataset, sample_size=30):
    sampled = random.sample(dataset, sample_size)
    print(f"Randomly selected {sample_size} questions out of {len(dataset)} total")
    return sampled, sample_size


# ---------------------------------------------------
# PHASE 1 — RAG RETRIEVAL (sequential — avoids OpenAI 429s)
# ---------------------------------------------------

def fetch_rag(item):
    try:
        rag_result = query_rag(item["question"])
        return {
            "id": item.get("id"),
            "question": item["question"],
            "expected": item["answer"],
            "predicted": rag_result["answer"],
            "sources": rag_result["sources"],
            "error": None
        }
    except Exception as e:
        return {
            "id": item.get("id"),
            "question": item["question"],
            "expected": item["answer"],
            "predicted": "ERROR",
            "sources": [],
            "error": str(e)
        }


def run_rag_phase(sampled_dataset, sample_size):

    print(f"\n⏳ Phase 1: Fetching RAG answers (sequential, {RAG_DELAY_SECONDS}s delay)...")
    rag_results = []

    for i, item in enumerate(sampled_dataset, 1):
        result = fetch_rag(item)
        rag_results.append(result)
        print(f"  RAG [{i:02d}/{sample_size}] {result['question'][:70]}...")
        if i < sample_size:
            time.sleep(RAG_DELAY_SECONDS)  # wait between calls to respect rate limit

    print(f"✅ Phase 1 complete\n")
    return rag_results


# ---------------------------------------------------
# PHASE 2 — JUDGE EVALUATION (parallel, Ollama bound)
# ---------------------------------------------------

def judge_answer(item):

    if item["error"]:
        return "FAIL"

    prompt = f"""
You are evaluating a question answering system.

Expected answer:
{item['expected']}

Predicted answer:
{item['predicted']}

Rules:
- NUMBERS, PERCENTAGES, and CURRENCY VALUES must match exactly. For example, 16.43 and 15.71 are different — that is WRONG. But "$2.4 billion" and "USD 2.4 billion" are the same — that is CORRECT.
- The predicted answer may be a longer sentence or differently phrased — that is fine as long as the core value or fact is correct.
- If the answer is YES/NO, the predicted must agree with the expected direction.
- If the answer is a name or label, it must refer to the same entity.

Return ONLY YES if the predicted answer is correct.
Return ONLY NO if the numeric value or fact is wrong.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"].strip().lower()
    return "PASS" if "yes" in answer else "FAIL"


def run_judge_phase(rag_results, sample_size):

    print(f"⏳ Phase 2: Judging answers ({JUDGE_WORKERS} parallel workers)...")
    final_results = [None] * sample_size
    completed = 0

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as executor:
        future_to_index = {executor.submit(judge_answer, item): i for i, item in enumerate(rag_results)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            item = rag_results[index]
            status = future.result()
            completed += 1

            status_icon = "✓" if status == "PASS" else "✗"
            print(f"  Judge [{completed:02d}/{sample_size}] {status_icon}  {item['question'][:70]}...")

            final_results[index] = {
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected"],
                "predicted_answer": item["predicted"],
                "status": status,
                "sources": item["sources"],
                **({"error": item["error"]} if item["error"] else {})
            }

    print(f"✅ Phase 2 complete\n")
    return final_results


# ---------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------

def get_next_run_number():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RESULTS_DIR) if f.startswith("evaluation_results_run") and f.endswith(".json")]
    return len(existing) + 1


def evaluate():

    dataset = load_dataset()
    sampled_dataset, sample_size = sample_questions(dataset)

    run_number  = get_next_run_number()
    result_file = os.path.join(RESULTS_DIR, f"evaluation_results_run{run_number}.json")

    start_time = time.time()
    print(f"\nStarting RAG evaluation — Run {run_number}")

    rag_results   = run_rag_phase(sampled_dataset, sample_size)
    final_results = run_judge_phase(rag_results, sample_size)

    elapsed = round(time.time() - start_time, 1)

    correct_answers = sum(1 for r in final_results if r["status"] == "PASS")
    accuracy = correct_answers / sample_size

    summary = {
        "run_number": run_number,
        "total_questions_in_dataset": len(dataset),
        "questions_sampled": sample_size,
        "correct_answers": correct_answers,
        "accuracy": round(accuracy * 100, 2),
        "time_taken_seconds": elapsed
    }

    output = {
        "summary": summary,
        "results": final_results
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Run number    : {run_number}")
    print(f"Correct       : {correct_answers}/{sample_size}")
    print(f"Accuracy      : {summary['accuracy']}%")
    print(f"Time taken    : {elapsed}s")
    print(f"Results saved : {result_file}")


# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":
    evaluate()