import json
import time
import re
import numpy as np
from openai import OpenAI
from pipelines.rag_pipeline import query_rag

import ollama


DATASET_FILE = "evaluation_dataset.json"
RESULT_FILE = "evaluation_results.json"


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

def load_dataset():

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------
# ANSWER EVALUATION
# ---------------------------------------------------

import ollama

def judge_answer(expected, predicted):

    prompt = f"""
You are evaluating a question answering system.

Expected answer:
{expected}

Predicted answer:
{predicted}

If the predicted answer contains the same meaning or correct value,
return ONLY YES.

Otherwise return ONLY NO.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"].strip().lower()

    if "yes" in answer:
        return "PASS"
    else:
        return "FAIL"

# ---------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------

def evaluate():

    dataset = load_dataset()

    results = []

    total_questions = len(dataset)
    correct_answers = 0

    print("\nStarting RAG evaluation")
    print("Total questions:", total_questions)

    for i, item in enumerate(dataset, 1):

        question = item["question"]
        expected = item["answer"]

        try:

            rag_result = query_rag(question)

            predicted = rag_result["answer"]
            sources = rag_result["sources"]

            status = judge_answer(expected, predicted)

            if status == "PASS":
                correct_answers += 1

            results.append({

                "question": question,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "status": status,
                #"similarity_score": score,
                "sources": sources
            })

        except Exception as e:

            results.append({

                "question": question,
                "expected_answer": expected,
                "predicted_answer": "ERROR",
                "status": "FAIL",
                "error": str(e),
                "sources": []
            })

        time.sleep(0.3)

    accuracy = correct_answers / total_questions

    summary = {

        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": round(accuracy * 100, 2)
    }

    output = {

        "summary": summary,
        "results": results
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nEvaluation complete")
    print("Accuracy:", summary["accuracy"], "%")
    print("Results saved to:", RESULT_FILE)


# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":
    evaluate()