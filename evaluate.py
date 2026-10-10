import json
import time
from rag import load_and_index_documents, process_query, clear_vector_store

def run_evaluation():
    # Setup
    clear_vector_store()
    docs, chunks, vs = load_and_index_documents()

    with open("eval_questions.json", "r") as f:
        questions = json.load(f)

    report = {
        "total": len(questions),
        "correct": 0,
        "fallback_correct": 0,
        "total_response_time": 0,
        "results": []
    }

    print(f"Running evaluation on {len(questions)} questions...")

    for q in questions:
        start_time = time.time()
        result = process_query(q["question"], chunks, vs)
        response_time = time.time() - start_time

        answer = result["answer"]

        # Check correctness
        is_correct = False
        if q["in_docs"]:
            # Check if answer contains expected keywords
            if any(key.lower() in answer.lower() for key in q["expected_keywords"]):
                is_correct = True
        else:
            # Check if answer is exactly the fallback
            if "I don't have that information" in answer:
                is_correct = True
                report["fallback_correct"] += 1

        if is_correct:
            report["correct"] += 1

        report["total_response_time"] += response_time
        report["results"].append({
            "question": q["question"],
            "correct": is_correct,
            "response_time": round(response_time, 3)
        })

    # Final report
    avg_time = report["total_response_time"] / report["total"]
    accuracy = (report["correct"] / report["total"]) * 100

    report_md = f"""# Evaluation Report

- Total Questions: {report["total"]}
- Correct Answers: {report["correct"]}
- Accuracy: {accuracy:.2f}%
- Average Response Time: {avg_time:.3f} seconds

## Detailed Results

| Question | Correct | Response Time (s) |
|----------|---------|-------------------|
"""
    for res in report["results"]:
        report_md += f"| {res['question']} | {'✅' if res['correct'] else '❌'} | {res['response_time']} |\n"

    with open("eval_report.md", "w") as f:
        f.write(report_md)

    print("Evaluation complete. Report generated: eval_report.md")

if __name__ == "__main__":
    run_evaluation()