import json
import time
from pathlib import Path
from evaluators.engine import RAGEvaluator
from evaluators.reporter import BenchmarkReporter

def run_benchmark(
    dateset_path: str = "data/golden_dataset.json",
    reports_dir: str = "reports",
    export_json: bool = True,
    export_csv: bool = True,
):
    """
    Executes a bulk evaluation benchmark over  a dataset and produces aggregate summary metrics.    
    """
    path = Path(dateset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dateset_path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    evaluator = RAGEvaluator()
    reporter = BenchmarkReporter(reports_dir=reports_dir)
    results = []
    start_time = time.time()

    print(f"\nStarting bulk evaluation benchmark on {len(dataset)} items...")
    print("=" * 80)

    total_faith = 0.0
    total_rel = 0.0
    passed_gates = 0

    for item in dataset:
        item_id = item.get("id", "N/A")
        topic = item.get("topic", "General")
        query = item.get("query", "")
        context = item.get("context", "")
        response = item.get("response", "")

        # Evaluate Faithfulness & Relevancy
        faith = evaluator.evaluate_faithfulness(query=query, context=context, response=response)
        rel = evaluator.evaluate_relevancy(query=query, response=response)

        # Gate pass criteria: Both scores meet 0.85 threshold
        passed = faith.score >= 0.85 and rel.score >= 0.85
        if passed:
            passed_gates += 1

        total_faith += faith.score
        total_rel += rel.score

        status_symbol = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{item_id}] {topic:<32} | Faith: {faith.score:.2f} | Rel: {rel.score:.2f} | {status_symbol}")

        results.append({
            "id": item_id,
            "topic": topic,
            "query": query,
            "context": context,
            "reponse": response,
            "faithfulness_score": faith.score,
            "faithfulness_reasoning": faith.reasoning,
            "unsupported_claims": faith.unsupported_claims,
            "relevancy_score": rel.score,
            "relevancy_reasoning": rel.reasoning,
            "missing_aspects": rel.missing_aspects,
            "passed_gate": passed
        })
    elapsed_time = time.time() - start_time
    total_items = len(dataset)
    avg_faith = total_faith / total_items if total_items > 0 else 0.0
    avg_rel = total_rel / total_items if total_items > 0 else 0.0
    pass_rate = (passed_gates / total_items) * 100 if total_items > 0 else 0.0

    reporter.print_summary_report(
        total_items=total_items,
        passed_gates=passed_gates,
        pass_rate=pass_rate,
        avg_faith=avg_faith,
        avg_rel=avg_rel,
        elapsed_time=elapsed_time
    )

    reporter.export_artifacts(
        results=results,
        total_items=total_items,
        pass_rate=pass_rate,
        avg_faith=avg_faith,
        avg_rel=avg_rel,
        elapsed_time=elapsed_time,
        export_json=export_json,
        export_csv=export_csv
    )

    return results

if __name__ == "__main__":
    run_benchmark()
