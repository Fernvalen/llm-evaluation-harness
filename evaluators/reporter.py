import csv
import json
from datetime import datetime
from pathlib import Path

class BenchmarkReporter:
    """
    Handles report generation, output formatting,
    and artifact exports for evaluation benchmark runs.
    """
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)

    def print_summary_report(
            self,
            total_items: int,
            passed_gates: int,
            pass_rate: float,
            avg_faith: float,
            avg_rel: float,
            elapsed_time: float
    ):
        """Prints a clean ASCII summary block to the terminal console."""
        print("=" * 80)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 80)
        print(f"• Total Test Cases:        {total_items}")
        print(f"• Quality Gate Pass Rate:  {pass_rate:.1f}% ({passed_gates}/{total_items})")
        print(f"• Mean Faithfulness:       {avg_faith:.2f}")
        print(f"• Mean Relevancy:          {avg_rel:.2f}")
        print(f"• Execution Time:          {elapsed_time:.2f}s")
        print("=" * 80)

    def export_artifacts(
        self,
        results: list[dict],
        total_items: int,
        pass_rate: float,
        avg_faith: float,
        avg_rel: float,
        elapsed_time: float,
        export_json: bool = False, # User toggle for JSON file
        export_csv:bool = False   # User toggle for CSV file
    ):
        """Generates timestamped JSON and CSV execution artifacts in the reports directory."""
        if not (export_json or export_csv):
            print("Artifact export skipped per user configuration.")
            return
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON Artifact Generation
        if export_json:
            json_path = self.reports_dir / f"benchmark_report_{timestamp}.json"
            report_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "total_test_cases": total_items,
                    "pass_rate_pct": round(pass_rate, 2),
                    "mean_faithfulness": round(avg_faith, 2),
                    "mean_relevancy": round(avg_rel, 2),
                    "execution_time_seconds": round(elapsed_time, 2), 
                },
                "results": results
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"JSON Artifact saved: {json_path}\n")

        # CSV Artifact Generation
        if export_csv:
            csv_path = self.reports_dir / f"benchmark_report_{timestamp}.csv"
            csv_headers = [
                "id",
                "topic",
                "passed_gate",
                "faithfulness_score",
                "relevancy_score",
                "unsupported_claims_count",
                "missing_aspects_count"
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "id": r["id"],
                        "topic": r["topic"],
                        "passed_gate": r["passed_gate"],
                        "faithfulness_score": r["faithfulness_score"],
                        "relevancy_score": r["relevancy_score"],
                        "unsupported_claims_count": len(r.get("unsupported_claims", [])),
                        "missing_aspects_count": len(r.get("missing_aspects",[]))
                    })
            print(f"CSV Artifact saved: {csv_path}\n")

