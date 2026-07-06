"""CSV report generation for traffic survey results."""

import csv
from pathlib import Path

import pandas as pd


class ReportGenerator:
    """Generate summary and detailed CSV reports."""

    def generate_report(self, speed_estimator, wrong_side_detector, counter, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for track_id, history in speed_estimator.positions.items():
            if not history:
                continue
            speed_values = speed_estimator.speeds.get(track_id, [])
            avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0
            max_speed = max(speed_values) if speed_values else 0.0
            rows.append({
                "Vehicle ID": track_id,
                "Vehicle Type": "Unknown",
                "Max Speed": round(max_speed, 2),
                "Average Speed": round(avg_speed, 2),
                "Overspeed": int(max_speed > 60),
                "Wrong Side": int(track_id in wrong_side_detector.wrong_side_ids),
                "Time": history[-1][1],
                "Direction": "unknown",
            })

        if rows:
            pd.DataFrame(rows).to_csv(output, index=False)
        else:
            pd.DataFrame(columns=[
                "Vehicle ID",
                "Vehicle Type",
                "Max Speed",
                "Average Speed",
                "Overspeed",
                "Wrong Side",
                "Time",
                "Direction",
            ]).to_csv(output, index=False)

        summary = {
            "Total Vehicles": counter.total_count,
            "Cars": counter.vehicle_count["Car"],
            "Motorcycles": counter.vehicle_count["Motorcycle"],
            "Bus": counter.vehicle_count["Bus"],
            "Truck": counter.vehicle_count["Truck"],
            "Wrong Side Count": len(wrong_side_detector.wrong_side_events),
            "Overspeed Count": len(speed_estimator.overspeed_events),
        }
        summary_path = output.with_name("summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
