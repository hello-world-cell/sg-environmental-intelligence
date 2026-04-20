"""
Singapore Environmental Intelligence Pipeline
Orchestrates the full ETL + quality + recommendation pipeline.
"""

from src.extract import extract
from src.transform import transform
from src.quality import quality
from src.recommend import recommend


def run_pipeline():
    """Run the end-to-end pipeline."""
    print("Starting Singapore Environmental Intelligence Pipeline...")

    # Step 1: Extract raw environmental data
    raw_data = extract.run()

    # Step 2: Transform into processed/curated datasets
    processed_data = transform.run(raw_data)

    # Step 3: Run data quality checks
    quality.run(processed_data)

    # Step 4: Generate recommendations
    recommendations = recommend.run(processed_data)

    print("Pipeline complete.")
    return recommendations


if __name__ == "__main__":
    run_pipeline()
