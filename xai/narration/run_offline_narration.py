import argparse
import glob
import json
import os
import yaml

from xai.narration import narration_router, prompt_builder, llm_narrator

import time

def run_offline_narration(reports_dir: str, config_path: str):
    """
    Reads all JSON reports in the given directory, generates LLM narrations
    and overwrites the JSON file, saving the response time.
    """
    if not os.path.isdir(reports_dir):
        print(f"❌ Directory not found: {reports_dir}")
        return

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config {config_path}: {e}")
        return

    report_files = sorted(glob.glob(os.path.join(reports_dir, "*.json")))
    print(f"Found {len(report_files)} reports in {reports_dir}.")

    processed = 0

    for file_path in report_files:
        try:
            with open(file_path, "r") as f:
                report = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse JSON: {file_path}")
            continue

        print(f"Narrating {os.path.basename(file_path)}...")

        # 1. Route to get template
        template_key, _ = narration_router.route(report)

        # 2. Build Prompt
        system_prompt, user_prompt = prompt_builder.build_prompt(template_key, report)

        # 3. Call LLM
        start_time = time.time()
        narration = llm_narrator.narrate(system_prompt, user_prompt, config)
        response_time = time.time() - start_time

        # 4. Update Report
        report["narration"] = narration
        report["narration_response_time_s"] = round(response_time, 2)
        print(f"    -> Done in {response_time:.4f}s")

        # 5. Save back
        with open(file_path, "w") as f:
            json.dump(report, f, indent=4)
        
        processed += 1

    print(f"\n✅ Offline Narration Complete!")
    print(f"Processed: {processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Narration on saved XAI reports.")
    parser.add_argument(
        "--reports_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing report_*.json files."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="xai/config/xai_config.yaml", 
        help="Path to xai_config.yaml"
    )
    args = parser.parse_args()

    run_offline_narration(args.reports_dir, args.config)
