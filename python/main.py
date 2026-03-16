import argparse
import sys
import os
from pathlib import Path

from data_parser import read_dataset, write_results
from evaluation import run_experiment

def main():
    parser = argparse.ArgumentParser(
        description="LLM Code Inference Pipeline: Evaluates models on CodeOcean or CodeTrans datasets."
    )

    # sys.argv[1] -> dataset_path
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset file (e.g., 'data/codetrans_test.csv' or 'codeocean_val.json'). "
             "The filename determines the processing logic (starts with 'codetrans' vs others)."
    )

    # sys.argv[2] -> experiment_id
    parser.add_argument(
        "experiment_id",
        type=str,
        help="The experiment identifier used to fetch prompt templates (e.g., 'm1', 'm2', or 'm3')."
    )

    args = parser.parse_args()

    # 1. Load the data
    print(f"--- Loading dataset: {args.dataset_path} ---")
    dataset = read_dataset(args.dataset_path)

    # if args.experiment_id == 'm4':
    #     # Shuffle first (same technique as m1)
    #     dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    #     # Count unique combinations of type and lang

    #     # Stratified sampling across (type, lang)
    #     dataset = (
    #         dataset
    #         .groupby(['type', 'lang'], group_keys=False)
    #         .apply(lambda x: x.iloc[0:250])
    #     )

    #     print(dataset.groupby('lang')['type'].value_counts())
    # elif args.experiment_id=='m1':
    #     # Shuffle then get head for incremental sampling
    #     dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    #     dataset = dataset.head(100)
    # else:
    #     raise NotImplementedError("Experiment not implemented")

    # 2. Determine Dataset Type
    # We use the filename to decide which prompts to pull
    filename = Path(args.dataset_path).name
    dataset_type = "codetrans" if filename.startswith("codetrans") else "codeocean"

    # 3. Prepare output
    output_name = f"{dataset_type}_{args.experiment_id}_results"
    output_path = f"./data/results/{output_name}.jsonl"
    if os.path.exists(output_path):
        choice = input(f"Output file {output_path} exists. Recreate (r) or append (a)? ").strip().lower()
        if choice == 'r':
            mode = 'w'
        elif choice == 'a':
            mode = 'a'
        else:
            print("Invalid choice. Defaulting to append.")
            mode = 'a'
    else:
        mode = 'w'

    # 4. Run Inference
    # Using the refactored run_experiment which handles all models and prompt types
    print(f"--- Starting Inference (Dataset: {dataset_type}, Exp: {args.experiment_id}) ---")
    results_df = run_experiment(
        df=dataset, 
        dataset_name=dataset_type, 
        experiment_id=args.experiment_id
    )

    # 5. Save results
    write_results(results_df, output_name, mode)
    print(f"--- Success: Results saved to {output_name}.jsonl ---")

if __name__ == "__main__":
    main()