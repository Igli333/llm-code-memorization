import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Any

# Assuming these are your custom modules
from prompts import prompts_templates
# from model_hf import Model
from models_api import ModelApi

# Configuration
MODELS_API = [
     #"gpt-3.5-turbo", 
    # "gpt-4o-mini", 
    "gpt-5.2",
    "openrouter/openai/gpt-oss-120b",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
    "openrouter/openai/gpt-5-mini",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/google/gemini-3-flash-preview",
    "openrouter/qwen/qwen3-coder", 
    # "gemini-2.5-pro", 
    # "gemini-2.5-flash"
]

MODELS_HF = [
    # "codellama/CodeLlama-7b-Instruct-hf",
    # "google/gemma-7b-it",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "Qwen/Qwen3-8B"
]

PROMPT_TYPES = ['supervised_prompt']


def run_experiment(
    df: pd.DataFrame, 
    dataset_name: str, 
    experiment_id: str
) -> pd.DataFrame:
    """
    Runs inference across all models. Preserves all original columns from df.
    """
    results = []
    records = df.to_dict(orient="records")

    # 1. API-based Models
    for model_name in MODELS_API:
        model_engine = ModelApi(model_name)
        results.extend(_infer_loop(model_engine, model_name, records, dataset_name, experiment_id))

    # 2. HuggingFace Models (with automated memory cleanup)
    for model_name in MODELS_HF:
        model_engine = Model(model_name)
        try:
            results.extend(_infer_loop(model_engine, model_name, records, dataset_name, experiment_id))
        finally:
            model_engine.unload()

    return pd.DataFrame(results)


def _infer_loop(
    model_engine: Any,
    model_name: str,
    records: List[Dict],
    dataset_name: str,
    experiment_id: str
) -> List[Dict]:
    """
    Internal helper to iterate through records and prompt types.
    """
    rows = []
    
    for record in tqdm(records, desc=f"[{dataset_name}] {model_name}"):
        for p_type in PROMPT_TYPES:
            # Retrieve template
            template = prompts_templates[dataset_name][experiment_id][p_type]
            
            # Format using raw record keys
            prompt_text = template.format(**record)

            # Direct inference (No extraction)
            output = model_engine.infer(prompt_text)

            # Create row preserving all original columns + new metadata
            row = {
                **record, 
                "model_name": model_name,
                "prompt_type": p_type,
                "model_output": output
            }
            rows.append(row)
            
    return rows
