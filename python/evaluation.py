import pandas as pd
from tqdm.auto import tqdm

from prompts import prompts_templates
from model_hf import Model
from models_api import ModelApi

models_api = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-5-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
]

models_hf = [
    "codellama/CodeLlama-7b-Instruct-hf",
    "google/gemma-7b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B"
]

prompt_types = ['zero_shot_prompt', 'supervised_prompt', 'over_supervised_prompt']


def infer_ocean(dataset, experiment):
    if experiment == 1:
        df = dataset.rename(columns={'src_code': 'input', 'tgt_code': 'target'})

        return run_inference(
            df=df,
            dataset_name="codeocean",
            experiment="m1",
            extra_fields_fn=lambda r: {
                "src_lang": r["src_lang"],
                "tgt_lang": r["tgt_lang"],
            })
    elif experiment == 2:
        df = dataset[["index", "input", "target"]].rename(columns={"index": "id"})
        return run_inference(
            df=df,
            dataset_name="codeocean",
            experiment="m3",
        )
    else:
        df = dataset[["id", "name", "java_code"]].rename(
            columns={"java_code": "input", "name": "target"}
        )

        return run_inference(
            df=df,
            dataset_name="codeocean",
            experiment="m3",
        )


def infer_trans(dataset, experiment):
    if experiment == 1:
        df = dataset[['index', 'java_line', 'cs_line']].rename(
            columns={'java_line': 'input', 'cs_line': 'target', 'index': 'id'}
        )
        df['src_lang'] = 'java'
        df['tgt_lang'] = 'cs'

        return run_inference(
            df=df,
            dataset_name="codetrans",
            experiment="m1",
            extra_fields_fn=lambda r: {
                "src_lang": r["src_lang"],
                "tgt_lang": r["tgt_lang"],
            })
    elif experiment == 2:
        df = dataset[['index', 'input', 'target']].rename(columns={"index": "id"})
        return run_inference(
            df=df,
            dataset_name="codetrans",
            experiment="m3",
        )
    else:
        df = dataset[["index", "group", "input", "target"]].rename(
            columns={"index": "id"}
        )

        return run_inference(
            df=df,
            dataset_name="codetrans",
            experiment="m3",
            extra_fields_fn=lambda r: {"group": r["group"]},
        )


def run_inference(df, dataset_name, experiment, extra_fields_fn=None):
    rows = []
    records = df.to_dict(orient="records")

    for m in models_api:
        model = ModelApi(m)
        rows.extend(infer(model, m, records, dataset_name, experiment, extra_fields_fn))

    for m in models_hf:
        model = Model(m)
        try:
            rows.extend(infer(model, m, records, dataset_name, experiment, extra_fields_fn))
        finally:
            model.unload()

    return pd.DataFrame(rows)


def infer(model, model_name, records, dataset_name, experiment, extra_fields_fn=None):
    rows = []
    for r in tqdm(records, desc=f"{model_name} {dataset_name}"):
        for prompt_type in prompt_types:
            prompt = prompts_templates[dataset_name][experiment][prompt_type]

            output = model.infer(prompt.format(**r))

            # TODO: Perform some code extraction from text responses, language models
            #       tend to not return only code no matter how much they are persuaded
            #       to do so.

            row = {
                "id": r["id"],
                "model": model_name,
                "prompting": prompt_type,
                "target": r["target"],
                "output": output,
            }

            if extra_fields_fn:
                row.update(extra_fields_fn(r))

            rows.append(row)
    return rows
