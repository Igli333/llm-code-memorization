import pandas as pd

from prompts.prompts import prompts_templates
from model_hf import Model

# TODO: check names in documentation
models = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-5-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "codellama-7b",  # and above maybe
    "Llama-3.1",
    "Qwen-3-8b"
]

prompt_types = ['zero_shot', 'supervised', 'over_supervised']


def infer_ocean(dataset, experiment):
    if experiment == 1:
        df = dataset.rename(columns={'src_code': 'input', 'tgt_code': 'target'})

        return run_inference(
            df=df,
            dataset_name="CodeOcean",
            experiment="m1",
            extra_fields_fn=lambda r: {
                "src_lang": r["src_lang"],
                "tgt_lang": r["tgt_lang"],
            })
    elif experiment == 2:
        df = dataset[["index", "input", "target"]].rename(columns={"index": "id"})
        return run_inference(
            df=df,
            dataset_name="CodeOcean",
            experiment="m3",
        )
    else:
        df = dataset[["id", "name", "java_code"]].rename(
            columns={"java_code": "input", "name": "target"}
        )

        return run_inference(
            df=df,
            dataset_name="CodeOcean",
            experiment="m3",
        )


def infer_trans(dataset, experiment):
    if experiment == 1:
        df = dataset[['id', 'java_line', 'cs_line']].rename(
            columns={'java_line': 'input', 'cs_line': 'target', 'id': 'index'}
        )
        df['src_lang'] = 'java'
        df['tgt_lang'] = 'cs'

        return run_inference(
            df=df,
            dataset_name="CodeTrans",
            experiment="m1",
            extra_fields_fn=lambda r: {
                "src_lang": r["src_lang"],
                "tgt_lang": r["tgt_lang"],
            })
    elif experiment == 2:
        df = dataset[['index', 'input', 'target']].rename(columns={"index": "id"})
        return run_inference(
            df=df,
            dataset_name="CodeTrans",
            experiment="m3",
        )
    else:
        df = dataset[["index", "group", "input", "target"]].rename(
            columns={"index": "id"}
        )

        return run_inference(
            df=df,
            dataset_name="CodeTrans",
            experiment="m3",
            extra_fields_fn=lambda r: {"group": r["group"]},
        )


def run_inference(df, dataset_name, experiment, extra_fields_fn=None):
    rows = []
    records = df.to_dict(orient="records")

    for m in models:
        model = Model(m)
        try:
            for r in records:
                for prompt_type in prompt_types:
                    prompt = prompts_templates[dataset_name][experiment][prompt_type]
                    # TODO: Make sure to check for m3 if there are additional fields needed
                    output = model.infer(prompt.format(r["input"]))

                    row = {
                        "id": r["id"],
                        "model": m,
                        "prompting": prompt_type,
                        "target": r["target"],
                        "output": output,
                    }

                    if extra_fields_fn:
                        row.update(extra_fields_fn(r))

                    rows.append(row)
        finally:
            model.unload()

    return pd.DataFrame(rows)
