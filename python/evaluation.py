import pandas as pd

from model_hf import Model

models = []


def infer_ocean(dataset, experiment):
    if experiment == 1:
        df = dataset.rename(columns={'src_code': 'input', 'tgt_code': 'target'})
        output = adversarial_prompting(df, "CodeOcean")
    elif experiment == 2:
        df = dataset[['input', 'target']]
        output = df.copy()
        for m in models:
            model = Model(m)
            output[m] = model.infer('<prompt>' + output['input'])
            model.unload()
    else:
        df = dataset[['name', 'java_code']]
        output = df.copy()
        for m in models:
            model = Model(m)
            output[m] = model.infer('<prompt>' + output['name'])
            model.unload()

    return output


def infer_trans(dataset, experiment):
    if experiment == 1:
        df = dataset[['java_line', 'cs_line']]
        df = df.rename(columns={'java_line': 'input', 'cs_line': 'target'})
        df['src_lang'] = 'java'
        df['tgt_lang'] = 'cs'
        output = adversarial_prompting(df, "CodeTrans")
    elif experiment == 2:
        df = dataset[['input', 'target']]
        output = df.copy()
        for m in models:
            model = Model(m)
            output[m] = model.infer('<prompt>' + output['input'])
            model.unload()
    else:
        df = dataset[['group', 'input', 'target']]
        output = df.copy()
        for m in models:
            model = Model(m)
            output[m] = model.infer('<prompt>' + output['input'])
            model.unload()
    return output


def adversarial_prompting(df, dataset):
    # Do we need input here? Or even target, could just save an id for reference
    output = pd.DataFrame(columns=['model', 'prompting', 'src_lang', 'tgt_lang', 'input', 'target', 'output'])
    prompt_types = ['zero_shot', 'supervised', 'over_supervised']

    input_dict = df.T.to_dict.values()

    for m in models:
        model = Model(m)  # to be specified further when the models are chosen

        for i in input_dict:
            input_code = i['input']
            for prompt_type in prompt_types:
                o = model.infer(prompting(prompt_type, input_code, dataset))
                r = pd.DataFrame({
                    'model': m,
                    'prompting': prompt_type,
                    'src_lang': i['src_lang'],
                    'tgt_lang': i['tgt_lang'],
                    'input': input_code,
                    'target': i['target'],
                    'output': o
                })

                output = pd.concat([output, r])

        model.unload()

    return output


def prompting(prompting_type, code, dataset):
    if prompting_type == 'zero_shot':
        return '<prompt>' + code
    elif prompting_type == 'supervised':
        return '<prompt>' + code
    else:
        # use dataset param here
        return '<prompt>' + code
