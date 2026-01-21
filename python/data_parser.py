import pandas as pd
import json

def read_dataset(name):
    return pd.read_csv('./data/datasets/' + name + '.csv')

def write_results(results, name, mode='w'):
    path = './data/results/'
    if mode == 'a':
        with open(path + name + '.jsonl', 'a') as f:
            for record in results.to_dict('records'):
                f.write(json.dumps(record) + '\n')
    else:
        results.to_json(path + name + '.jsonl', orient="records", lines=True)