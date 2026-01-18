import pandas as pd

def read_dataset(name):
    return pd.read_csv('./data/datasets/' + name + '.csv')

def write_results(results, name):
    path = './data/results/'
    results.to_json(path + name + '.jsonl', orient="records", lines=True)