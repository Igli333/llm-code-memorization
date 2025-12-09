import sys

from data_parser import read_dataset, write_results
from evaluation import infer_ocean, infer_trans


def run(data, experiment):
    dataset = read_dataset(data)

    if data.startswith('codetrans'):
        results = infer_trans(dataset, experiment)
    else:
        results = infer_ocean(dataset, experiment)

    write_results(results, data + "_" + experiment)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
