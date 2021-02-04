"""
Generate diagrams from the evalution data
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple

import csv
import os

eval_path = "./evaluation/first-dataset"

sets = [
    {"id": "train", "name": "Training"},
    {"id": "valid", "name": "Validation"}
]

models = [
    {"id": "one-stage", "name": "One-step"},
    {"id": "two-stage", "name": "Two-step"},
    {"id": "horizon", "name": "Horizon"}
]


def load_error_stats(set: dict, models: list) -> Tuple[np.array, np.array]:
    # Load errors
    mean_errors = np.zeros(len(models))
    stddevs = np.zeros(len(models))
    for i, model in enumerate(models):
        filename = f"eval_error_{set['id']}_{model['id']}.csv"
        full_path = os.path.join(eval_path, filename)
        if not os.path.exists(full_path):
            print("Could not open file:", full_path)

        with open(full_path) as file:
            reader = csv.reader(file, delimiter=',', quotechar='"')
            rows = [row for row in reader]
            # The first row contains the column names, so skip it
            values = np.array(rows[1:][0], np.float32)
            mean_errors[i] = values[0]
            stddevs[i] = values[1]

    return mean_errors, stddevs


for set in sets:
    # Load errors
    mean_errors, stddevs = load_error_stats(set, models)

    print("Mean Error:", mean_errors)
    print("Stddev:", stddevs)

    model_names = [model['name'] for model in models]
    x_pos = np.arange(len(model_names))
    fig, ax = plt.subplots()
    ax.bar(x_pos, mean_errors, yerr=stddevs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Predicted Position Error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title('Predicted Position Error')
    ax.yaxis.grid(True)

    # Save the figure and show
    # TODO: Save to a different file for every set
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()





