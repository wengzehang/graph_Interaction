"""
Generate diagrams from the evalution data
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple

import csv
import os

eval_path = "./evaluation/first-dataset"
plot_stddev_whiskers = False

sets = [
    {"id": "train", "name": "Training"},
    {"id": "valid", "name": "Validation"},
    {"id": "test", "name": "Test"}
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
            return None, None

        with open(full_path) as file:
            reader = csv.reader(file, delimiter=',', quotechar='"')
            rows = [row for row in reader]
            # The first row contains the column names, so skip it
            values = np.array(rows[1:][0], np.float32)
            mean_errors[i] = values[0]
            stddevs[i] = values[1]

    return mean_errors, stddevs


def save_error_plot(filename):
    fig, ax = plt.subplots()
    ax.set_ylabel('Mean Position Error')

    model_names = [model['name'] for model in models]
    x_pos = np.arange(len(model_names))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title(f"Single Frame Prediction: Mean Position Error")

    for i, set in enumerate(sets):
        mean_errors, stddevs = load_error_stats(set, models)
        if mean_errors is None:
            continue

        pos = x_pos + (i-1) * 0.25

        yerr = stddevs if plot_stddev_whiskers else None
        ax.bar(pos, mean_errors, width=0.25, yerr=yerr, align='center', alpha=0.5, ecolor='black', capsize=10)

    ax.set_ylim(0)

    plt.tight_layout()
    plt.savefig(filename)


def load_horizon_stats(set: dict, models: list) -> np.array:
    # Load errors
    errors = [None] * len(models)
    for i, model in enumerate(models):
        filename = f"eval_horizon_{set['id']}_{model['id']}.csv"
        full_path = os.path.join(eval_path, filename)
        if not os.path.exists(full_path):
            print("Could not open file:", full_path)
            return None

        with open(full_path) as file:
            reader = csv.reader(file, delimiter=',', quotechar='"')
            rows = [row[0] for row in reader]
            # The first row contains the column names
            # We set it to zero (0-frame prediction has 0 error)
            rows[0] = 0.0
            errors[i] = rows

    return np.array(errors, np.float32)


def save_horizon_plot(set: dict, filename: str):
    errors = load_horizon_stats(set, models)
    if errors is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylabel('Mean Position Error')

    x_pos = np.arange(errors.shape[1])

    model_names = [model['name'] for model in models]

    ax.set_xticks(x_pos)
    ax.set_title(f"Horizon Prediction: Mean Position Error ({set['name']})")

    for i, frame_errors in enumerate(errors):
        ax.plot(x_pos, frame_errors, label=model_names[i])

    ax.legend()
    ax.set_ylim(0)
    ax.set_xlim(0, x_pos[-1])

    plt.tight_layout()
    plt.savefig(filename)


plot_path = os.path.join(eval_path, "plot_error_bars.png")
save_error_plot(plot_path)

for set in sets:
    plot_path = os.path.join(eval_path, f"plot_horizon_bars_{set['id']}.png")
    save_horizon_plot(set, plot_path)

plt.show()





