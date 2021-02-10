"""
Generate diagrams from the evalution data
"""

import Datasets

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple
import argparse
import csv
import os


def load_error_stats(eval_path: str, subset: Datasets.Subset, models: list) -> Tuple[np.array, np.array]:
    # Load errors
    mean_errors = np.zeros(len(models))
    stddevs = np.zeros(len(models))
    for i, model in enumerate(models):
        filename = f"error_{subset.filename()}_{model['id']}.csv"
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


def save_error_plot(eval_path: str, filename: str):
    fig, ax = plt.subplots()
    ax.set_ylabel('Mean Position Error')

    model_names = [model['name'] for model in models]
    x_pos = np.arange(len(model_names))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title(f"Single Frame Prediction: Mean Position Error")

    for i, set in enumerate(Datasets.Subset):
        mean_errors, stddevs = load_error_stats(eval_path, set, models)
        if mean_errors is None:
            continue

        pos = x_pos + (i-1) * 0.25

        yerr = stddevs if plot_stddev_whiskers else None
        ax.bar(pos, mean_errors, width=0.25, yerr=yerr, align='center', alpha=0.5, ecolor='black', capsize=10)

    ax.set_ylim(0)

    plt.tight_layout()
    path = os.path.join(eval_path, filename)
    plt.savefig(path)
    print("Saved:", path)


def load_horizon_stats(eval_path: str, subset: Datasets.Subset, models: list) -> np.array:
    # Load errors
    errors = [None] * len(models)
    for i, model in enumerate(models):
        filename = f"horizon_{subset.filename()}_{model['id']}.csv"
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


def save_horizon_plot(eval_path: str, subset: Datasets.Subset, filename: str):
    errors = load_horizon_stats(eval_path, subset, models)
    if errors is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylabel('Mean Position Error')

    x_pos = np.arange(errors.shape[1])

    model_names = [model['name'] for model in models]

    ax.set_xticks(x_pos)
    ax.set_title(f"Horizon Prediction: Mean Position Error ({subset.name()})")

    for i, frame_errors in enumerate(errors):
        ax.plot(x_pos, frame_errors, label=model_names[i])

    ax.legend()
    ax.set_ylim(0)
    ax.set_xlim(0, x_pos[-1])

    plt.tight_layout()
    path = os.path.join(eval_path, filename)
    plt.savefig(path)
    print("Saved:", path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a prediction model for deformable bag manipulation')
    parser.add_argument('--plot_stddev_whiskers', type=bool, default=False)

    args, _ = parser.parse_known_args()

    plot_stddev_whiskers = args.plot_stddev_whiskers

    models = [
        {"id": "one-stage", "name": "One-step"},
        {"id": "two-stage", "name": "Two-step"},
        {"id": "horizon", "name": "Horizon"}
    ]

    for task in Datasets.tasks:
        print("Creating evaluation diagrams for task:", task.index)

        # Use a separate path to store the models for each task
        eval_path = f"./models/task-{task.index}/evaluation"

        if not os.path.exists(eval_path):
            print(f"No evaluation directory found for task {task.index}: {eval_path}")
            continue

        plot_filename = "plot_error_bars.png"
        save_error_plot(eval_path, plot_filename)

        for subset in Datasets.Subset:
            plot_filename = f"plot_horizon_bars_{subset.filename()}.png"
            save_horizon_plot(eval_path, subset, plot_filename)





