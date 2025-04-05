import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from colour import Color
from matplotlib.lines import Line2D

FIGSIZE = (6, 4)
LINEWIDTH = 2.0
FONTSIZE = 12


def plot_loss_accs(
    statistics,
    multiple_runs=False,
    log_x=False,
    log_y=False,
    figsize=FIGSIZE,
    linewidth=LINEWIDTH,
    fontsize=FONTSIZE,
    fileName=None,
    filePath=None,
    show=True,
):

    rows, cols = 1, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # set up colors
    color_1 = "tab:blue"  # #1f77b4
    color_2 = "tab:red"  # #d62728

    same_steps = False
    if multiple_runs:
        all_steps = statistics["all_steps"]
        same_steps = all(
            len(steps) == len(all_steps[0]) for steps in all_steps
        )  # Check if all runs have the same number of steps
        if same_steps:
            all_steps = np.array(all_steps[0]) + 1e-0  # Add 1e-0 to avoid log(0)
        else:
            all_steps = [
                np.array(steps) + 1e-0 for steps in all_steps
            ]  # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else:
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(["accuracy", "loss"]):
        ax = fig.add_subplot(rows, cols, i + 1)
        if multiple_runs:
            zs = np.array(statistics["train"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                # ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                ax.plot(
                    all_steps, zs_mean, "-", color=color_1, label=f"Train", lw=linewidth
                )
                ax.fill_between(
                    all_steps,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=color_1,
                    alpha=0.2,
                )
            else:
                for j, z in enumerate(zs):
                    ax.plot(
                        all_steps[j],
                        z,
                        "-",
                        color=colors[j],
                        label=f"Train",
                        lw=linewidth / 2,
                    )

            zs = np.array(statistics["test"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(
                    all_steps, zs_mean, "-", color=color_2, label=f"Eval", lw=linewidth
                )
                ax.fill_between(
                    all_steps,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=color_2,
                    alpha=0.2,
                )
            else:
                for j, z in enumerate(zs):
                    ax.plot(
                        all_steps[j],
                        z,
                        "--",
                        color=colors[j],
                        label=f"Eval",
                        lw=linewidth / 2,
                    )

        else:
            ax.plot(
                all_steps,
                statistics["train"][key],
                "-",
                color=color_1,
                label=f"Train",
                lw=linewidth,
            )
            ax.plot(
                all_steps,
                statistics["test"][key],
                "-",
                color=color_2,
                label=f"Eval",
                lw=linewidth,
            )

        if log_x:
            ax.set_xscale("log")
        # if log_y : ax.set_yscale('log')
        if log_y and key == "loss":
            ax.set_yscale("log")  # No need to log accuracy
        ax.tick_params(axis="y", labelsize="x-large")
        ax.tick_params(axis="x", labelsize="x-large")
        ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
        if key == "accuracy":
            s = "Accuracy"
        if key == "loss":
            s = "Loss"
        # ax.set_ylabel(s, fontsize=fontsize)
        ax.set_title(s, fontsize=fontsize)
        ax.grid(True)
        if multiple_runs and (not same_steps):
            legend_elements = [
                Line2D([0], [0], color="k", lw=linewidth, linestyle="-", label="Train"),
                Line2D([0], [0], color="k", lw=linewidth, linestyle="--", label="Eval"),
            ]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else:
            ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None:
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(
            f"{filePath}/{fileName}" + ".pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_accs_q4(
    statistics,
    multiple_runs=False,
    log_x=False,
    log_y=False,
    figsize=FIGSIZE,
    linewidth=LINEWIDTH,
    fontsize=FONTSIZE,
    fileName=None,
    filePath=None,
    show=True,
):

    rows, cols = 3, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # set up colors
    color_1 = "tab:blue"  # #1f77b4
    color_2 = "tab:red"  # #d62728

    same_steps = False
    if multiple_runs:
        all_steps = statistics["all_steps"]
        same_steps = all(
            len(steps) == len(all_steps[0]) for steps in all_steps
        )  # Check if all runs have the same number of steps
        if same_steps:
            all_steps = np.array(all_steps[0]) + 1e-0  # Add 1e-0 to avoid log(0)
        else:
            all_steps = [
                np.array(steps) + 1e-0 for steps in all_steps
            ]  # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else:
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(
        [
            "accuracy",
            "loss",
            "binary-accuracy",
            "binary-loss",
            "ternary-acc",
            "ternary-loss",
        ]
    ):
        ax = fig.add_subplot(rows, cols, i + 1)
        if multiple_runs:
            zs = np.array(statistics["train"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                # ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                ax.plot(
                    all_steps, zs_mean, "-", color=color_1, label=f"Train", lw=linewidth
                )
                ax.fill_between(
                    all_steps,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=color_1,
                    alpha=0.2,
                )
            else:
                for j, z in enumerate(zs):
                    ax.plot(
                        all_steps[j],
                        z,
                        "-",
                        color=colors[j],
                        label=f"Train",
                        lw=linewidth / 2,
                    )

            zs = np.array(statistics["test"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(
                    all_steps, zs_mean, "-", color=color_2, label=f"Eval", lw=linewidth
                )
                ax.fill_between(
                    all_steps,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=color_2,
                    alpha=0.2,
                )
            else:
                for j, z in enumerate(zs):
                    ax.plot(
                        all_steps[j],
                        z,
                        "--",
                        color=colors[j],
                        label=f"Eval",
                        lw=linewidth / 2,
                    )

        else:
            ax.plot(
                all_steps,
                statistics["train"][key],
                "-",
                color=color_1,
                label=f"Train",
                lw=linewidth,
            )
            ax.plot(
                all_steps,
                statistics["test"][key],
                "-",
                color=color_2,
                label=f"Eval",
                lw=linewidth,
            )

        if log_x:
            ax.set_xscale("log")
        # if log_y : ax.set_yscale('log')
        if log_y and "loss" in key:
            ax.set_yscale("log")  # No need to log accuracy

        ax.tick_params(axis="y", labelsize="x-large")
        ax.tick_params(axis="x", labelsize="x-large")

        if i >= 4:
            ax.set_xlabel("Training Steps (t)", fontsize=fontsize)

        if key == "accuracy":
            s = "Accuracy"
        if key == "loss":
            s = "Loss"
        if key == "binary-accuracy":
            s = "Binary accuracy"
        if key == "binary-loss":
            s = "Binary Loss"
        if key == "ternary-acc":
            s = "Ternary Accuracy"
        if key == "ternary-loss":
            s = "Ternary Loss"

        ax.set_title(s, fontsize=fontsize)

        ax.grid(True)
        if multiple_runs and (not same_steps):
            legend_elements = [
                Line2D([0], [0], color="k", lw=linewidth, linestyle="-", label="Train"),
                Line2D([0], [0], color="k", lw=linewidth, linestyle="--", label="Eval"),
            ]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else:
            ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None:
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(
            f"{filePath}/{fileName}" + ".pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_accs_q7(
    statistics,
    multiple_runs=False,
    log_x=False,
    log_y=False,
    figsize=FIGSIZE,
    linewidth=LINEWIDTH,
    fontsize=FONTSIZE,
    fileName=None,
    filePath=None,
    show=True,
):

    rows, cols = 3, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # set up colors
    color_1 = "tab:blue"  # #1f77b4
    color_2 = "tab:red"  # #d62728

    same_steps = False
    if multiple_runs:
        all_steps = statistics["all_steps"]
        same_steps = all(
            len(steps) == len(all_steps[0]) for steps in all_steps
        )  # Check if all runs have the same number of steps
        if same_steps:
            all_steps = np.array(all_steps[0]) + 1e-0  # Add 1e-0 to avoid log(0)
        else:
            all_steps = [
                np.array(steps) + 1e-0 for steps in all_steps
            ]  # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else:
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(["accuracy", "loss", "l2-norm"]):
        ax = fig.add_subplot(rows, cols, i + 1)
        if multiple_runs:
            zs = np.array(statistics["train"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                # ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                ax.plot(
                    all_steps, zs_mean, "-", color=color_1, label=f"Train", lw=linewidth
                )
                ax.fill_between(
                    all_steps,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=color_1,
                    alpha=0.2,
                )
            else:
                for j, z in enumerate(zs):
                    ax.plot(
                        all_steps[j],
                        z,
                        "-",
                        color=colors[j],
                        label=f"Train",
                        lw=linewidth / 2,
                    )

            zs = np.array(statistics["test"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(
                    all_steps, zs_mean, "-", color=color_2, label=f"Eval", lw=linewidth
                )
                ax.fill_between(
                    all_steps,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=color_2,
                    alpha=0.2,
                )
            else:
                for j, z in enumerate(zs):
                    ax.plot(
                        all_steps[j],
                        z,
                        "--",
                        color=colors[j],
                        label=f"Eval",
                        lw=linewidth / 2,
                    )

        else:
            ax.plot(
                all_steps,
                statistics["train"][key],
                "-",
                color=color_1,
                label=f"Train",
                lw=linewidth,
            )
            ax.plot(
                all_steps,
                statistics["test"][key],
                "-",
                color=color_2,
                label=f"Eval",
                lw=linewidth,
            )

        if log_x:
            ax.set_xscale("log")
        # if log_y : ax.set_yscale('log')
        if log_y and "loss" in key:
            ax.set_yscale("log")  # No need to log accuracy

        ax.tick_params(axis="y", labelsize="x-large")
        ax.tick_params(axis="x", labelsize="x-large")

        if i >= 4:
            ax.set_xlabel("Training Steps (t)", fontsize=fontsize)

        if key == "accuracy":
            s = "Accuracy"
        if key == "loss":
            s = "Loss"
        if key == "l2-norm":
            s = "Parameter L2 Norm"

        ax.set_title(s, fontsize=fontsize)

        ax.grid(True)
        if multiple_runs and (not same_steps):
            legend_elements = [
                Line2D([0], [0], color="k", lw=linewidth, linestyle="-", label="Train"),
                Line2D([0], [0], color="k", lw=linewidth, linestyle="--", label="Eval"),
            ]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else:
            ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None:
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(
            f"{filePath}/{fileName}" + ".pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_accs_multiple_configs(
    list_of_statistics,
    list_of_statistics_labels,
    suffix,
    multiple_runs=False,
    log_x=False,
    log_y=False,
    figsize=FIGSIZE,
    linewidth=LINEWIDTH,
    fontsize=FONTSIZE,
    fileName=None,
    filePath=None,
    show=True,
):
    """Plots loss/accuracy but with multiple configs."""

    rows, cols = 2, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    num_colors = len(list_of_statistics_labels)
    lb = Color("lightblue")
    blue = Color("navy")
    color_1 = [i.hex for i in list(lb.range_to(blue, num_colors))]

    orange = Color("orange")
    red = Color("red")
    color_2 = [i.hex for i in list(orange.range_to(red, num_colors))]

    same_steps = False
    if multiple_runs:
        all_steps = list_of_statistics[0]["all_steps"]
        same_steps = all(
            len(steps) == len(all_steps[0]) for steps in all_steps
        )  # Check if all runs have the same number of steps
        if same_steps:
            all_steps = np.array(all_steps[0]) + 1e-0  # Add 1e-0 to avoid log(0)
        else:
            all_steps = [
                np.array(steps) + 1e-0 for steps in all_steps
            ]  # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else:
        all_steps = np.array(list_of_statistics[0]["all_steps"]) + 1e-0

    for i, key in enumerate(["accuracy", "loss"]):
        for j, mode_key in enumerate(["train", "test"]):
            ax = fig.add_subplot(rows, cols, cols * i + j + 1)
            color = color_1 if mode_key == "train" else color_2

            for k, (label, statistics) in enumerate(
                zip(list_of_statistics_labels, list_of_statistics)
            ):
                if multiple_runs:
                    zs = np.array(statistics[mode_key][key])
                    if same_steps:
                        zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                        # ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                        ax.plot(
                            all_steps,
                            zs_mean,
                            "-",
                            color=color[k],
                            label=f"{mode_key.capitalize()}-{suffix}={label}",
                            lw=linewidth,
                        )
                        ax.fill_between(
                            all_steps,
                            zs_mean - zs_std,
                            zs_mean + zs_std,
                            color=color[k],
                            alpha=0.2,
                        )
                    else:
                        for m, z in enumerate(zs):
                            ax.plot(
                                all_steps[m],
                                z,
                                "-",
                                color=colors[m],
                                label=mode_key.capitalize(),
                                lw=linewidth / 2,
                            )
                else:
                    ax.plot(
                        all_steps,
                        statistics[mode_key][key],
                        "-",
                        color=color[k],
                        label=f"{mode_key.capitalize()}-{suffix}={label}",
                        lw=linewidth,
                    )

                if log_x:
                    ax.set_xscale("log")
                # if log_y : ax.set_yscale('log')
                if log_y and key == "loss":
                    ax.set_yscale("log")  # No need to log accuracy

                ax.tick_params(axis="y", labelsize="x-large")
                ax.tick_params(axis="x", labelsize="x-large")
                if i == 1:
                    ax.set_xlabel("Training Steps (t)", fontsize=fontsize)

                if key == "accuracy":
                    s = "Accuracy"
                if key == "loss":
                    s = "Loss"

                # ax.set_ylabel(s, fontsize=fontsize)
                ax.set_title(s, fontsize=fontsize)

                ax.grid(True)
                if multiple_runs and (not same_steps):
                    legend_elements = [
                        Line2D(
                            [0],
                            [0],
                            color="k",
                            lw=linewidth,
                            linestyle="-",
                            label="Train",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="k",
                            lw=linewidth,
                            linestyle="--",
                            label="Eval",
                        ),
                    ]
                    ax.legend(handles=legend_elements, fontsize=fontsize)
                else:
                    ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None:
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(
            f"{filePath}/{fileName}" + ".pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

    if show:
        plt.show()
    else:
        plt.close()
