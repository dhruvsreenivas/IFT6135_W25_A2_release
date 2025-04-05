import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from colour import Color

from checkpointing import (
    get_all_checkpoints_per_trials,
    get_extrema_performance_steps_per_trials,
)
from plotter import (
    plot_loss_accs_multiple_configs,
    plot_loss_accs_q4,
    plot_loss_accs_q7,
)
from run_exp import bool_flag


def get_extrema_dict_from_args(args, M=2, seeds=[0, 42]):
    # first get all checkpoints
    all_checkpoint_paths = []
    for seed, m in zip(seeds, range(M)):
        print(f"Model {m+1}/{M}")
        args.exp_id = m  # Set the experiment id
        args.seed = seed  # Set the seed

        checkpoint_path = os.path.join(args.log_dir, str(args.exp_id))
        i = 0
        while os.path.exists(checkpoint_path):
            all_checkpoint_paths.append(checkpoint_path)

            i += 1
            checkpoint_path = os.path.join(args.log_dir, str(i))

    _, all_metrics = get_all_checkpoints_per_trials(
        all_checkpoint_paths, args.exp_name, just_files=True, verbose=args.verbose
    )

    # now get the extrema from this set of metrics.
    extrema_dict = get_extrema_performance_steps_per_trials(all_metrics)
    return extrema_dict


def q3p1_plot_losses_and_accuracies(args, M=2, seeds=[0, 42]):
    # first get all the data associated with each `r_train` in a dict.

    r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    r_train_statistics = []
    for r_train in r_trains:
        all_checkpoint_paths = []
        log_dir = os.path.join(args.log_dir, f"r-train-{r_train}")
        checkpoint_path = os.path.join(log_dir, str(args.exp_id))
        i = 0
        while os.path.exists(checkpoint_path):
            all_checkpoint_paths.append(checkpoint_path)

            i += 1
            checkpoint_path = os.path.join(args.log_dir, str(i))

        exp_name = f"scale_data_{r_train}"
        _, all_metrics = get_all_checkpoints_per_trials(
            all_checkpoint_paths, exp_name, just_files=True, verbose=args.verbose
        )
        r_train_statistics.append(all_metrics)

    plot_loss_accs_multiple_configs(
        r_train_statistics,
        r_trains,
        suffix="r-train",
        multiple_runs=True,
        log_x=False,
        log_y=False,
        fileName="q3p1",
        filePath=args.log_dir,
        show=True,
    )


def q3p2_plotting(args):
    r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    min_train_losses_mean = []
    min_train_losses_std = []
    max_train_accs_mean = []
    max_train_accs_std = []
    min_test_losses_mean = []
    min_test_losses_std = []
    max_test_accs_mean = []
    max_test_accs_std = []

    min_train_losses_step_mean = []
    min_train_losses_step_std = []
    max_train_accs_step_mean = []
    max_train_accs_step_std = []
    min_test_losses_step_mean = []
    min_test_losses_step_std = []
    max_test_accs_step_mean = []
    max_test_accs_step_std = []

    # set up colors
    color_1 = "tab:blue"  # #1f77b4
    color_2 = "tab:red"  # #d62728
    fontsize = 12

    for r_train in r_trains:
        all_checkpoint_paths = []
        log_dir = os.path.join(args.log_dir, f"r-train-{r_train}")
        checkpoint_path = os.path.join(log_dir, str(args.exp_id))
        i = 0
        while os.path.exists(checkpoint_path):
            all_checkpoint_paths.append(checkpoint_path)

            i += 1
            checkpoint_path = os.path.join(args.log_dir, str(i))

        exp_name = f"scale_data_{r_train}"
        _, all_metrics = get_all_checkpoints_per_trials(
            all_checkpoint_paths, exp_name, just_files=True, verbose=args.verbose
        )
        extrema_dict = get_extrema_performance_steps_per_trials(all_metrics)

        min_train_losses_mean.append(extrema_dict["min_train_loss"])
        min_train_losses_std.append(extrema_dict["min_train_loss_std"])
        min_train_losses_step_mean.append(extrema_dict["min_train_loss_step"])
        min_train_losses_step_std.append(extrema_dict["min_train_loss_step_std"])
        min_test_losses_mean.append(extrema_dict["min_test_loss"])
        min_test_losses_std.append(extrema_dict["min_test_loss_std"])
        min_test_losses_step_mean.append(extrema_dict["min_test_loss_step"])
        min_test_losses_step_std.append(extrema_dict["min_test_loss_step_std"])

        max_train_accs_mean.append(extrema_dict["max_train_accuracy"])
        max_train_accs_std.append(extrema_dict["max_train_accuracy_std"])
        max_train_accs_step_mean.append(extrema_dict["max_train_accuracy_step"])
        max_train_accs_step_std.append(extrema_dict["max_train_accuracy_step_std"])
        max_test_accs_mean.append(extrema_dict["max_test_accuracy"])
        max_test_accs_std.append(extrema_dict["max_test_accuracy_std"])
        max_test_accs_step_mean.append(extrema_dict["max_test_accuracy_step"])
        max_test_accs_step_std.append(extrema_dict["max_test_accuracy_step_std"])

    # now we just plot each one.
    rows, cols = 4, 1
    figsize = (6, 4)
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # j is between train and val, i: loss/acc/tf_l/tf_a
    for i in range(rows):
        ax = fig.add_subplot(rows, cols, i + 1)

        if i == 0:
            # min loss
            ax.plot(
                r_trains,
                np.array(min_train_losses_mean),
                label="train",
                color=color_1,
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(min_train_losses_mean) - np.array(min_train_losses_std),
                np.array(min_train_losses_mean) + np.array(min_train_losses_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                r_trains,
                np.array(min_test_losses_mean),
                label="eval",
                color=color_2,
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(min_test_losses_mean) - np.array(min_test_losses_std),
                np.array(min_test_losses_mean) + np.array(min_test_losses_std),
                color=color_2,
                alpha=0.2,
            )
            ax.set_yscale("log")
            ax.legend(fontsize=fontsize)
        elif i == 1:
            # max acc
            ax.plot(
                r_trains,
                np.array(max_train_accs_mean),
                label="train",
                color=color_1,
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(max_train_accs_mean) - np.array(max_train_accs_std),
                np.array(max_train_accs_mean) + np.array(max_train_accs_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                r_trains,
                np.array(max_test_accs_mean),
                label="eval",
                color=color_2,
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(max_test_accs_mean) - np.array(max_test_accs_std),
                np.array(max_test_accs_mean) + np.array(max_test_accs_std),
                color=color_2,
                alpha=0.2,
            )
        elif i == 2:
            # min loss step
            ax.plot(
                r_trains,
                np.array(min_train_losses_step_mean),
                color=color_1,
                label="train",
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(min_train_losses_step_mean)
                - np.array(min_train_losses_step_std),
                np.array(min_train_losses_step_mean)
                + np.array(min_train_losses_step_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                r_trains,
                np.array(min_test_losses_step_mean),
                color=color_2,
                label="eval",
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(min_test_losses_step_mean)
                - np.array(min_test_losses_step_std),
                np.array(min_test_losses_step_mean)
                + np.array(min_test_losses_step_std),
                color=color_2,
                alpha=0.2,
            )
        else:
            # max acc step
            ax.plot(
                r_trains,
                np.array(max_train_accs_step_mean),
                color=color_1,
                label="train",
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(max_train_accs_step_mean) - np.array(max_train_accs_step_std),
                np.array(max_train_accs_step_mean) + np.array(max_train_accs_step_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                r_trains,
                np.array(max_test_accs_step_mean),
                color=color_2,
                label="eval",
                lw=2.0,
            )
            ax.fill_between(
                r_trains,
                np.array(max_test_accs_step_mean) - np.array(max_test_accs_step_std),
                np.array(max_test_accs_mean) + np.array(max_test_accs_step_std),
                color=color_2,
                alpha=0.2,
            )

        # if i == 3 we set the xlabel
        if i == 3:
            ax.set_xlabel("Data fraction (r_train)", fontsize=fontsize)

        # if i == 0 we set the ylabel
        if i == 0:
            ylabel = "Min loss"
        elif i == 1:
            ylabel = "Max accuracy"
        elif i == 2:
            ylabel = "Min loss step"
        else:
            ylabel = "Max accuracy step"

        ax.set_ylabel(ylabel, fontsize=fontsize)

    plt.savefig(
        os.path.join(args.log_dir, "q3p2.pdf"),
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )


def q4p2_plotting(args, M=2, seeds=[0, 42]):
    all_checkpoint_paths = []
    for seed, m in zip(seeds, range(M)):
        print(f"Model {m+1}/{M}")
        args.exp_id = m  # Set the experiment id
        args.seed = seed  # Set the seed

        checkpoint_path = os.path.join(args.log_dir, str(args.exp_id))
        i = 0
        while os.path.exists(checkpoint_path):
            all_checkpoint_paths.append(checkpoint_path)

            i += 1
            checkpoint_path = os.path.join(args.log_dir, str(i))

    _, all_metrics = get_all_checkpoints_per_trials(
        all_checkpoint_paths, args.exp_name, just_files=True, verbose=args.verbose
    )

    plot_loss_accs_q4(
        all_metrics,
        multiple_runs=True,
        log_x=False,
        log_y=False,
        fileName="q4p2",
        filePath=args.log_dir,
        show=False,
    )


def q5p1_plotting(args, M=2, seeds=[0, 42]):

    Ls = [1, 2, 3]
    for L in Ls:
        l_statistics = []
        for d in [64, 128, 256]:
            all_checkpoint_paths = []
            log_dir = os.path.join(args.log_dir, f"layers-{L}-dim-{d}")
            checkpoint_path = os.path.join(log_dir, str(args.exp_id))
            i = 0
            while os.path.exists(checkpoint_path):
                all_checkpoint_paths.append(checkpoint_path)

                i += 1
                checkpoint_path = os.path.join(args.log_dir, str(i))

            exp_name = f"scale_model_layers_{L}_dim_{d}"
            _, all_metrics = get_all_checkpoints_per_trials(
                all_checkpoint_paths, exp_name, just_files=True, verbose=args.verbose
            )
            l_statistics.append(all_metrics)

        plot_loss_accs_multiple_configs(
            l_statistics,
            [64, 128, 256],
            suffix="dims",
            multiple_runs=True,
            log_x=False,
            log_y=False,
            fileName=f"q5p1_L={L}",
            filePath=args.log_dir,
            show=True,
        )


def q5p2_plotting(args):
    Ls = [1, 2, 3]
    Ds = [64, 128, 256]

    L_dict = dict()

    for L in Ls:
        min_train_losses_mean = []
        min_train_losses_std = []
        max_train_accs_mean = []
        max_train_accs_std = []
        min_test_losses_mean = []
        min_test_losses_std = []
        max_test_accs_mean = []
        max_test_accs_std = []

        min_train_losses_step_mean = []
        min_train_losses_step_std = []
        max_train_accs_step_mean = []
        max_train_accs_step_std = []
        min_test_losses_step_mean = []
        min_test_losses_step_std = []
        max_test_accs_step_mean = []
        max_test_accs_step_std = []

        for D in Ds:
            all_checkpoint_paths = []
            log_dir = os.path.join(args.log_dir, f"layers-{L}-dim-{D}")
            checkpoint_path = os.path.join(log_dir, str(args.exp_id))
            i = 0
            while os.path.exists(checkpoint_path):
                all_checkpoint_paths.append(checkpoint_path)

                i += 1
                checkpoint_path = os.path.join(args.log_dir, str(i))

            exp_name = f"scale_model_layers_{L}_dim_{D}"
            _, all_metrics = get_all_checkpoints_per_trials(
                all_checkpoint_paths,
                exp_name,
                just_files=True,
                verbose=args.verbose,
            )
            extrema_dict = get_extrema_performance_steps_per_trials(all_metrics)

            # all scalars, corresponding to stats over (d, L)
            min_train_losses_mean.append(extrema_dict["min_train_loss"])
            min_train_losses_std.append(extrema_dict["min_train_loss_std"])
            min_train_losses_step_mean.append(extrema_dict["min_train_loss_step"])
            min_train_losses_step_std.append(extrema_dict["min_train_loss_step_std"])
            min_test_losses_mean.append(extrema_dict["min_test_loss"])
            min_test_losses_std.append(extrema_dict["min_test_loss_std"])
            min_test_losses_step_mean.append(extrema_dict["min_test_loss_step"])
            min_test_losses_step_std.append(extrema_dict["min_test_loss_step_std"])

            max_train_accs_mean.append(extrema_dict["max_train_accuracy"])
            max_train_accs_std.append(extrema_dict["max_train_accuracy_std"])
            max_train_accs_step_mean.append(extrema_dict["max_train_accuracy_step"])
            max_train_accs_step_std.append(extrema_dict["max_train_accuracy_step_std"])
            max_test_accs_mean.append(extrema_dict["max_test_accuracy"])
            max_test_accs_std.append(extrema_dict["max_test_accuracy_std"])
            max_test_accs_step_mean.append(extrema_dict["max_test_accuracy_step"])
            max_test_accs_step_std.append(extrema_dict["max_test_accuracy_step_std"])

        # now we have the stats for a given L, for all d's.
        # add to global dict.

        L_dict[L] = [
            min_train_losses_mean,
            min_train_losses_std,
            min_train_losses_step_mean,
            min_train_losses_step_std,
            min_test_losses_mean,
            min_test_losses_std,
            min_test_losses_step_mean,
            min_test_losses_step_std,
            max_train_accs_mean,
            max_train_accs_std,
            max_train_accs_step_mean,
            max_train_accs_step_std,
            max_test_accs_mean,
            max_test_accs_std,
            max_test_accs_step_mean,
            max_test_accs_step_std,
        ]

    # now, we create figure as before
    num_colors = 3
    lb = Color("lightblue")
    blue = Color("navy")
    color_1 = [i.hex for i in list(lb.range_to(blue, num_colors))]

    orange = Color("orange")
    red = Color("red")
    color_2 = [i.hex for i in list(orange.range_to(red, num_colors))]

    rows, cols = 4, 1
    figsize = (6, 4)
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # now, for each L, we plot over D.
    for i in range(rows):
        ax = fig.add_subplot(rows, cols, i + 1)

        if i == 0:
            # min loss
            for L in [1, 2, 3]:
                # get the min train/test losses for L
                (
                    min_train_losses_mean,
                    min_train_losses_std,
                    min_test_losses_mean,
                    min_test_losses_std,
                ) = (
                    L_dict[L][0],
                    L_dict[L][1],
                    L_dict[L][4],
                    L_dict[L][5],
                )
                ax.plot(
                    Ds,
                    np.array(min_train_losses_mean),
                    label=f"train-L={L}",
                    color=color_1[L - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(min_train_losses_mean) - np.array(min_train_losses_std),
                    np.array(min_train_losses_mean) + np.array(min_train_losses_std),
                    color=color_1[L - 1],
                    alpha=0.2,
                )

                ax.plot(
                    Ds,
                    np.array(min_test_losses_mean),
                    label="eval",
                    color=color_2[L - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(min_test_losses_mean) - np.array(min_test_losses_std),
                    np.array(min_test_losses_mean) + np.array(min_test_losses_std),
                    color=color_2[L - 1],
                    alpha=0.2,
                )

            ax.set_yscale("log")
            ax.legend(fontsize=12)
        elif i == 1:
            # max acc

            for L in [1, 2, 3]:
                # get the min train/test losses for L
                (
                    max_train_accs_mean,
                    max_train_accs_std,
                    max_test_accs_mean,
                    max_test_accs_std,
                ) = (
                    L_dict[L][8],
                    L_dict[L][9],
                    L_dict[L][12],
                    L_dict[L][13],
                )

                ax.plot(
                    Ds,
                    np.array(max_train_accs_mean),
                    label=f"train-L={L}",
                    color=color_1[L - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(max_train_accs_mean) - np.array(max_train_accs_std),
                    np.array(max_train_accs_mean) + np.array(max_train_accs_std),
                    color=color_1[L - 1],
                    alpha=0.2,
                )

                ax.plot(
                    Ds,
                    np.array(max_test_accs_mean),
                    label=f"eval-L={L}",
                    color=color_2[L - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(max_test_accs_mean) - np.array(max_test_accs_std),
                    np.array(max_test_accs_mean) + np.array(max_test_accs_std),
                    color=color_2[L - 1],
                    alpha=0.2,
                )
            ax.legend(fontsize=12)
        elif i == 2:
            # min loss step

            for L in [1, 2, 3]:
                # get the min train/test losses for L
                (
                    min_train_losses_step_mean,
                    min_train_losses_step_std,
                    min_test_losses_step_mean,
                    min_test_losses_step_std,
                ) = (
                    L_dict[L][2],
                    L_dict[L][3],
                    L_dict[L][6],
                    L_dict[L][7],
                )
                ax.plot(
                    Ds,
                    np.array(min_train_losses_step_mean),
                    color=color_1[L - 1],
                    label=f"train-L={L}",
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(min_train_losses_step_mean)
                    - np.array(min_train_losses_step_std),
                    np.array(min_train_losses_step_mean)
                    + np.array(min_train_losses_step_std),
                    color=color_1[L - 1],
                    alpha=0.2,
                )

                ax.plot(
                    Ds,
                    np.array(min_test_losses_step_mean),
                    color=color_2[L - 1],
                    label=f"eval-L={L}",
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(min_test_losses_step_mean)
                    - np.array(min_test_losses_step_std),
                    np.array(min_test_losses_step_mean)
                    + np.array(min_test_losses_step_std),
                    color=color_2[L - 1],
                    alpha=0.2,
                )
            ax.legend(fontsize=12)
        else:
            # max acc step

            for L in [1, 2, 3]:
                # get the min train/test losses for L
                (
                    max_train_accs_step_mean,
                    max_train_accs_step_std,
                    max_test_accs_step_mean,
                    max_test_accs_step_std,
                ) = (
                    L_dict[L][10],
                    L_dict[L][11],
                    L_dict[L][14],
                    L_dict[L][15],
                )
                ax.plot(
                    Ds,
                    np.array(max_train_accs_step_mean),
                    color=color_1[L - 1],
                    label=f"train-L={L}",
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(max_train_accs_step_mean)
                    - np.array(max_train_accs_step_std),
                    np.array(max_train_accs_step_mean)
                    + np.array(max_train_accs_step_std),
                    color=color_1[L - 1],
                    alpha=0.2,
                )

                ax.plot(
                    Ds,
                    np.array(max_test_accs_step_mean),
                    color=color_2[L - 1],
                    label="eval",
                    lw=2.0,
                )
                ax.fill_between(
                    Ds,
                    np.array(max_test_accs_step_mean)
                    - np.array(max_test_accs_step_std),
                    np.array(max_test_accs_mean) + np.array(max_test_accs_step_std),
                    color=color_2[L - 1],
                    alpha=0.2,
                )

        # if i == 3 we set the xlabel
        if i == 3:
            ax.set_xlabel("Dimension of model (D)", fontsize=12)

        # if i == 0 we set the ylabel
        if i == 0:
            ylabel = "Min loss"
        elif i == 1:
            ylabel = "Max accuracy"
        elif i == 2:
            ylabel = "Min loss step"
        else:
            ylabel = "Max accuracy step"

        ax.set_ylabel(ylabel, fontsize=12)

    plt.savefig(
        os.path.join(args.log_dir, "q5p2.pdf"),
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )


def q6p1_plotting(args):
    Ts = [20001]
    Bs = [32, 64, 128, 256, 512]
    for T in Ts:
        t_statistics = []
        for B in Bs:
            all_checkpoint_paths = []
            log_dir = os.path.join(args.log_dir, f"batch-size-{B}")
            checkpoint_path = os.path.join(log_dir, str(args.exp_id))
            i = 0
            while os.path.exists(checkpoint_path):
                all_checkpoint_paths.append(checkpoint_path)

                i += 1
                checkpoint_path = os.path.join(args.log_dir, str(i))

            exp_name = f"scale_compute_{B}"
            _, all_metrics = get_all_checkpoints_per_trials(
                all_checkpoint_paths, exp_name, just_files=True, verbose=args.verbose
            )
            t_statistics.append(all_metrics)

        plot_loss_accs_multiple_configs(
            t_statistics,
            [32, 64, 128, 256, 512],
            suffix="batch-size",
            multiple_runs=True,
            log_x=False,
            log_y=False,
            fileName=f"q6p1",
            filePath=args.log_dir,
            show=True,
        )


def q6p2_plotting(args):
    # first get the data with B on the x-axis, and the times for each of the T_max on the y-axis
    Bs = [32, 64, 128, 256, 512]
    T_maxs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    T_max_dict = dict()

    for T_max in T_maxs:
        # get the data for all Bs here.
        min_train_losses_mean = []
        min_train_losses_std = []
        max_train_accs_mean = []
        max_train_accs_std = []
        min_test_losses_mean = []
        min_test_losses_std = []
        max_test_accs_mean = []
        max_test_accs_std = []

        min_train_losses_step_mean = []
        min_train_losses_step_std = []
        max_train_accs_step_mean = []
        max_train_accs_step_std = []
        min_test_losses_step_mean = []
        min_test_losses_step_std = []
        max_test_accs_step_mean = []
        max_test_accs_step_std = []

        for B in Bs:
            all_checkpoint_paths = []
            log_dir = os.path.join(args.log_dir, f"batch-size-{B}")
            checkpoint_path = os.path.join(log_dir, str(args.exp_id))
            i = 0
            while os.path.exists(checkpoint_path):
                all_checkpoint_paths.append(checkpoint_path)

                i += 1
                checkpoint_path = os.path.join(args.log_dir, str(i))

            exp_name = f"scale_compute_{B}"
            _, all_metrics = get_all_checkpoints_per_trials(
                all_checkpoint_paths,
                exp_name,
                just_files=True,
                verbose=args.verbose,
            )
            extrema_dict = get_extrema_performance_steps_per_trials(all_metrics, T_max)

            # all scalars, this corresponds over batch size.
            min_train_losses_mean.append(extrema_dict["min_train_loss"])
            min_train_losses_std.append(extrema_dict["min_train_loss_std"])
            min_train_losses_step_mean.append(extrema_dict["min_train_loss_step"])
            min_train_losses_step_std.append(extrema_dict["min_train_loss_step_std"])
            min_test_losses_mean.append(extrema_dict["min_test_loss"])
            min_test_losses_std.append(extrema_dict["min_test_loss_std"])
            min_test_losses_step_mean.append(extrema_dict["min_test_loss_step"])
            min_test_losses_step_std.append(extrema_dict["min_test_loss_step_std"])

            max_train_accs_mean.append(extrema_dict["max_train_accuracy"])
            max_train_accs_std.append(extrema_dict["max_train_accuracy_std"])
            max_train_accs_step_mean.append(extrema_dict["max_train_accuracy_step"])
            max_train_accs_step_std.append(extrema_dict["max_train_accuracy_step_std"])
            max_test_accs_mean.append(extrema_dict["max_test_accuracy"])
            max_test_accs_std.append(extrema_dict["max_test_accuracy_std"])
            max_test_accs_step_mean.append(extrema_dict["max_test_accuracy_step"])
            max_test_accs_step_std.append(extrema_dict["max_test_accuracy_step_std"])

        # now we have the stats for a given T_max, for all Bs
        # add to global dict.

        T_max_dict[T_max] = [
            min_train_losses_mean,
            min_train_losses_std,
            min_train_losses_step_mean,
            min_train_losses_step_std,
            min_test_losses_mean,
            min_test_losses_std,
            min_test_losses_step_mean,
            min_test_losses_step_std,
            max_train_accs_mean,
            max_train_accs_std,
            max_train_accs_step_mean,
            max_train_accs_step_std,
            max_test_accs_mean,
            max_test_accs_std,
            max_test_accs_step_mean,
            max_test_accs_step_std,
        ]

    # now, we create figure as before
    num_colors = len(T_maxs)
    lb = Color("lightblue")
    blue = Color("navy")
    color_1 = [i.hex for i in list(lb.range_to(blue, num_colors))]

    orange = Color("orange")
    red = Color("red")
    color_2 = [i.hex for i in list(orange.range_to(red, num_colors))]

    rows, cols = 4, 1
    figsize = (6, 4)
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # now, for each T_max, we plot over B
    for i in range(rows):
        ax = fig.add_subplot(rows, cols, i + 1)

        if i == 0:
            # min loss
            for T_max in T_maxs:
                # get the min train/test losses for L
                (
                    min_train_losses_mean,
                    min_train_losses_std,
                    min_test_losses_mean,
                    min_test_losses_std,
                ) = (
                    T_max_dict[T_max][0],
                    T_max_dict[T_max][1],
                    T_max_dict[T_max][4],
                    T_max_dict[T_max][5],
                )
                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(min_train_losses_mean),
                    label=f"train-tmax={T_max}",
                    color=color_1[int(10 * T_max) - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(min_train_losses_mean) - np.array(min_train_losses_std),
                    np.array(min_train_losses_mean) + np.array(min_train_losses_std),
                    color=color_1[int(10 * T_max) - 1],
                    alpha=0.2,
                )

                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(min_test_losses_mean),
                    label=f"eval-tmax={T_max}",
                    color=color_2[int(10 * T_max) - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(min_test_losses_mean) - np.array(min_test_losses_std),
                    np.array(min_test_losses_mean) + np.array(min_test_losses_std),
                    color=color_2[int(10 * T_max) - 1],
                    alpha=0.2,
                )

            ax.set_yscale("log")
        elif i == 1:
            # max acc

            for T_max in T_maxs:
                # get the min train/test losses for L
                (
                    max_train_accs_mean,
                    max_train_accs_std,
                    max_test_accs_mean,
                    max_test_accs_std,
                ) = (
                    T_max_dict[T_max][8],
                    T_max_dict[T_max][9],
                    T_max_dict[T_max][12],
                    T_max_dict[T_max][13],
                )

                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(max_train_accs_mean),
                    label=f"train-tmax={T_max}",
                    color=color_1[int(10 * T_max) - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(max_train_accs_mean) - np.array(max_train_accs_std),
                    np.array(max_train_accs_mean) + np.array(max_train_accs_std),
                    color=color_1[int(10 * T_max) - 1],
                    alpha=0.2,
                )

                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(max_test_accs_mean),
                    label=f"eval-tmax={T_max}",
                    color=color_2[int(10 * T_max) - 1],
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(max_test_accs_mean) - np.array(max_test_accs_std),
                    np.array(max_test_accs_mean) + np.array(max_test_accs_std),
                    color=color_2[int(10 * T_max) - 1],
                    alpha=0.2,
                )
        elif i == 2:
            # min loss step

            for T_max in T_maxs:
                # get the min train/test losses for L
                (
                    min_train_losses_step_mean,
                    min_train_losses_step_std,
                    min_test_losses_step_mean,
                    min_test_losses_step_std,
                ) = (
                    T_max_dict[T_max][2],
                    T_max_dict[T_max][3],
                    T_max_dict[T_max][6],
                    T_max_dict[T_max][7],
                )
                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(min_train_losses_step_mean),
                    color=color_1[int(10 * T_max) - 1],
                    label=f"train-tmax={T_max}",
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(min_train_losses_step_mean)
                    - np.array(min_train_losses_step_std),
                    np.array(min_train_losses_step_mean)
                    + np.array(min_train_losses_step_std),
                    color=color_1[int(10 * T_max) - 1],
                    alpha=0.2,
                )

                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(min_test_losses_step_mean),
                    color=color_2[int(10 * T_max) - 1],
                    label=f"eval-tmax={T_max}",
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(min_test_losses_step_mean)
                    - np.array(min_test_losses_step_std),
                    np.array(min_test_losses_step_mean)
                    + np.array(min_test_losses_step_std),
                    color=color_2[int(10 * T_max) - 1],
                    alpha=0.2,
                )
            ax.legend(fontsize=12)
        else:
            # max acc step
            for T_max in T_maxs:
                # get the min train/test losses for L
                (
                    max_train_accs_step_mean,
                    max_train_accs_step_std,
                    max_test_accs_step_mean,
                    max_test_accs_step_std,
                ) = (
                    T_max_dict[T_max][10],
                    T_max_dict[T_max][11],
                    T_max_dict[T_max][14],
                    T_max_dict[T_max][15],
                )
                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(max_train_accs_step_mean),
                    color=color_1[int(10 * T_max) - 1],
                    label=f"eval-tmax={T_max}",
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(max_train_accs_step_mean)
                    - np.array(max_train_accs_step_std),
                    np.array(max_train_accs_step_mean)
                    + np.array(max_train_accs_step_std),
                    color=color_1[int(10 * T_max) - 1],
                    alpha=0.2,
                )

                ax.plot(
                    np.log2(np.array(Bs)),
                    np.array(max_test_accs_step_mean),
                    color=color_2[int(10 * T_max) - 1],
                    label=f"eval-tmax={T_max}",
                    lw=2.0,
                )
                ax.fill_between(
                    np.log2(np.array(Bs)),
                    np.array(max_test_accs_step_mean)
                    - np.array(max_test_accs_step_std),
                    np.array(max_test_accs_mean) + np.array(max_test_accs_step_std),
                    color=color_2[int(10 * T_max) - 1],
                    alpha=0.2,
                )

        # if i == 3 we set the xlabel
        if i == 3:
            ax.set_xlabel("Batch size (B)", fontsize=12)

        # if i == 0 we set the ylabel
        if i == 0:
            ylabel = "Min loss"
        elif i == 1:
            ylabel = "Max accuracy"
        elif i == 2:
            ylabel = "Min loss step"
        else:
            ylabel = "Max accuracy step"

        ax.set_ylabel(ylabel, fontsize=12)

    plt.savefig(
        os.path.join(args.log_dir, "q6p2.pdf"),
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )


def q7p1_plotting(args, M=2, seeds=[0, 42]):
    wds = [0.25, 0.5, 0.75, 1.0]

    for wd in wds:
        all_checkpoint_paths = []
        for seed, m in zip(seeds, range(M)):
            print(f"Model {m+1}/{M}")
            args.exp_id = m  # Set the experiment id
            args.seed = seed  # Set the seed

            checkpoint_path = os.path.join(
                args.log_dir, f"weight-decay-{wd}", str(args.exp_id)
            )
            i = 0
            while os.path.exists(checkpoint_path):
                all_checkpoint_paths.append(checkpoint_path)

                i += 1
                checkpoint_path = os.path.join(
                    args.log_dir, f"weight-decay-{wd}", str(i)
                )

        _, all_metrics = get_all_checkpoints_per_trials(
            all_checkpoint_paths,
            args.exp_name + f"_{wd}",
            just_files=True,
            verbose=args.verbose,
        )

        plot_loss_accs_q7(
            all_metrics,
            multiple_runs=True,
            log_x=False,
            log_y=False,
            fileName=f"q7p1_wd={wd}",
            filePath=args.log_dir,
            show=False,
        )


def q7p2_plotting(args):
    wds = [0.25, 0.5, 0.75, 1.0]

    min_train_losses_mean = []
    min_train_losses_std = []
    max_train_accs_mean = []
    max_train_accs_std = []
    min_test_losses_mean = []
    min_test_losses_std = []
    max_test_accs_mean = []
    max_test_accs_std = []

    min_train_losses_step_mean = []
    min_train_losses_step_std = []
    max_train_accs_step_mean = []
    max_train_accs_step_std = []
    min_test_losses_step_mean = []
    min_test_losses_step_std = []
    max_test_accs_step_mean = []
    max_test_accs_step_std = []

    # set up colors
    color_1 = "tab:blue"  # #1f77b4
    color_2 = "tab:red"  # #d62728
    fontsize = 12

    for wd in wds:
        all_checkpoint_paths = []
        log_dir = os.path.join(args.log_dir, f"weight-decay-{wd}")
        checkpoint_path = os.path.join(log_dir, str(args.exp_id))
        i = 0
        while os.path.exists(checkpoint_path):
            all_checkpoint_paths.append(checkpoint_path)

            i += 1
            checkpoint_path = os.path.join(args.log_dir, str(i))

        exp_name = f"regularization_{wd}"
        _, all_metrics = get_all_checkpoints_per_trials(
            all_checkpoint_paths, exp_name, just_files=True, verbose=args.verbose
        )
        extrema_dict = get_extrema_performance_steps_per_trials(all_metrics)

        min_train_losses_mean.append(extrema_dict["min_train_loss"])
        min_train_losses_std.append(extrema_dict["min_train_loss_std"])
        min_train_losses_step_mean.append(extrema_dict["min_train_loss_step"])
        min_train_losses_step_std.append(extrema_dict["min_train_loss_step_std"])
        min_test_losses_mean.append(extrema_dict["min_test_loss"])
        min_test_losses_std.append(extrema_dict["min_test_loss_std"])
        min_test_losses_step_mean.append(extrema_dict["min_test_loss_step"])
        min_test_losses_step_std.append(extrema_dict["min_test_loss_step_std"])

        max_train_accs_mean.append(extrema_dict["max_train_accuracy"])
        max_train_accs_std.append(extrema_dict["max_train_accuracy_std"])
        max_train_accs_step_mean.append(extrema_dict["max_train_accuracy_step"])
        max_train_accs_step_std.append(extrema_dict["max_train_accuracy_step_std"])
        max_test_accs_mean.append(extrema_dict["max_test_accuracy"])
        max_test_accs_std.append(extrema_dict["max_test_accuracy_std"])
        max_test_accs_step_mean.append(extrema_dict["max_test_accuracy_step"])
        max_test_accs_step_std.append(extrema_dict["max_test_accuracy_step_std"])

    # now we just plot each one.
    rows, cols = 4, 1
    figsize = (6, 4)
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

    # j is between train and val, i: loss/acc/tf_l/tf_a
    for i in range(rows):
        ax = fig.add_subplot(rows, cols, i + 1)

        if i == 0:
            # min loss
            ax.plot(
                wds,
                np.array(min_train_losses_mean),
                label="train",
                color=color_1,
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(min_train_losses_mean) - np.array(min_train_losses_std),
                np.array(min_train_losses_mean) + np.array(min_train_losses_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                wds,
                np.array(min_test_losses_mean),
                label="eval",
                color=color_2,
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(min_test_losses_mean) - np.array(min_test_losses_std),
                np.array(min_test_losses_mean) + np.array(min_test_losses_std),
                color=color_2,
                alpha=0.2,
            )
            ax.set_yscale("log")
            ax.legend(fontsize=fontsize)
        elif i == 1:
            # max acc
            ax.plot(
                wds,
                np.array(max_train_accs_mean),
                label="train",
                color=color_1,
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(max_train_accs_mean) - np.array(max_train_accs_std),
                np.array(max_train_accs_mean) + np.array(max_train_accs_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                wds,
                np.array(max_test_accs_mean),
                label="eval",
                color=color_2,
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(max_test_accs_mean) - np.array(max_test_accs_std),
                np.array(max_test_accs_mean) + np.array(max_test_accs_std),
                color=color_2,
                alpha=0.2,
            )
        elif i == 2:
            # min loss step
            ax.plot(
                wds,
                np.array(min_train_losses_step_mean),
                color=color_1,
                label="train",
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(min_train_losses_step_mean)
                - np.array(min_train_losses_step_std),
                np.array(min_train_losses_step_mean)
                + np.array(min_train_losses_step_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                wds,
                np.array(min_test_losses_step_mean),
                color=color_2,
                label="eval",
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(min_test_losses_step_mean)
                - np.array(min_test_losses_step_std),
                np.array(min_test_losses_step_mean)
                + np.array(min_test_losses_step_std),
                color=color_2,
                alpha=0.2,
            )
        else:
            # max acc step
            ax.plot(
                wds,
                np.array(max_train_accs_step_mean),
                color=color_1,
                label="train",
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(max_train_accs_step_mean) - np.array(max_train_accs_step_std),
                np.array(max_train_accs_step_mean) + np.array(max_train_accs_step_std),
                color=color_1,
                alpha=0.2,
            )

            ax.plot(
                wds,
                np.array(max_test_accs_step_mean),
                color=color_2,
                label="eval",
                lw=2.0,
            )
            ax.fill_between(
                wds,
                np.array(max_test_accs_step_mean) - np.array(max_test_accs_step_std),
                np.array(max_test_accs_mean) + np.array(max_test_accs_step_std),
                color=color_2,
                alpha=0.2,
            )

        # if i == 3 we set the xlabel
        if i == 3:
            ax.set_xlabel("Weight decay", fontsize=fontsize)

        # if i == 0 we set the ylabel
        if i == 0:
            ylabel = "Min loss"
        elif i == 1:
            ylabel = "Max accuracy"
        elif i == 2:
            ylabel = "Min loss step"
        else:
            ylabel = "Max accuracy step"

        ax.set_ylabel(ylabel, fontsize=fontsize)

    plt.savefig(
        os.path.join(args.log_dir, "q7p2.pdf"),
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment for assignment 2.")

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--p",
        type=int,
        default=31,
        help="maximum number of digits in the arithmetic expression (default: %(default)s).",
    )
    data.add_argument(
        "--operator",
        type=str,
        default="+",
        choices=["+", "-", "*", "/"],
        help="arithmetic operator to use (default: %(default)s).",
    )
    data.add_argument(
        "--r_train",
        type=float,
        default=0.5,
        help="ratio of training data (default: %(default)s).",
    )
    data.add_argument(
        "--operation_orders",
        type=int,
        nargs="+",
        choices=[2, 3, [2, 3]],
        default=[2],
        help="list of orders of operations to use (default: %(default)s).",
    )
    data.add_argument(
        "--train_batch_size",
        type=int,
        default=512,
        help="batch size for training (default: %(default)s).",
    )
    data.add_argument(
        "--eval_batch_size",
        type=int,
        default=2**12,
        help="batch size for evaluation (default: %(default)s).",
    )
    data.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of processes to use for data loading (default: %(default)s).",
    )

    # Model
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "gpt"],
        help="name of the model to run (default: %(default)s).",
    )
    model.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="number of heads in the  transformer model (default: %(default)s).",
    )
    model.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="number of layers in the model (default: %(default)s).",
    )
    model.add_argument(
        "--embedding_size",
        type=int,
        default=2**7,
        help="embeddings dimension (default: %(default)s).",
    )
    model.add_argument(
        "--hidden_size",
        type=int,
        default=2**7,
        help="hidden size of the lstm model (default: %(default)s).",
    )
    model.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout rate (default: %(default)s).",
    )
    model.add_argument(
        "--share_embeddings",
        type=bool_flag,
        default=False,
        help="share embeddings between the embedding and the classifier (default: %(default)s).",
    )
    model.add_argument(
        "--bias_classifier",
        type=bool_flag,
        default=True,
        help="use bias in the classifier (default: %(default)s).",
    )

    # Optimization
    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="optimizer name (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate for the optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for the SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=1e-0,
        help="weight decay (default: %(default)s).",
    )

    # Training
    training = parser.add_argument_group("Training")
    training.add_argument(
        "--n_steps",
        type=int,
        default=10**4 + 1,
        help="number of training steps (default: %(default)s).",
    )
    training.add_argument(
        "--eval_first",
        type=int,
        default=10**2,
        help="Evaluate the model continuously for the first n steps (default: %(default)s).",
    )
    training.add_argument(
        "--eval_period",
        type=int,
        default=10**2,
        help="Evaluate the model every n steps (default: %(default)s).",
    )
    training.add_argument(
        "--print_step",
        type=int,
        default=10**2,
        help="print the training loss every n steps (default: %(default)s).",
    )
    training.add_argument(
        "--save_model_step",
        type=int,
        default=10**3,
        help="save the model every n steps (default: %(default)s).",
    )
    training.add_argument(
        "--save_statistic_step",
        type=int,
        default=10**3,
        help="save the statistics every n steps (default: %(default)s).",
    )

    # Experiment & Miscellaneous
    misc = parser.add_argument_group("Experiment & Miscellaneous")
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to store tensors on (default: %(default)s).",
    )
    misc.add_argument(
        "--exp_id",
        type=int,
        default=0,
        help="experiment id (default: %(default)s).",
    )
    misc.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="experiment name (default: %(default)s).",
    )
    misc.add_argument(
        "--log_dir",
        type=str,
        default="../logs",
        help="directory to save the logs (default: %(default)s).",
    )
    misc.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: %(default)s).",
    )
    misc.add_argument(
        "--verbose", action="store_true", help="print additional information."
    )
    misc.add_argument(
        "--multiple",
        action="store_true",
        help="whether to train multiple models (seeds 0, 42)",
    )

    args = parser.parse_args()

    # q3p1_plot_losses_and_accuracies(args)
    # q3p2_plotting(args)
    # q4p2_plotting(args)
    # q5p1_plotting(args)
    # q5p2_plotting(args)
    # q6p1_plotting(args)
    # q6p2_plotting(args)
    # q7p1_plotting(args)
    q7p2_plotting(args)
