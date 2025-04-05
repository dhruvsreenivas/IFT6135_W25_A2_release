import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch.utils.data import DataLoader

from data import get_arithmetic_dataset
from gpt import GPT
from run_exp import bool_flag

DEFAULT_PATH = "/network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/gpt/default/0"


def get_args():
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
    data.add_argument(
        "--q4_modified", action="store_true", help="whether to modify dataset for q4"
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
    return args


def plot_attention_weights(args):
    """Plots attention weights of the transformer for a given set of 2 datapoints."""

    # load in dataset
    (train_dataset, _), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(
        args.p,
        args.p,
        args.operator,
        args.r_train,
        args.operation_orders,
        is_symmetric=False,
        shuffle=True,
        seed=args.seed,
    )
    vocabulary_size = len(tokenizer)

    # initialize model first
    model = GPT(
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        embedding_size=args.embedding_size,
        vocabulary_size=vocabulary_size,
        sequence_length=MAX_LENGTH,
        multiplier=4,
        dropout=args.dropout,
        non_linearity="gelu",
        padding_index=padding_index,
        bias_attention=True,
        bias_classifier=args.bias_classifier,
        share_embeddings=args.share_embeddings,
    )

    # load in model weights
    path = os.path.join(
        DEFAULT_PATH,
        "default_state_10002_acc=0.9625779986381531_loss=0.07061094790697098.pth",
    )
    state_dicts = torch.load(path, map_location="cpu")

    model.load_state_dict(state_dicts["model_state_dict"])
    model.eval()

    # now grab the first batch of data
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # we grab the attention heads for the first batch
    batch = next(iter(train_dataloader))
    batch_x = batch[0]  # [2, seq_len]

    # now we decode to get the actual tokens
    batch_tokens = [tokenizer.decode(dp) for dp in batch_x]
    torch.save(batch_tokens, "tokens.pt")

    _, (_, attn_weights) = model(
        batch_x
    )  # [2, num_layers, num_heads, seq_len, seq_len]
    torch.save(attn_weights, "attn_weights.pt")

    for b in range(2):
        # create figure for this particular datapoint
        rows, cols = args.num_layers, args.num_heads
        figsize = (6, 4)
        fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))

        source = batch_tokens[b].split(" ")

        num = 1
        for i in range(args.num_layers):
            for j in range(args.num_heads):
                ax = fig.add_subplot(rows, cols, num)
                # get the heads for this particular batch.
                layer_head_attn_weights = (
                    attn_weights[b, i, j, :, :].cpu().numpy()
                )  # [seq_len, seq_len]
                assert layer_head_attn_weights.shape == (6, 6)

                plt.imshow(layer_head_attn_weights)
                plt.title(f"Layer {i}, Head {j}")
                plt.ylabel("x")
                plt.xlabel("x (context)")
                plt.colorbar()

                ax.set_xticks(np.arange(len(source)), labels=source)
                ax.set_yticks(np.arange(len(source)), labels=source)

                num += 1

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "/network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/gpt/default",
                f"q8p1_batch_{b}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )
        plt.clf()


if __name__ == "__main__":
    args = get_args()

    plot_attention_weights(args)
