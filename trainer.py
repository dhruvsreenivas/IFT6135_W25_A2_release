import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

########################################################################################
########################################################################################


def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction="mean"):
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    The equation can be :
        - "[BOS] [a] [+] [b] [=] [r] [EOS] [PAD] [PAD]", in that case target is "[a] [+] [b] [=] [r] [EOS] [PAD] [PAD]"
        - "[BOS] [a] [+] [b] [+] [c] [=] [r] [EOS]", in that case target is "[a] [+] [b] [+] [c] [=] [r] [EOS]"

    Let :
        - B : batch size
        - S : sequence length
        - V : vocabulary size

    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        A tensor containing the logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        A tensor containing the target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence (each sample has exactly one '=').
    mask : torch.LongTensor of shape (B, S)
        A mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        - 'none': no reduction will be applied
        - 'mean': average the output of the batch dimension.
        - 'sum': sum the output of the batch dimension.

    Returns
    -------
    loss : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The negative log-likelihood loss computed over the valid (non-PAD) RHS tokens.
    accuracy : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The accuracy over the batch where a sequence is counted as correct only if
        all valid RHS tokens are predicted correctly.
    """
    # ==========================
    # TODO: Write your code here
    # ==========================

    seq_len = targets.size(1)

    # first get log P from Y -> [bs, seq, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # now, we gather along the vocab dim -> [bs, seq]
    target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)

    # now we need to get the `relevant` target log probs -- between = and pad
    # basically need to set an `eq_mask` that is zero whenever the token is before the eq_positions and 1 otherwise
    eq_mask = (
        1 - (torch.arange(seq_len).unsqueeze(0) <= eq_positions.unsqueeze(1)).long()
    )

    # now we take an AND between `eq_mask` and `mask`
    rhs_mask = (mask * eq_mask).to(target_log_probs.device)

    # finally compute masked loss -> unreduced loss is of shape [bs]
    unreduced_loss = -(target_log_probs * rhs_mask.float()).sum(
        -1
    ) / rhs_mask.float().sum(-1)
    if reduction == "none":
        loss = unreduced_loss
    elif reduction == "mean":
        loss = unreduced_loss.mean()
    else:
        loss = unreduced_loss.sum()

    # compute accuracy -> # [bs, seq]
    argmax_indices = torch.argmax(logits, -1)
    unreduced_accuracy = (argmax_indices == targets).float()

    # now, we're fully right if we're equal to the mask sum -- basically whenever we're valid, we should have a 1
    unreduced_accuracy = (unreduced_accuracy * rhs_mask.float()).sum(
        -1
    ) == rhs_mask.float().sum(-1)
    unreduced_accuracy = unreduced_accuracy.float()

    if reduction == "none":
        accuracy = unreduced_accuracy
    elif reduction == "mean":
        accuracy = unreduced_accuracy.mean()
    else:
        accuracy = unreduced_accuracy.sum()

    return loss, accuracy


########################################################################################
########################################################################################


@torch.no_grad()
def eval_model(model, loader, device, q4=False):
    model.eval()
    acc = 0
    loss = 0
    n = 0

    if q4:
        binary_loss = 0
        binary_acc = 0
        ternary_loss = 0
        ternary_acc = 0

        binary_n = 0
        ternary_n = 0

    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch  # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x)  # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(
            logits, batch_y, eq_positions, mask
        )
        n += batch_x.shape[0]
        loss += batch_loss.cpu().item() * batch_x.shape[0]
        acc += batch_acc.cpu() * batch_x.shape[0]

        # add loss/acc for binary/ternary ops if required
        # hacky but minimal change to the original code
        if q4:
            unreduced_batch_loss, unreduced_batch_acc = get_loss_and_accuracy(
                logits, batch_y, eq_positions, mask, reduction="none"
            )  # [bs]
            binary_mask = (eq_positions == 3).to(batch_x.device).float()
            ternary_mask = (eq_positions == 5).to(batch_x.device).float()

            binary_batch_loss = (unreduced_batch_loss * binary_mask).sum()
            ternary_batch_loss = (unreduced_batch_loss * ternary_mask).sum()

            binary_batch_acc = (unreduced_batch_acc * binary_mask).sum()
            ternary_batch_acc = (unreduced_batch_acc * ternary_mask).sum()

            binary_batch_n = binary_mask.sum()
            ternary_batch_n = ternary_mask.sum()

            # now add to global stuff
            binary_loss += binary_batch_loss.cpu().item()
            ternary_loss += ternary_batch_loss.cpu().item()
            binary_acc += binary_batch_acc.cpu().item()
            ternary_acc += ternary_batch_acc.cpu().item()
            binary_n += binary_batch_n.cpu().item()
            ternary_n += ternary_batch_n.cpu().item()

    ##########
    # You can add more metrics in the dictionary (e.g., l2 norm of the parameters, etc.)
    ##########

    param_l2_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_l2_norm += (p**2).sum()

    param_l2_norm = torch.sqrt(param_l2_norm)

    ##############
    # grab the number of parameters of the model
    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_embedding_params = sum(
        p.numel() for p in model.embedding.parameters() if p.requires_grad
    )
    num_params = num_total_params - num_embedding_params

    logs = {
        "loss": loss / n,
        "accuracy": acc / n,
        "num-params": num_params,
        "l2-norm": param_l2_norm,
    }
    if q4:
        logs.update(
            {
                "binary-loss": binary_loss / binary_n,
                "binary-accuracy": binary_acc / binary_n,
                "ternary-loss": ternary_loss / ternary_n,
                "ternary-accuracy": ternary_acc / ternary_n,
            }
        )

    return logs


########################################################################################
########################################################################################


def train(
    model,
    train_loader,
    train_loader_for_eval,
    test_loader,
    optimizer,
    scheduler,
    device,
    exp_name: str,
    checkpoint_path: str,
    n_steps: int,
    eval_first: int = 0,
    eval_period: int = 1,
    print_step: int = 1,
    save_model_step: int = 1,
    save_statistic_step: int = 1,
    verbose=True,
    q4=False,
):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    exp_name (str) : experiment name
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    n_steps (int) : Number of training steps
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    n_steps = total_epochs * len(train_loader)

    if verbose:
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############

    all_metrics = defaultdict(lambda: [])  # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: [])  # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: [])  # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, device, q4=q4)
    for k, v in train_statistics.items():
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device, q4=q4)
    for k, v in test_statistics.items():
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0

    ######################
    # Save model
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(
        state,
        f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth",
    )

    ##############

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if verbose:
        to_print = "\n" + " | ".join(
            f"Train {k} : {v:.6f}" for k, v in train_statistics.items()
        )
        to_print += " | " + " | ".join(
            f"Test {k} : {v:.6f}" for k, v in test_statistics.items()
        )
        to_print += f" | lr = {current_lr}"
        print(to_print)

    ##############

    cur_step = 1
    tol_step = 0

    for epoch in tqdm(range(1, total_epochs + 1), desc="Training", total=total_epochs):

        start_time = time.time()

        for i, batch in enumerate(train_loader):
            batch_x, batch_y, eq_positions, mask = batch  # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x)  # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================
            # TODO: Write your code here
            # ==========================

            # scheduler.step()
            # current_lr = scheduler.optimizer.param_groups[0]["lr"]

            # ==========================
            # ==========================

            if (
                cur_step in [1, n_steps]
                or cur_step % eval_period == 0
                or cur_step <= eval_first
            ):
                train_statistics = eval_model(
                    model, train_loader_for_eval, device, q4=q4
                )
                for k, v in train_statistics.items():
                    all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device, q4=q4)
                for k, v in test_statistics.items():
                    all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

            if verbose and (cur_step in [1, n_steps] or cur_step % print_step == 0):
                to_print = "\n" + " | ".join(
                    f"Train {k} : {v:.6f}" for k, v in train_statistics.items()
                )
                to_print += " | " + " | ".join(
                    f"Test {k} : {v:.6f}" for k, v in test_statistics.items()
                )
                to_print += f" | lr = {current_lr}"
                print(to_print)

            if (
                cur_step in [1, n_steps]
                or cur_step % save_model_step == 0
                or cur_step <= eval_first
            ):
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(
                    state,
                    f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth",
                )

            if cur_step in [1, n_steps] or cur_step % save_statistic_step == 0:
                # to_save = {k:v for k, v in all_metrics.items()}
                to_save = {
                    k: dict(v) if isinstance(v, defaultdict) else v
                    for k, v in all_metrics.items()
                }  # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

            cur_step += 1

        # ==========================
        # TODO: Write your code here
        # ==========================

        # scheduler.step()
        # current_lr = scheduler.optimizer.param_groups[0]["lr"]

        # ==========================
        # ==========================

        ##############
        # You can implement early stopping here.
        # That is, if the model does not improve for a certain number of steps, you can stop the training.
        ##############

        end_time = time.time()
        elapsed_time = end_time - start_time
        if verbose:
            print(f"Elapsed time for one step : {elapsed_time} seconds")

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(
        state,
        f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth",
    )

    train_statistics = eval_model(model, train_loader_for_eval, device, q4=q4)
    for k, v in train_statistics.items():
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device, q4=q4)
    for k, v in test_statistics.items():
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {
        k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()
    }  # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics
