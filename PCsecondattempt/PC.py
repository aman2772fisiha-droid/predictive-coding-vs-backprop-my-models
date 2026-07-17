"""
pc_mnist_4_hidden.py

A real MNIST predictive-coding example with four hidden layers.

This is deliberately written to look close to the pseudocode:

    predicted_input  = hidden @ W_x + b_x
    predicted_hidden = label_one_hot @ W_y + b_h
    eps_input        = input - predicted_input
    eps_hidden       = hidden - predicted_hidden
    hidden           = hidden - eta * dE/dhidden

The model is generative/top-down during training:

    label y  ---> hidden h ---> image x

During training, x and y are clamped, and h is inferred by an inner loop.
During testing, y is unknown, so we try every possible digit 0..9, infer h for
that candidate label, compute its free energy, and choose the label with the
lowest energy.

This is slower than standard backprop because every batch uses an inner
inference loop.

Run examples:

    python pc_mnist_4_hidden.py --epochs 3 --train_subset 10000 --test_subset 2000

Small debug run:

    python pc_mnist_4_hidden.py --epochs 1 --train_subset 1024 --test_subset 256 --debug_first_batch

No MNIST download smoke test:

    python pc_mnist_4_hidden.py --smoke_test
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ============================================================
# Configuration
# ============================================================


@dataclass
class PCConfig:
    # Data/model dimensions
    image_dim: int = 28 * 28
    num_classes: int = 10
    hidden_dim: list[int] = field(default_factory=lambda: [392,196,98,49])

    # Data loading
    batch_size: int = 128

    # Hidden-state inference loop
    inference_steps: int = 20
    eval_inference_steps: int = 25
    hidden_lr: float = 0.5
    hidden_clip: float = 2.0

    # Energy weights
    recon_weight: float = 1.0
    hidden_weight: float = 10.0

    # Slow weight learning
    weight_lr: float = 0.005
    weight_decay: float = 1e-5

    # Use local PC/Hebbian-style updates or Adam on the PC energy.
    # local is closest to the pseudocode.
    weight_update: str = "local"
    adam_lr: float = 1e-3


# ============================================================
# Utility helpers
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#Makes our experiment reproducible.


def flatten_mnist(images: torch.Tensor) -> torch.Tensor:
    """[B, 1, 28, 28] -> [B, 784]."""
    return images.view(images.size(0), -1)
# So every image becomes one long vector.



def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()
# In predictive coding, these one-hot vectors are often used as the clamped output layer during training



def maybe_subset(dataset, subset_size: Optional[int], seed: int):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)
#This lets you train on only part of the dataset.


# ============================================================
# four-hidden-layer predictive-coding model
# ============================================================


class FourHiddenPC(nn.Module):
    """
4-hidden-layer predictive coding network.

Shapes:
    x_flat     : [B,784]
    y_one_hot  : [B,10]

Hidden states:
    h1 : [B,392]
    h2 : [B,196]
    h3 : [B,98]
    h4 : [B,49]

Generative predictions:
    μx = h1 @ Wx + bx
    μ1 = h2 @ W1 + b1
    μ2 = h3 @ W2 + b2
    μ3 = h4 @ W3 + b3
    μ4 = y  @ W4 + b4

Prediction errors:
    εx = x - μx
    ε1 = h1 - μ1
    ε2 = h2 - μ2
    ε3 = h3 - μ3
    ε4 = h4 - μ4
"""

    def __init__(self, config: PCConfig):
        super().__init__()
        self.config = config

        h1, h2, h3, h4 = config.hidden_dim

        # -----------------------------
        # Generative weights
        # -----------------------------

        # h1 -> input
        self.W_x = nn.Parameter(
            torch.randn(h1, config.image_dim)
            * (1.0 / math.sqrt(h1))
        )
        self.b_x = nn.Parameter(
            torch.zeros(config.image_dim)
        )

        # h2 -> h1
        self.W_1 = nn.Parameter(
            torch.randn(h2, h1)
            * (1.0 / math.sqrt(h2))
        )
        self.b_1 = nn.Parameter(
            torch.zeros(h1)
        )

        # h3 -> h2
        self.W_2 = nn.Parameter(
            torch.randn(h3, h2)
            * (1.0 / math.sqrt(h3))
        )
        self.b_2 = nn.Parameter(
            torch.zeros(h2)
        )

        # h4 -> h3
        self.W_3 = nn.Parameter(
            torch.randn(h4, h3)
            * (1.0 / math.sqrt(h4))
        )
        self.b_3 = nn.Parameter(
            torch.zeros(h3)
        )

        # label -> h4
        self.W_4 = nn.Parameter(
            torch.randn(config.num_classes, h4)
            * (1.0 / math.sqrt(config.num_classes))
        )
        self.b_4 = nn.Parameter(
            torch.zeros(h4)
        )

    def predict_input(self, h1: torch.Tensor) -> torch.Tensor:
        """
        h1 predicts the input x.
        h1: [B,392]
        output: [B,784]
        """
        return h1 @ self.W_x + self.b_x


    def predict_h1(self, h2: torch.Tensor) -> torch.Tensor:
        """
        h2 predicts h1.
        h2: [B,196]
        output: [B,392]
        """
        return h2 @ self.W_1 + self.b_1


    def predict_h2(self, h3: torch.Tensor) -> torch.Tensor:
        """
        h3 predicts h2.
        h3: [B,98]
        output: [B,196]
        """
        return h3 @ self.W_2 + self.b_2


    def predict_h3(self, h4: torch.Tensor) -> torch.Tensor:
        """
        h4 predicts h3.
        h4: [B,49]
        output: [B,98]
        """
        return h4 @ self.W_3 + self.b_3


    def predict_h4(self, y_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Label predicts h4.
        y: [B,10]
        output: [B,49]
        """
        return y_one_hot @ self.W_4 + self.b_4

    def predict_h4_from_label(self, y_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Predict h4 from the label.

        y_one_hot: [B,10]
        output:    [B,49]
        """
        return y_one_hot @ self.W_4 + self.b_4

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================ 
# PC energy and hidden-state inference
#============================================================


def pc_energy_parts(
    model: FourHiddenPC,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    hidden_states: list[torch.Tensor],
    config: PCConfig,
) -> Dict[str, torch.Tensor]:
    """
    Computes predictive coding energy for a 4-hidden-layer model.

    Architecture:

        y -> h4 -> h3 -> h2 -> h1 -> x


    Predictions:

        x_hat = h1 W_x + b_x
        h1_hat = h2 W_1 + b_1
        h2_hat = h3 W_2 + b_2
        h3_hat = h4 W_3 + b_3
        h4_hat = y W_4 + b_4
    """

    h1, h2, h3, h4 = hidden_states

    # Top-down predictions
    predicted_input = model.predict_input(h1)

    predicted_h1 = model.predict_h1(h2)

    predicted_h2 = model.predict_h2(h3)

    predicted_h3 = model.predict_h3(h4)

    predicted_h4 = model.predict_h4(y_one_hot)


    # Prediction errors
    eps_input = x_flat - predicted_input

    eps_h1 = h1 - predicted_h1

    eps_h2 = h2 - predicted_h2

    eps_h3 = h3 - predicted_h3

    eps_h4 = h4 - predicted_h4


    # Energy terms

    recon = 0.5 * eps_input.pow(2).mean(dim=1)

    hidden_energy = (
        0.5 * eps_h1.pow(2).mean(dim=1)
        +0.5 * eps_h2.pow(2).mean(dim=1)
        +0.5 * eps_h3.pow(2).mean(dim=1)
        +0.5 * eps_h4.pow(2).mean(dim=1)
    )


    total = (
        config.recon_weight * recon
        +
        config.hidden_weight * hidden_energy
    )


    return {
        "total_per_sample": total,
        "recon_per_sample": recon,
        "hidden_per_sample": hidden_energy,

        "eps_input": eps_input,
        "eps_h1": eps_h1,
        "eps_h2": eps_h2,
        "eps_h3": eps_h3,
        "eps_h4": eps_h4,

        "predicted_input": predicted_input,
        "predicted_h1": predicted_h1,
        "predicted_h2": predicted_h2,
        "predicted_h3": predicted_h3,
        "predicted_h4": predicted_h4,
    }

# ====================================================
#now the inference function
# ====================================================
@torch.no_grad()
def infer_hidden_manual(
    model: FourHiddenPC,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    config: PCConfig,
    steps: Optional[int] = None,
    collect_trace: bool = False,
):

    if steps is None:
        steps = config.inference_steps


    h1_dim, h2_dim, h3_dim, h4_dim = config.hidden_dim


    # Initial states from label prediction
    h4 = model.predict_h4(y_one_hot).detach().clone()

    h3 = model.predict_h3(h4).detach().clone()

    h2 = model.predict_h2(h3).detach().clone()

    h1 = model.predict_h1(h2).detach().clone()


    hidden_states = [h1, h2, h3, h4]


    trace = [] if collect_trace else None


    for step in range(steps + 1):

        parts = pc_energy_parts(
            model,
            x_flat,
            y_one_hot,
            hidden_states,
            config
        )


        if collect_trace:
            trace.append(
                {
                    "step": step,
                    "energy":
                        parts["total_per_sample"].mean().item()
                }
            )


        if step == steps:
            break


        eps_x = parts["eps_input"]

        eps1 = parts["eps_h1"]

        eps2 = parts["eps_h2"]

        eps3 = parts["eps_h3"]

        eps4 = parts["eps_h4"]


        #
        # Hidden gradients
        #

        # h1 receives:
        # reconstruction error from x
        # error from h2 prediction

        grad_h1 = (
            -config.recon_weight *
            (eps_x @ model.W_x.t())
            +
            config.hidden_weight * eps1
        )


        # h2 receives:
        # error from h1 prediction
        # error from h3 prediction

        grad_h2 = (
            -config.hidden_weight *
            (eps1 @ model.W_1.t())
            +
            config.hidden_weight * eps2
        )


        # h3 receives:

        grad_h3 = (
            -config.hidden_weight *
            (eps2 @ model.W_2.t())
            +
            config.hidden_weight * eps3
        )


        # h4 receives:

        grad_h4 = (
            -config.hidden_weight *
            (eps3 @ model.W_3.t())
            +
            config.hidden_weight * eps4
        )


        # Gradient descent updates

        h1 -= config.hidden_lr * grad_h1
        h2 -= config.hidden_lr * grad_h2
        h3 -= config.hidden_lr * grad_h3
        h4 -= config.hidden_lr * grad_h4


        # non-linear predictive coding neurons

        h1.tanh_()
        h2.tanh_()
        h3.tanh_()
        h4.tanh_()


        hidden_states = [h1, h2, h3, h4]


    final_parts = pc_energy_parts(
        model,
        x_flat,
        y_one_hot,
        hidden_states,
        config
    )


    stats = {
        "energy":
            final_parts["total_per_sample"].mean().item(),

        "recon":
            final_parts["recon_per_sample"].mean().item(),

        "hidden":
            final_parts["hidden_per_sample"].mean().item(),
    }


    return (
        [h.detach() for h in hidden_states],
        stats,
        trace
    )
# ============================================================
# Weight learning
# ============================================================


@torch.no_grad()
def local_pc_weight_update(
    model: FourHiddenPC,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    hidden_states: list[torch.Tensor],
    config: PCConfig,
) -> Dict[str, float]:

    """
    Local predictive coding weight update.

    Generative hierarchy:

        y -> h4 -> h3 -> h2 -> h1 -> x


    Local updates:

        W_x += lr * h1.T @ eps_x

        W_1 += lr * h2.T @ eps_h1

        W_2 += lr * h3.T @ eps_h2

        W_3 += lr * h4.T @ eps_h3

        W_4 += lr * y.T  @ eps_h4
    """


    batch_size = x_flat.size(0)

    h1, h2, h3, h4 = hidden_states


    # Calculate current prediction errors

    parts = pc_energy_parts(
        model,
        x_flat,
        y_one_hot,
        hidden_states,
        config,
    )


    eps_x  = parts["eps_input"]

    eps_h1 = parts["eps_h1"]

    eps_h2 = parts["eps_h2"]

    eps_h3 = parts["eps_h3"]

    eps_h4 = parts["eps_h4"]


    lr = config.weight_lr


    # -----------------------------------------
    # Weight decay
    # -----------------------------------------

    if config.weight_decay > 0:

        decay = 1.0 - lr * config.weight_decay

        model.W_x.mul_(decay)
        model.W_1.mul_(decay)
        model.W_2.mul_(decay)
        model.W_3.mul_(decay)
        model.W_4.mul_(decay)



    # -----------------------------------------
    # Local Hebbian / prediction-error updates
    # -----------------------------------------


    # h1 -> x

    model.W_x.sub_(
        lr
        * config.recon_weight
        * (h1.t() @ eps_x)
        / batch_size
    )

    model.b_x.sub_(
        lr
        * config.recon_weight
        * eps_x.mean(dim=0)
    )



    # h2 -> h1

    model.W_1.sub_(
        lr
        * config.hidden_weight
        * (h2.t() @ eps_h1)
        / batch_size
    )

    model.b_1.add_(
        lr
        * config.hidden_weight
        * eps_h1.mean(dim=0)
    )



    # h3 -> h2

    model.W_2.sub_(
        lr
        * config.hidden_weight
        * (h3.t() @ eps_h2)
        / batch_size
    )

    model.b_2.add_(
        lr
        * config.hidden_weight
        * eps_h2.mean(dim=0)
    )



    # h4 -> h3

    model.W_3.sub_(
        lr
        * config.hidden_weight
        * (h4.t() @ eps_h3)
        / batch_size
    )

    model.b_3.add_(
        lr
        * config.hidden_weight
        * eps_h3.mean(dim=0)
    )



    # y -> h4

    model.W_4.sub_(
        lr
        * config.hidden_weight
        * (y_one_hot.t() @ eps_h4)
        / batch_size
    )

    model.b_4.add_(
        lr
        * config.hidden_weight
        * eps_h4.mean(dim=0)
    )



    return {

        "energy":
            parts["total_per_sample"].mean().item(),

        "recon":
            parts["recon_per_sample"].mean().item(),

        "hidden":
            parts["hidden_per_sample"].mean().item(),

        "eps_input_rms":
            eps_x.pow(2).mean().sqrt().item(),

        "eps_h1_rms":
            eps_h1.pow(2).mean().sqrt().item(),

        "eps_h2_rms":
            eps_h2.pow(2).mean().sqrt().item(),

        "eps_h3_rms":
            eps_h3.pow(2).mean().sqrt().item(),

        "eps_h4_rms":
            eps_h4.pow(2).mean().sqrt().item(),
    }

# ============================================================
# Prediction by free energy
# ============================================================


@torch.no_grad()
def predict_by_free_energy(
    model: FourHiddenPC,
    images: torch.Tensor,
    config: PCConfig,
    steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict labels without a feedforward classifier.

    For each possible digit c:

        1. Clamp y = c

        2. Infer hidden states:
           
           h1, h2, h3, h4

        3. Compute final predictive-coding energy:

           E(x,h,y=c)

        4. Choose class with lowest energy.


    Returns:

        predictions:
            [B]

        energies:
            [B,10]
    """

    x_flat = flatten_mnist(images)

    batch_size = x_flat.size(0)

    energies = []


    for candidate in range(config.num_classes):

        # Create candidate labels
        labels = torch.full(
            (batch_size,),
            candidate,
            dtype=torch.long,
            device=images.device,
        )


        # Convert to one-hot
        y_oh = one_hot(
            labels,
            config.num_classes
        )


        # Infer all hidden layers
        hidden_states, _, _ = infer_hidden_manual(
            model,
            x_flat,
            y_oh,
            config,
            steps=(
                steps
                if steps is not None
                else config.eval_inference_steps
            ),
            collect_trace=False,
        )


        # Compute final energy for this candidate label
        parts = pc_energy_parts(
            model,
            x_flat,
            y_oh,
            hidden_states,
            config,
        )


        energies.append(
            parts["total_per_sample"]
        )


    # Shape:
    # list of 10 tensors [B]
    #
    # becomes:
    # [B,10]

    energy_matrix = torch.stack(
        energies,
        dim=1
    )


    predictions = energy_matrix.argmin(dim=1)


    return predictions, energy_matrix

# ============================================================
# Data loading
# ============================================================

def build_mnist_loaders(
    config: PCConfig,
    data_dir: str = "./data",
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    num_workers: int = 0,
):
    """
    Build MNIST dataloaders for the predictive coding model.

    Output:
        images: [B,1,28,28]
        labels: [B]

    Later:
        flatten_mnist(images) -> [B,784]
        one_hot(labels,10)    -> [B,10]
    """

    from torchvision import datasets, transforms


    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )


    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )


    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )


    # Optional smaller datasets for debugging
    train_dataset = maybe_subset(
        train_dataset,
        train_subset,
        seed=0,
    )

    test_dataset = maybe_subset(
        test_dataset,
        test_subset,
        seed=1,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


    return (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
    )


# ============================================================
# Training and evaluation
# ============================================================


def print_model_explanation(
    model: FourHiddenPC,
    config: PCConfig
) -> None:

    print("=" * 70)
    print("4-hidden-layer Predictive Coding MNIST model")
    print("=" * 70)

    print(model)
    print()

    print(f"Total parameters: {model.num_parameters():,}")
    print()

    h1_dim, h2_dim, h3_dim, h4_dim = config.hidden_dim

    print("Shape meaning")
    print("-------------")
    print("MNIST image x:           [B,1,28,28]")
    print("Flattened image x_flat:  [B,784]")
    print(f"Label one-hot y:         [B,{config.num_classes}]")
    print(f"h1:                      [B,{h1_dim}]")
    print(f"h2:                      [B,{h2_dim}]")
    print(f"h3:                      [B,{h3_dim}]")
    print(f"h4:                      [B,{h4_dim}]")

    print()

    print("Generative predictions")
    print("----------------------")
    print("x_hat  = h1 @ W_x + b_x")
    print("h1_hat = h2 @ W_1 + b_1")
    print("h2_hat = h3 @ W_2 + b_2")
    print("h3_hat = h4 @ W_3 + b_3")
    print("h4_hat = y  @ W_4 + b_4")

    print()

    print("Prediction errors")
    print("------------------")
    print("eps_x  = x  - x_hat")
    print("eps_h1 = h1 - h1_hat")
    print("eps_h2 = h2 - h2_hat")
    print("eps_h3 = h3 - h3_hat")
    print("eps_h4 = h4 - h4_hat")

    print()

    print("Training difference from backprop")
    print("---------------------------------")
    print(
        "PC: clamp x and y -> infer h1,h2,h3,h4 -> "
        "update local generative weights"
    )
    print(
        "Weights updated: W_x,W_1,W_2,W_3,W_4"
    )

    print("=" * 70)



def debug_first_batch(
    model: FourHiddenPC,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: PCConfig,
) -> None:

    print("\n" + "=" * 70)
    print("Debugging one PC inference loop")
    print("=" * 70)

    x_flat = flatten_mnist(images)
    y_oh = one_hot(labels, config.num_classes)

    print(f"images shape:       {tuple(images.shape)}")
    print(f"x_flat shape:       {tuple(x_flat.shape)}")
    print(f"labels shape:       {tuple(labels.shape)}")
    print(f"y_one_hot shape:    {tuple(y_oh.shape)}")

    hidden_states, stats, trace = infer_hidden_manual(
        model,
        x_flat,
        y_oh,
        config,
        steps=min(config.inference_steps, 10),
        collect_trace=True,
    )

    h1, h2, h3, h4 = hidden_states

    print()
    print(f"h1 shape:            {tuple(h1.shape)}")
    print(f"h2 shape:            {tuple(h2.shape)}")
    print(f"h3 shape:            {tuple(h3.shape)}")
    print(f"h4 shape:            {tuple(h4.shape)}")

    print()
    print("Inner inference trace")
    print("--------------------")

    print("step | energy")
    print("-----+-----------")

    for row in trace:
        print(
            f"{row['step']:4d} | "
            f"{row['energy']:.6f}"
        )

    print("=" * 70 + "\n")



def train_one_epoch(
    model: FourHiddenPC,
    dataloader: DataLoader,
    config: PCConfig,
    device: torch.device,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    log_interval: int = 100,
) -> Dict[str, float]:

    model.train()

    totals = {
        "energy": 0.0,
        "recon": 0.0,
        "hidden": 0.0,
        "eps_input_rms": 0.0,
        "eps_h1_rms": 0.0,
        "eps_h2_rms": 0.0,
        "eps_h3_rms": 0.0,
        "eps_h4_rms": 0.0,
    }

    n_samples = 0


    for batch_idx, (images, labels) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        x_flat = flatten_mnist(images)
        y_oh = one_hot(labels, config.num_classes)


        # ---------------------------------------------
        # Fast loop:
        # infer hidden states while weights are fixed
        # ---------------------------------------------

        hidden_states, _, _ = infer_hidden_manual(
            model,
            x_flat,
            y_oh,
            config,
            steps=config.inference_steps,
            collect_trace=False,
        )


        # ---------------------------------------------
        # Slow loop:
        # update weights
        # ---------------------------------------------

        if config.weight_update == "local":

            update_stats = local_pc_weight_update(
                model,
                x_flat,
                y_oh,
                hidden_states,
                config,
            )


        elif config.weight_update == "adam":

            if optimizer is None:
                raise ValueError(
                    "Adam optimizer required"
                )

            update_stats = adam_pc_weight_update(
                model,
                optimizer,
                x_flat,
                y_oh,
                hidden_states,
                config,
            )


        else:
            raise ValueError(
                f"Unknown weight update {config.weight_update}"
            )


        batch_size = images.size(0)

        n_samples += batch_size


        for key in totals:
            if key in update_stats:
                totals[key] += update_stats[key] * batch_size



        if batch_idx % log_interval == 0:

            eval_images = images[:min(32, batch_size)]
            eval_labels = labels[:min(32, batch_size)]


            preds, _ = predict_by_free_energy(
                model,
                eval_images,
                config,
                steps=min(
                    config.eval_inference_steps,
                    10
                ),
            )


            mini_acc = (
                100.0 *
                (preds == eval_labels)
                .float()
                .mean()
                .item()
            )


            print(
                f"Epoch {epoch:02d} | "
                f"Batch {batch_idx:04d}/{len(dataloader):04d} | "
                f"E={update_stats['energy']:.5f} | "
                f"Recon={update_stats['recon']:.5f} | "
                f"Hidden={update_stats['hidden']:.5f} | "
                f"mini free-energy acc={mini_acc:.1f}%"
            )


    return {
        key: totals[key] / max(1, n_samples)
        for key in totals
    }



@torch.no_grad()
def evaluate(
    model: FourHiddenPC,
    dataloader: DataLoader,
    config: PCConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:

    model.eval()

    total = 0
    correct = 0
    total_energy = 0.0


    for batch_idx, (images, labels) in enumerate(dataloader):

        if max_batches is not None and batch_idx >= max_batches:
            break


        images = images.to(device)
        labels = labels.to(device)


        preds, energy_matrix = predict_by_free_energy(
            model,
            images,
            config,
            steps=config.eval_inference_steps,
        )


        winning_energy = energy_matrix.min(dim=1).values

        total_energy += winning_energy.sum().item()

        correct += (
            preds == labels
        ).sum().item()

        total += labels.numel()


    return {
        "loss": total_energy / max(1,total),
        "accuracy": 100.0 * correct / max(1,total),
    }



@torch.no_grad()
def show_sample_predictions(
    model: FourHiddenPC,
    dataloader: DataLoader,
    config: PCConfig,
    device: torch.device,
    n: int = 10,
) -> None:

    model.eval()

    images, labels = next(iter(dataloader))

    images = images[:n].to(device)
    labels = labels[:n].to(device)


    preds, energies = predict_by_free_energy(
        model,
        images,
        config,
    )


    print("\n" + "=" * 70)
    print("Sample predictions by lowest predictive-coding free energy")
    print("=" * 70)


    for i in range(images.size(0)):

        winning_energy = energies[i, preds[i]].item()

        true_energy = energies[i, labels[i]].item()


        print(
            f"Image {i:02d} | "
            f"Prediction: {preds[i].item()} | "
            f"Ground Truth: {labels[i].item()} | "
            f"E_pred={winning_energy:.5f} | "
            f"E_true={true_energy:.5f}"
        )
# ============================================================
# Smoke test
# ============================================================


def run_smoke_test(device: torch.device) -> None:

    print("Running smoke test with random MNIST-shaped tensors.")


    # Small network for testing
    config = PCConfig(
        hidden_dim=[32,16,8,4],
        inference_steps=5,
        eval_inference_steps=5,
        batch_size=8,
    )


    model = FourHiddenPC(config).to(device)


    # Fake MNIST batch
    images = torch.rand(
        8,
        1,
        28,
        28,
        device=device
    )

    labels = torch.randint(
        0,
        config.num_classes,
        (8,),
        device=device
    )


    x_flat = flatten_mnist(images)

    y_oh = one_hot(
        labels,
        config.num_classes
    )


    # -----------------------------------------
    # Infer hidden states
    # -----------------------------------------

    hidden_states, stats, trace = infer_hidden_manual(
        model,
        x_flat,
        y_oh,
        config,
        collect_trace=True,
    )


    h1, h2, h3, h4 = hidden_states


    # -----------------------------------------
    # Update weights
    # -----------------------------------------

    update_stats = local_pc_weight_update(
        model,
        x_flat,
        y_oh,
        hidden_states,
        config,
    )


    # -----------------------------------------
    # Free-energy prediction
    # -----------------------------------------

    preds, energies = predict_by_free_energy(
        model,
        images,
        config,
    )


    print()

    print(f"images:        {tuple(images.shape)}")

    print(f"x_flat:        {tuple(x_flat.shape)}")

    print(f"y_one_hot:     {tuple(y_oh.shape)}")


    print()

    print(f"h1:            {tuple(h1.shape)}")

    print(f"h2:            {tuple(h2.shape)}")

    print(f"h3:            {tuple(h3.shape)}")

    print(f"h4:            {tuple(h4.shape)}")


    print()

    print(f"energies:      {tuple(energies.shape)}")

    print(f"predictions:   {tuple(preds.shape)}")


    print()

    print(f"final stats:   {stats}")

    print(f"update stats:  {update_stats}")


    print()

    print("Smoke test passed.")


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="4-hidden-layer Predictive Coding MNIST model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50
    )

    parser.add_argument(
        "--eval_inference_steps",
        type=int,
        default=100
    )

    parser.add_argument(
        "--hidden_lr",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--weight_lr",
        type=float,
        default=0.005
    )

    parser.add_argument(
        "--recon_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--hidden_weight",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--train_subset",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--test_subset",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=50
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="pc_checkpoints"
    )

    parser.add_argument(
        "--debug_first_batch",
        action="store_true"
    )

    parser.add_argument(
        "--smoke_test",
        action="store_true"
    )


    return parser.parse_args()



def main() -> None:


    args = parse_args()

    set_seed(args.seed)


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print(f"Using device: {device}")


    if args.smoke_test:

        run_smoke_test(device)

        return



    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = PCConfig(

        image_dim=28*28,

        num_classes=10,

        hidden_dim=[
            392,
            196,
            98,
            49
        ],

        batch_size=args.batch_size,

        inference_steps=args.inference_steps,

        eval_inference_steps=args.eval_inference_steps,

        hidden_lr=args.hidden_lr,

        weight_lr=args.weight_lr,

        recon_weight=args.recon_weight,

        hidden_weight=args.hidden_weight,

        weight_update="local",
    )



    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_dataset, test_dataset, train_loader, test_loader = (
        build_mnist_loaders(
            config,
            data_dir=args.data_dir,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
        )
    )



    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)

    print(
        f"Training images used: {len(train_dataset)}"
    )

    print(
        f"Testing images used:  {len(test_dataset)}"
    )

    first_image, first_label = train_dataset[0]

    print(
        f"First image shape:    {tuple(first_image.shape)}"
    )

    print(
        f"First label:          {first_label}"
    )



    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = FourHiddenPC(config).to(device)


    print_model_explanation(
        model,
        config
    )



    print(
        f"Using local PC error-based weight updates "
        f"with lr={config.weight_lr}"
    )



    # --------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------

    images, labels = next(iter(train_loader))


    images = images.to(device)

    labels = labels.to(device)



    print(
        "\nInitial one-batch free-energy prediction check"
    )


    init_preds, _ = predict_by_free_energy(
        model,
        images[:min(32, images.size(0))],
        config,
        steps=min(
            config.eval_inference_steps,
            10
        ),
    )


    init_acc = (
        100.0 *
        (
            init_preds ==
            labels[:init_preds.numel()]
        )
        .float()
        .mean()
        .item()
    )


    print(
        f"Initial mini free-energy accuracy: "
        f"{init_acc:.2f}%"
    )



    if args.debug_first_batch:

        debug_first_batch(
            model,
            images,
            labels,
            config
        )



    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    save_dir = Path(args.checkpoint_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    best_accuracy = -1.0


    best_path = (
        save_dir /
        "best_pc_four_hidden.pth"
    )



    print("\n" + "=" * 70)

    print(
        "Starting Predictive Coding Training"
    )

    print("=" * 70)



    for epoch in range(
        1,
        args.epochs + 1
    ):


        train_stats = train_one_epoch(
            model,
            train_loader,
            config,
            device,
            epoch=epoch,
            optimizer=None,
            log_interval=args.log_interval,
        )


        test_stats = evaluate(
            model,
            test_loader,
            config,
            device,
        )



        print(

            f"Epoch [{epoch}/{args.epochs}] | "

            f"Train E: {train_stats['energy']:.5f} | "

            f"Train Recon: {train_stats['recon']:.5f} | "

            f"Train Hidden: {train_stats['hidden']:.5f} | "

            f"Test E: {test_stats['loss']:.5f} | "

            f"Test Acc: {test_stats['accuracy']:.2f}%"

        )



        if test_stats["accuracy"] > best_accuracy:


            best_accuracy = test_stats["accuracy"]


            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "config":
                        config.__dict__,

                    "args":
                        vars(args),

                    "best_accuracy":
                        best_accuracy,
                },

                best_path,
            )


            print(
                f"New best PC model saved: "
                f"{best_path} | "
                f"Acc={best_accuracy:.2f}%"
            )



    print("\n" + "=" * 70)

    print("Training Finished")

    print("=" * 70)


    print(
        f"Best Test Accuracy: {best_accuracy:.2f}%"
    )



    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    checkpoint = torch.load(
        best_path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    final_stats = evaluate(
        model,
        test_loader,
        config,
        device
    )



    print("\n" + "=" * 70)

    print("Final Evaluation")

    print("=" * 70)


    print(
        f"Final free-energy loss: "
        f"{final_stats['loss']:.5f}"
    )


    print(
        f"Final accuracy: "
        f"{final_stats['accuracy']:.2f}%"
    )



    show_sample_predictions(
        model,
        test_loader,
        config,
        device,
        n=10
    )



if __name__ == "__main__":

    main()