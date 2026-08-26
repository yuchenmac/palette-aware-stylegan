"""Finite CIELAB palette learning from Section 2.3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from preprocessing.preprocess import srgb_to_lab
from priors.structural_prior import StructuralPrior


@dataclass(frozen=True)
class PaletteCodebook:
    """Sorted CIELAB centroids and their weighted target proportions."""

    centroids: np.ndarray
    proportions: np.ndarray

    def tensors(self, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        return (
            torch.as_tensor(self.centroids, dtype=torch.float32, device=device),
            torch.as_tensor(self.proportions, dtype=torch.float32, device=device),
        )


def weighted_sample_without_replacement(
    values: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the per-patch distance-weighted sampling in Eq. (3)."""

    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) != len(weights):
        raise ValueError("Palette values and weights must have equal length.")
    if len(values) == 0:
        return values.reshape(0, 3), weights.astype(np.float32)
    count = min(int(count), len(values))
    total = weights.sum()
    probabilities = weights / total if total > 0.0 else None
    indices = rng.choice(len(values), size=count, replace=False, p=probabilities)
    return values[indices], weights[indices].astype(np.float32)


class PalettePrior:
    """Learn the finite palette codebook with weighted K-means, Eqs. (3)-(5)."""

    def __init__(
        self,
        k: int = 9,
        max_samples_per_patch: int = 50000,
        random_seed: int = 42,
        max_iter: int = 50,
        tolerance: float = 1.0e-5,
        epsilon: float = 1.0e-8,
    ) -> None:
        self.k = int(k)
        self.max_samples_per_patch = int(max_samples_per_patch)
        self.random_seed = int(random_seed)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.epsilon = float(epsilon)
        if self.k < 1:
            raise ValueError("Palette size must be positive.")

    def sample_patch(
        self,
        lab: np.ndarray,
        flat_fill_mask: np.ndarray,
        distance_weight: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and weight stable flat-fill pixels as specified by Eq. (3)."""

        mask = np.asarray(flat_fill_mask, dtype=bool).squeeze()
        weights = np.asarray(distance_weight, dtype=np.float32).squeeze()
        pixels = np.asarray(lab, dtype=np.float32)[mask]
        selected_weights = weights[mask]
        return weighted_sample_without_replacement(
            pixels, selected_weights, self.max_samples_per_patch, rng
        )

    def fit(
        self,
        lab_patches: Sequence[np.ndarray],
        flat_fill_masks: Sequence[np.ndarray],
        distance_weights: Sequence[np.ndarray],
    ) -> PaletteCodebook:
        """Aggregate patch samples and solve weighted K-means in Eq. (4)."""

        if not (len(lab_patches) == len(flat_fill_masks) == len(distance_weights)):
            raise ValueError("Palette patch inputs must have equal lengths.")
        rng = np.random.default_rng(self.random_seed)
        sample_values: List[np.ndarray] = []
        sample_weights: List[np.ndarray] = []
        for lab, mask, weight in zip(lab_patches, flat_fill_masks, distance_weights):
            values, weights = self.sample_patch(lab, mask, weight, rng)
            if len(values):
                sample_values.append(values)
                sample_weights.append(weights)
        if not sample_values:
            raise ValueError("No flat-fill pixels are available for palette learning.")
        values = np.concatenate(sample_values, axis=0)
        weights = np.concatenate(sample_weights, axis=0).astype(np.float64)
        if len(values) < self.k:
            raise ValueError("The palette sample pool contains fewer pixels than K.")
        centroids, assignments = self._weighted_kmeans(values, weights, rng)
        proportions = np.asarray(
            [weights[assignments == index].sum() for index in range(self.k)],
            dtype=np.float64,
        )
        proportions /= max(proportions.sum(), self.epsilon)
        order = np.argsort(-proportions)
        return PaletteCodebook(
            centroids=centroids[order].astype(np.float32),
            proportions=proportions[order].astype(np.float32),
        )

    def fit_rgb_patches(
        self,
        rgb_patches: Iterable[Union[np.ndarray, Tensor]],
        structural_prior: StructuralPrior,
    ) -> PaletteCodebook:
        """Learn a codebook directly from training-split RGB patches only."""

        labs: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        structural_prior = structural_prior.cpu().eval()
        for patch in rgb_patches:
            array = patch.detach().cpu().numpy() if isinstance(patch, Tensor) else np.asarray(patch)
            if array.ndim == 3 and array.shape[0] == 3:
                array = np.moveaxis(array, 0, -1)
            array = array.astype(np.float32)
            if array.min(initial=0.0) < 0.0:
                array = (array + 1.0) * 0.5
            array = np.clip(array, 0.0, 1.0)
            tensor = (
                torch.from_numpy(np.moveaxis(array, -1, 0))
                .unsqueeze(0)
                .mul(2.0)
                .sub(1.0)
            )
            with torch.no_grad():
                output = structural_prior(tensor)
            labs.append(srgb_to_lab(array))
            masks.append(output.flat_fill_mask[0, 0].cpu().numpy())
            weights.append(output.distance_weight[0, 0].cpu().numpy())
        return self.fit(labs, masks, weights)

    def _weighted_kmeans(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted Lloyd optimisation with weighted K-means++ initialisation."""

        values64 = np.asarray(values, dtype=np.float64)
        weights64 = np.asarray(weights, dtype=np.float64).clip(min=0.0)
        base_probabilities = weights64 / max(weights64.sum(), self.epsilon)
        centroids = [values64[rng.choice(len(values64), p=base_probabilities)]]
        nearest = np.sum((values64 - centroids[0]) ** 2, axis=1)
        for _ in range(1, self.k):
            probabilities = weights64 * nearest
            if probabilities.sum() <= self.epsilon:
                index = int(rng.choice(len(values64), p=base_probabilities))
            else:
                index = int(rng.choice(len(values64), p=probabilities / probabilities.sum()))
            centroids.append(values64[index])
            nearest = np.minimum(nearest, np.sum((values64 - values64[index]) ** 2, axis=1))
        centres = np.stack(centroids, axis=0)
        previous_objective: Optional[float] = None
        assignments = np.zeros(len(values64), dtype=np.int64)
        for _ in range(self.max_iter):
            squared = ((values64[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            assignments = squared.argmin(axis=1)
            objective = float((weights64 * squared[np.arange(len(values64)), assignments]).sum())
            updated = centres.copy()
            for index in range(self.k):
                selected = assignments == index
                cluster_weight = weights64[selected].sum()
                if cluster_weight > self.epsilon:
                    updated[index] = (
                        values64[selected] * weights64[selected, None]
                    ).sum(axis=0) / cluster_weight
                else:
                    farthest = int(np.argmax(weights64 * squared.min(axis=1)))
                    updated[index] = values64[farthest]
            centres = updated
            if previous_objective is not None:
                relative_decrease = (previous_objective - objective) / max(
                    abs(previous_objective), self.epsilon
                )
                if 0.0 <= relative_decrease < self.tolerance:
                    break
            previous_objective = objective
        squared = ((values64[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        assignments = squared.argmin(axis=1)
        return centres, assignments
