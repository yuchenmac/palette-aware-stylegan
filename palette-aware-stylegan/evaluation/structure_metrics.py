"""Closure and line-continuity metrics from Eqs. (29)-(30)."""

from __future__ import annotations

import heapq
import math
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import convolve, label
from skimage.morphology import skeletonize
from torch import Tensor

from priors.structural_prior import StructuralPrior


ArrayLike = Union[np.ndarray, Tensor]
NEIGHBOURS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)


def edge_map_from_image(image: ArrayLike, structural_prior: StructuralPrior) -> np.ndarray:
    """Apply Eq. (1), threshold, and skeleton-ready binarisation to an image."""

    if isinstance(image, Tensor):
        tensor = image.detach().float().cpu()
    else:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = np.moveaxis(array, -1, 0)
        tensor = torch.from_numpy(array)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected a 3 x H x W or H x W x 3 image.")
    if float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0:
        tensor = tensor.mul(2.0).sub(1.0)
    with torch.no_grad():
        output = structural_prior(tensor.unsqueeze(0))
    edge = output.line_art > structural_prior.line_threshold
    return edge[0, 0].cpu().numpy().astype(bool)


def _skeleton_components(edge_map: ArrayLike) -> Tuple[np.ndarray, int]:
    array = edge_map.detach().cpu().numpy() if isinstance(edge_map, Tensor) else np.asarray(edge_map)
    skeleton = skeletonize(array.astype(bool))
    return label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))


def _endpoint_count(component: np.ndarray) -> int:
    neighbour_count = convolve(
        component.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    ) - component.astype(np.uint8)
    return int(np.count_nonzero(component & (neighbour_count == 1)))


def closure_rate(edge_map: ArrayLike) -> float:
    """Compute the closed-contour fraction in Eq. (29).

    A skeleton component with zero endpoints is counted as a closed contour;
    isolated one-pixel components are excluded.
    """

    labelled, count = _skeleton_components(edge_map)
    valid = 0
    closed = 0
    for index in range(1, count + 1):
        component = labelled == index
        if int(component.sum()) < 2:
            continue
        valid += 1
        if _endpoint_count(component) == 0:
            closed += 1
    return float(closed / valid) if valid else 0.0


def _farthest_distance(
    start: Tuple[int, int],
    coordinates: set[Tuple[int, int]],
) -> Tuple[Tuple[int, int], float]:
    distances = {start: 0.0}
    queue: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    farthest = start
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if distance > distances[farthest]:
            farthest = node
        for dy, dx in NEIGHBOURS:
            neighbour = (node[0] + dy, node[1] + dx)
            if neighbour not in coordinates:
                continue
            candidate = distance + (math.sqrt(2.0) if dy and dx else 1.0)
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return farthest, distances[farthest]


def _main_branch_length(component: np.ndarray) -> float:
    coordinates = {tuple(item) for item in np.argwhere(component)}
    if len(coordinates) <= 1:
        return float(len(coordinates))
    first = next(iter(coordinates))
    endpoint, _ = _farthest_distance(first, coordinates)
    _, diameter = _farthest_distance(endpoint, coordinates)
    return diameter + 1.0


def line_continuity_index(edge_map: ArrayLike) -> float:
    """Compute the unnormalised continuity index xi_len in Eq. (30).

    Main-branch length uses a two-sweep estimate of the 8-neighbour geodesic
    diameter, plus one pixel. It is exact for tree components; for cyclic
    components the two sweeps need not recover the exact diameter.
    """

    labelled, count = _skeleton_components(edge_map)
    lengths: List[float] = []
    endpoints: List[float] = []
    for index in range(1, count + 1):
        component = labelled == index
        if not component.any():
            continue
        lengths.append(_main_branch_length(component))
        endpoints.append(float(_endpoint_count(component)))
    if not lengths:
        return 0.0
    return float(np.mean(lengths) / (1.0 + np.mean(endpoints)))


def normalised_line_continuity(
    generated_edge_maps: Sequence[ArrayLike],
    reference_edge_maps: Sequence[ArrayLike],
    epsilon: float = 1.0e-8,
) -> float:
    """Normalise generated xi_len by the ground-truth mean as specified after Eq. (30)."""

    generated = [line_continuity_index(item) for item in generated_edge_maps]
    reference = [line_continuity_index(item) for item in reference_edge_maps]
    if not generated or not reference:
        raise ValueError("Generated and reference edge-map sets must be non-empty.")
    return float(np.mean(generated) / max(float(np.mean(reference)), epsilon))
