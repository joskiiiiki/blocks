from pathlib import Path

import numpy as np
import numpy.typing as npt
from platformdirs import user_data_path

from src.blocks import Block
from src.world.chunk import Chunk


def world_path(name: str) -> Path:
    return user_data_path("blocks") / name


def smooth_mask_ce(
    mask: npt.NDArray[np.bool_],
    death_limit: int = 3,
    birth_limit: int = 4,
) -> bool:
    """
    Vectorized cellular automata smoothing.
    About 100-1000x faster than loop version.
    """
    # Pad the mask to handle edges (using constant padding with True)
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=True)

    # Count neighbors using slicing (4-connected neighborhood)
    neighbor_count = (
        padded[:-2, 1:-1].astype(int)  # left
        + padded[2:, 1:-1].astype(int)  # right
        + padded[1:-1, :-2].astype(int)  # bottom
        + padded[1:-1, 2:].astype(int)  # top
    )

    # Create new mask
    new_mask = mask.copy()

    # Apply rules
    new_mask[neighbor_count < death_limit] = False
    new_mask[neighbor_count > birth_limit] = True

    # Check if anything changed
    changed = not np.array_equal(mask, new_mask)

    # Update in place
    mask[:] = new_mask

    return changed


def smooth_caves_with_neighbors(
    chunk: Chunk,
    left_chunk: Chunk | None,
    right_chunk: Chunk | None,
    min_y: int = 20,
    max_y: int = 256,
    death_limit: int = 3,
    birth_limit: int = 4,
) -> bool:
    """
    Smooth caves considering neighboring chunks.
    Handles edge blocks correctly.
    """
    region_height = max_y - min_y + 1

    # Extract cave regions from all three chunks
    center = chunk.blocks[:, min_y : max_y + 1]

    # Get neighbor columns (1 column from each side)
    if left_chunk is not None:
        left_col = left_chunk.blocks[
            -1:, min_y : max_y + 1
        ]  # Last column of left chunk
    else:
        left_col = np.ones((1, region_height), dtype=center.dtype) * Block.STONE.value

    if right_chunk is not None:
        right_col = right_chunk.blocks[
            0:1, min_y : max_y + 1
        ]  # First column of right chunk
    else:
        right_col = np.ones((1, region_height), dtype=center.dtype) * Block.STONE.value

    # Combine into extended region: [left_col, center, right_col]
    extended = np.concatenate([left_col, center, right_col], axis=0)

    # Create solid mask
    solid_mask = extended != Block.AIR.value

    # Pad for edge handling (top/bottom)
    padded = np.pad(solid_mask, ((1, 1), (1, 1)), mode="constant", constant_values=True)

    # Count neighbors (4-connected)
    neighbor_count = (
        padded[:-2, 1:-1].astype(int)  # left
        + padded[2:, 1:-1].astype(int)  # right
        + padded[1:-1, :-2].astype(int)  # bottom
        + padded[1:-1, 2:].astype(int)  # top
        + padded[:-2, :-2].astype(int)  # bottom-left diagonal
        + padded[2:, :-2].astype(int)  # bottom-right diagonal
        + padded[:-2, 2:].astype(int)  # top-left diagonal
        + padded[2:, 2:].astype(int)
    )

    # Apply CA rules on extended region
    new_solid = solid_mask.copy()
    new_solid[neighbor_count < death_limit] = False
    new_solid[neighbor_count > birth_limit] = True

    # Extract only the center chunk (skip the borrowed columns)
    center_new = new_solid[1:-1, :]  # Skip first and last column

    # Check if changed
    changed = not np.array_equal(solid_mask[1:-1, :], center_new)

    # Update the chunk (convert back to block types)
    chunk.blocks[:, min_y : max_y + 1] = np.where(
        center_new,
        chunk.blocks[:, min_y : max_y + 1],  # Keep existing solid type
        Block.AIR.value,
    )

    return changed
