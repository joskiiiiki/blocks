from enum import Enum
from typing import TypeAlias

import numpy as np

BlockData: TypeAlias = np.uint32  # Changed from uint64 to match actual usage
BLOCK_ID_MASK = 0xFF


class Block(Enum):
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WATER = 4
    Unknown = 255


NONCOLLIDABLE_BLOCKS = {Block.AIR.value, Block.WATER.value}


def get_block_id_checked(block_data: BlockData) -> Block:
    """
    Retrieves the block id from the block data - The first 8 bits of the block data

    Parameters
    ----------
    block_data : int
        integer representing the block data

    Returns
    -------
    block_id : Block
        enum representing the block id
    """
    block_id = block_data & BLOCK_ID_MASK
    return Block(block_id) if block_id in Block else Block.Unknown


def is_collidable(block_data: BlockData) -> bool:
    """
    Checks if the block is collidable

    Parameters
    ----------
    block_data : int
        integer representing the block data

    Returns
    -------
    is_collidable : bool
        True if the block is collidable, False otherwise
    """
    return block_data & BLOCK_ID_MASK not in NONCOLLIDABLE_BLOCKS
