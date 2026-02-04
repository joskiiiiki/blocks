from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import numpy as np
import pygame

from src import assets

BlockData: TypeAlias = np.uint32  # Changed from uint64 to match actual usage
BLOCK_ID_MASK = 0xFF


class Block(Enum):
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WATER = 4
    UNKNOWN = 255

    def is_collidable(self) -> bool:
        return self.value not in NONCOLLIDABLE_BLOCKS

    def __str__(self) -> str:
        return self.name

    def get_texture(self) -> pygame.Surface | None:
        return BLOCK_TEXTURES[self.value]

    def get_item(self) -> Item | None:
        id = blocks_to_items.get(self.value)
        if id:
            return Item(id)
        return None


BLOCK_TEXTURES = {
    Block.AIR.value: None,
    Block.STONE.value: assets.STONE_BLOCK,
    Block.DIRT.value: assets.DIRT_BLOCK,
    Block.GRASS.value: assets.GRASS_BLOCK,
    Block.WATER.value: assets.WATER_BLOCK,
    Block.UNKNOWN.value: assets.UNKNOWN_BLOCK,
}


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
    return Block(block_id) if block_id in Block else Block.UNKNOWN


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


class Item(Enum):
    STONE = 1
    DIRT = 2
    GRASS = 3
    WATER = 4
    UNKNOWN = 255

    def get_block_id(self) -> int | None:
        return ITEM_TO_BLOCK.get(self.value)

    def get_block(self) -> Block | None:
        block_id = self.get_block_id()
        return Block(block_id) if block_id is not None else None

    def get_texture(self) -> pygame.Surface | None:
        return ITEM_TEXTURES.get(self.value)


ITEM_TO_BLOCK: dict[int, int] = {
    Item.STONE.value: Block.STONE.value,
    Item.DIRT.value: Block.DIRT.value,
    Item.GRASS.value: Block.GRASS.value,
    Item.WATER.value: Block.WATER.value,
    Item.UNKNOWN.value: Block.UNKNOWN.value,
}

blocks_to_items = {block: item for item, block in ITEM_TO_BLOCK.items()}

ITEM_TEXTURES = {
    Item.STONE.value: assets.STONE_BLOCK,
    Item.DIRT.value: assets.DIRT_BLOCK,
    Item.GRASS.value: assets.GRASS_BLOCK,
    Item.WATER.value: assets.WATER_BLOCK,
    Item.UNKNOWN.value: assets.UNKNOWN_BLOCK,
}
