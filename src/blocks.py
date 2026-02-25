from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import numpy as np
import pygame

from src import assets

BlockData: TypeAlias = np.uint32  # Changed from uint64 to match actual usage
BLOCK_ID_MASK = 0b1111_1111
BLOCK_DATA_MASK = 2 * 32 - 1 - BLOCK_ID_MASK


class Block(Enum):
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WATER = 4
    LOG = 5
    LEAVES = 6
    PLANKS = 7
    TORCH = 8
    COPPER_TORCH = 9
    UNKNOWN = 255

    def is_collidable(self) -> bool:
        return self.value not in NONCOLLIDABLE_BLOCKS

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def get_texture_from_id(block_data: int) -> pygame.Surface | None:
        name = BLOCK_TO_TEXTURE[block_data]
        if not name:
            return None
        return assets.TEXTURES.get_texture(name)

    def get_texture_name(self) -> str | None:
        name = BLOCK_TO_TEXTURE.get(self.value)
        return name

    @staticmethod
    def get_tex_name_from_data(data: int) -> str | None:
        name = BLOCK_TO_TEXTURE.get(data)
        return name

    def with_data(self, *data: tuple[int, int]) -> int:
        d = [d[1] << d[0] for d in data]
        return self.value | sum(d) << 8

    def get_texture(self) -> pygame.Surface | None:
        name = BLOCK_TO_TEXTURE[self.value]
        if not name:
            return None
        print(assets.TEXTURES)
        return assets.TEXTURES.get_texture(name)

    def get_item(self) -> Item | None:
        id = blocks_to_items.get(self.value)
        if id:
            return Item(id)
        return None

    @property
    def id(self) -> int:
        return self.value


BLOCK_TO_TEXTURE: dict[int, str | None] = {
    Block.AIR.value: None,
    Block.STONE.value: "stone",
    Block.DIRT.value: "dirt",
    Block.GRASS.value: "grass",
    Block.WATER.value: "water",
    (Block.WATER.with_data((0, 1))): "water_top",
    Block.LOG.value: "log",
    Block.LEAVES.value: "leaves",
    Block.TORCH.value: "torch",
    Block.COPPER_TORCH.value: "copper_torch",
    Block.PLANKS.value: "planks",
    Block.UNKNOWN.value: "unknown",
}


NONCOLLIDABLE_BLOCKS = {Block.AIR.value, Block.WATER.value, Block.TORCH.value, Block.COPPER_TORCH.value}


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


def is_solid(block_data: BlockData) -> bool:
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
    LOG = 5
    LEAVES = 6
    PLANKS = 7
    TORCH = 8
    COPPER_TORCH = 9

    UNKNOWN = 255

    def get_block_id(self) -> int | None:
        return ITEM_TO_BLOCK.get(self.value)

    def get_block(self) -> Block | None:
        block_id = self.get_block_id()
        return Block(block_id) if block_id is not None else None

    @staticmethod
    def get_texture_from_id(id: int) -> pygame.Surface | None:
        name = ITEM_TEXTURES.get(id)
        if not name:
            return None
        return assets.TEXTURES.get_texture(name)

    def get_texture(self) -> pygame.Surface | None:
        name = ITEM_TEXTURES.get(self.value)
        if not name:
            return None
        return assets.TEXTURES.get_texture(name)


ITEM_TO_BLOCK: dict[int, int] = {
    Item.STONE.value: Block.STONE.value,
    Item.DIRT.value: Block.DIRT.value,
    Item.GRASS.value: Block.GRASS.value,
    Item.WATER.value: Block.WATER.value,
    Item.LOG.value: Block.LOG.value,
    Item.PLANKS.value: Block.PLANKS.value,
    Item.LEAVES.value: Block.LEAVES.value,
    Item.TORCH.value: Block.TORCH.value,
    Item.COPPER_TORCH.value: Block.COPPER_TORCH.value,
    Item.UNKNOWN.value: Block.UNKNOWN.value,
}

blocks_to_items = {block: item for item, block in ITEM_TO_BLOCK.items()}

ITEM_TEXTURES = {
    Item.STONE.value: "stone",
    Item.DIRT.value: "dirt",
    Item.GRASS.value: "grass",
    Item.WATER.value: "water",
    Item.LOG.value: "log",
    Item.LEAVES.value: "leaves",
    Item.TORCH.value: "torch",
    Item.PLANKS.value: "plank",
    Item.COPPER_TORCH.value: "copper_torch",
    Item.UNKNOWN.value: "unknown",
}

BLOCK_SPEED = {
    Block.WATER.value: 0.8,
}
