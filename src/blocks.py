from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

import numpy as np

from src import assets

BlockData: TypeAlias = np.uint32
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
    # ores
    COAL_ORE = 10
    IRON_ORE = 11
    AZURITE_ORE = 12
    EMERALD_ORE = 13
    DIAMOND_ORE = 14
    IRONQUARTZ_ORE = 15
    SHADOW_ORE = 16
    VOID_ORE = 17

    UNKNOWN = 255

    def is_collidable(self) -> bool:
        return self.value not in NONCOLLIDABLE_BLOCKS

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def get_texture_from_id(block_data: int) -> assets.Texture | None:
        name = BLOCK_TO_TEXTURE.get(block_data)
        if not name:
            return None
        return assets.get_texture(name)

    def get_texture_name(self) -> str | None:
        return BLOCK_TO_TEXTURE.get(self.value)

    @staticmethod
    def get_tex_name_from_data(data: int) -> str | None:
        return BLOCK_TO_TEXTURE.get(data)

    def with_data(self, *data: tuple[int, int]) -> int:
        d = [d[1] << d[0] for d in data]
        return self.value | sum(d) << 8

    def get_texture(self) -> assets.Texture | None:
        name = BLOCK_TO_TEXTURE.get(self.value)
        if not name:
            return None
        return assets.get_texture(name)

    def get_item(self) -> Item | None:
        id = blocks_to_items.get(self.value)
        if id:
            return Item(id)
        return None

    def get_dropped_item(self) -> Item | None:
        id = BLOCK_TO_DROP.get(self.value)
        if id:
            return Item(id)

        return self.get_item()

    def get_break_time_with(self, tool: Item | None = None) -> float:
        return break_time(self, tool)

    def get_break_time_default(self) -> float:
        return break_time_default(self)

    @property
    def id(self) -> int:
        return self.value


BLOCK_TO_TEXTURE: dict[int, str | None] = {
    Block.AIR.value: None,
    Block.STONE.value: "stone",
    Block.DIRT.value: "dirt",
    Block.GRASS.value: "grass",
    Block.WATER.value: "water",
    Block.WATER.with_data((0, 1)): "water_top",
    Block.LOG.value: "log",
    Block.LEAVES.value: "leaves",
    Block.TORCH.value: "torch",
    Block.COPPER_TORCH.value: "copper_torch",
    Block.PLANKS.value: "planks",
    Block.COAL_ORE.value: "coal_ore",
    Block.IRON_ORE.value: "iron_ore",
    Block.AZURITE_ORE.value: "azurite_ore",
    Block.EMERALD_ORE.value: "emerald_ore",
    Block.DIAMOND_ORE.value: "diamond_ore",
    Block.IRONQUARTZ_ORE.value: "ironquartz_ore",
    Block.SHADOW_ORE.value: "shadow_ore",
    Block.VOID_ORE.value: "void_ore",
    Block.UNKNOWN.value: "unknown",
}

NONCOLLIDABLE_BLOCKS = {
    Block.AIR.value,
    Block.WATER.value,
    Block.TORCH.value,
    Block.COPPER_TORCH.value,
}


def get_block_id_checked(block_data: BlockData) -> Block:
    block_id = block_data & BLOCK_ID_MASK
    return Block(block_id) if block_id in Block else Block.UNKNOWN


def is_solid(block_data: BlockData) -> bool:
    return block_data & BLOCK_ID_MASK not in NONCOLLIDABLE_BLOCKS


class Item(Enum):
    # placeable blocks
    STONE = 1
    DIRT = 2
    GRASS = 3
    WATER = 4
    LOG = 5
    LEAVES = 6
    PLANKS = 7
    TORCH = 8
    COPPER_TORCH = 9
    # ores (not placeable)
    COAL_ORE = 10
    IRON_ORE = 11
    AZURITE_ORE = 12
    EMERALD_ORE = 13
    DIAMOND_ORE = 14
    IRONQUARTZ_ORE = 15
    SHADOW_ORE = 16
    VOID_ORE = 17
    # drops / materials
    COAL = 20
    IRON_INGOT = 21
    AZURITE = 22
    EMERALD = 23
    DIAMOND = 24
    IRONQUARTZ_INGOT = 25
    SHADOW_GEM = 26
    VOID_SHARD = 27
    STICK = 28
    # tools & weapons
    IRON_SWORD = 30
    SWORD = 31
    AXE = 32
    BOW = 33
    ARROW = 34
    PICKAXE = 35
    # armor
    HELMET = 40
    CHESTPLATE = 41
    PANTS = 42
    SHOES = 43

    UNKNOWN = 255

    def get_block_id(self) -> int | None:
        return ITEM_TO_BLOCK.get(self.value)

    def get_block(self) -> Block | None:
        block_id = self.get_block_id()
        return Block(block_id) if block_id is not None else None

    @staticmethod
    def get_texture_from_id(id: int) -> assets.Texture | None:
        name = ITEM_TEXTURES.get(id)
        if not name:
            return None
        return assets.get_texture(name)

    def get_texture(self) -> assets.Texture | None:
        name = ITEM_TEXTURES.get(self.value)
        if not name:
            return None
        return assets.get_texture(name)

    def get_attack_speed(self) -> float:
        return attack_speed_of_item(self)

    def get_damage_multiplier(self) -> float:
        return ITEM_DAMAGE_MULTIPLIER.get(self.value, 1.0)


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
    Item.COAL_ORE.value: Block.COAL_ORE.value,
    Item.IRON_ORE.value: Block.IRON_ORE.value,
    Item.AZURITE_ORE.value: Block.AZURITE_ORE.value,
    Item.EMERALD_ORE.value: Block.EMERALD_ORE.value,
    Item.DIAMOND_ORE.value: Block.DIAMOND_ORE.value,
    Item.IRONQUARTZ_ORE.value: Block.IRONQUARTZ_ORE.value,
    Item.SHADOW_ORE.value: Block.SHADOW_ORE.value,
    Item.VOID_ORE.value: Block.VOID_ORE.value,
    Item.UNKNOWN.value: Block.UNKNOWN.value,
}
# ore blocks drop their respective gem/ingot item, not the ore block itself
BLOCK_TO_DROP: dict[int, int] = {
    Block.COAL_ORE.value: Item.COAL.value,
    Block.IRON_ORE.value: Item.IRON_INGOT.value,
    Block.AZURITE_ORE.value: Item.AZURITE.value,
    Block.EMERALD_ORE.value: Item.EMERALD.value,
    Block.DIAMOND_ORE.value: Item.DIAMOND.value,
    Block.IRONQUARTZ_ORE.value: Item.IRONQUARTZ_INGOT.value,
    Block.SHADOW_ORE.value: Item.SHADOW_GEM.value,
    Block.VOID_ORE.value: Item.VOID_SHARD.value,
}

blocks_to_items = {block: item for item, block in ITEM_TO_BLOCK.items()}

ITEM_TEXTURES: dict[int, str] = {
    Item.STONE.value: "stone",
    Item.DIRT.value: "dirt",
    Item.GRASS.value: "grass",
    Item.WATER.value: "water",
    Item.LOG.value: "log",
    Item.LEAVES.value: "leaves",
    Item.TORCH.value: "torch",
    Item.PLANKS.value: "planks",
    Item.COPPER_TORCH.value: "copper_torch",
    Item.COAL_ORE.value: "coal_ore",
    Item.IRON_ORE.value: "iron_ore",
    Item.AZURITE_ORE.value: "azurite_ore",
    Item.EMERALD_ORE.value: "emerald_ore",
    Item.DIAMOND_ORE.value: "diamond_ore",
    Item.IRONQUARTZ_ORE.value: "ironquartz_ore",
    Item.SHADOW_ORE.value: "shadow_ore",
    Item.VOID_ORE.value: "void_ore",
    Item.COAL.value: "coal",
    Item.IRON_INGOT.value: "iron_ingot",
    Item.AZURITE.value: "azurite",
    Item.EMERALD.value: "emerald",
    Item.DIAMOND.value: "diamond",
    Item.IRONQUARTZ_INGOT.value: "ironquartz_ingot",
    Item.SHADOW_GEM.value: "shadow_gem",
    Item.VOID_SHARD.value: "void_shard",
    Item.STICK.value: "stick",
    Item.IRON_SWORD.value: "sword",
    Item.SWORD.value: "sword",
    Item.AXE.value: "axe",
    Item.BOW.value: "bow",
    Item.ARROW.value: "arrow",
    Item.PICKAXE.value: "pickaxe_icon",
    Item.HELMET.value: "helmet",
    Item.CHESTPLATE.value: "chestplate",
    Item.PANTS.value: "pants",
    Item.SHOES.value: "shoes",
    Item.UNKNOWN.value: "unknown",
}


def damage_of_item(item: Item) -> float:
    return ITEM_DAMAGE_MULTIPLIER.get(item.value, 1.0)


ITEM_DAMAGE_MULTIPLIER: dict[int, float] = {
    Item.IRON_SWORD.value: 7.0,
    Item.SWORD.value: 7.0,
    Item.AXE.value: 9.0,
    Item.BOW.value: 4.0,
    Item.ARROW.value: 4.0,
    Item.PICKAXE.value: 3.0,
}


BLOCK_SPEED = {
    Block.WATER.value: 0.8,
}

# time in seconds to break each block without tool
BLOCK_BREAK_TIME: dict[int, float] = {
    Block.STONE.value: 2.0,
    Block.DIRT.value: 0.5,
    Block.GRASS.value: 0.5,
    Block.LOG.value: 1.5,
    Block.LEAVES.value: 0.3,
    Block.PLANKS.value: 1.0,
    Block.TORCH.value: 0.1,
    Block.COPPER_TORCH.value: 0.1,
    Block.COAL_ORE.value: 3.0,
    Block.IRON_ORE.value: 3.5,
    Block.AZURITE_ORE.value: 4.0,
    Block.EMERALD_ORE.value: 4.0,
    Block.DIAMOND_ORE.value: 5.0,
    Block.IRONQUARTZ_ORE.value: 4.5,
    Block.SHADOW_ORE.value: 5.0,
    Block.VOID_ORE.value: 6.0,
}

DEFAULT_BREAK_TIME = 1.0  # fallback for unlisted blocks


def break_time(block: Block, tool: Item | None = None) -> float:
    default = break_time_default(block)
    multiplier = 1.0 if tool is None else break_multiplier(tool, block)
    return default * multiplier


def break_time_default(block: Block) -> float:
    return BLOCK_BREAK_TIME.get(block.value, DEFAULT_BREAK_TIME)


type BreakBonus = Callable[[Block], float | None]


def _pickaxe_bonus(block: Block) -> float | None:
    return (
        0.3
        if block
        in {
            Block.STONE,
            Block.COAL_ORE,
            Block.IRON_ORE,
            Block.AZURITE_ORE,
            Block.EMERALD_ORE,
            Block.DIAMOND_ORE,
            Block.IRONQUARTZ_ORE,
            Block.SHADOW_ORE,
            Block.VOID_ORE,
        }
        else None
    )


def _axe_bonus(block: Block) -> float | None:
    return 0.4 if block in {Block.LOG, Block.PLANKS} else None


ITEM_BREAK_MULTIPLIER: dict[int, BreakBonus] = {
    Item.PICKAXE.value: _pickaxe_bonus,
    Item.AXE.value: _axe_bonus,
    Item.SWORD.value: lambda _: 0.6,
    Item.IRON_SWORD.value: lambda _: 0.6,
}


def break_multiplier(item: Item, block: Block) -> float:
    bonus = ITEM_BREAK_MULTIPLIER.get(item.value)
    if bonus is None:
        return 1.0
    return bonus(block) or 1.0


ITEM_ATTACK_SPEED: dict[int, float] = {
    Item.IRON_SWORD.value: 1.0,
    Item.SWORD.value: 1.0,
    Item.AXE.value: 0.5,  # slower, hits harder
    Item.PICKAXE.value: 0.7,
    Item.BOW.value: 0.6,
}


def attack_speed_of_item(item: Item) -> float:
    return ITEM_ATTACK_SPEED.get(item.value, 1.0)
