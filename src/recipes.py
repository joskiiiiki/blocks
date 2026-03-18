from collections.abc import Callable
from dataclasses import dataclass

from src.blocks import Item
from src.inventory import Stack

type MatchFunc = Callable[[list[Stack | None]], bool]

type Grid = list[Stack | None]


def _single_item_matcher(grid: Grid, stack: Stack) -> bool:
    prev: Stack | None = None
    for s in grid:
        if s is None:
            continue
        elif prev is None:
            prev = s
        # neither prev nor s is None so there are two stacks in the grid
        else:
            return False

    if prev is None:
        return False

    item_matches = prev[0] == stack[0]
    quantity_matches = prev[1] >= stack[1]
    return item_matches and quantity_matches


def build_single_matcher(stack: Stack) -> MatchFunc:
    return lambda grid: _single_item_matcher(grid, stack)


def strip_grid(grid: Grid) -> tuple[Grid, int, int]:
    rows = [grid[i * 3 : (i + 1) * 3] for i in range(3)]

    # find bounds directly instead of repeatedly slicing
    row_min, row_max = 3, 0
    col_min, col_max = 3, 0

    for r in range(3):
        for c in range(3):
            if rows[r][c] is not None:
                row_min = min(row_min, r)
                row_max = max(row_max, r)
                col_min = min(col_min, c)
                col_max = max(col_max, c)

    if row_max < row_min:  # empty grid
        return [], 0, 0

    width = col_max - col_min + 1
    height = row_max - row_min + 1
    truncated = [
        rows[r][c]
        for r in range(row_min, row_max + 1)
        for c in range(col_min, col_max + 1)
    ]
    return truncated, width, height


def _grid_matcher(grid: Grid, pattern: Grid, width: int, height: int) -> bool:
    stripped, sw, sh = strip_grid(grid)
    if sw != width or sh != height:
        return False
    stripped_pattern, _, _ = strip_grid(pattern)  # strip the pattern too
    for slot, expected in zip(stripped, stripped_pattern):
        if expected is None:
            if slot is not None:
                return False
        else:
            if slot is None or slot[0] != expected[0]:
                return False
    return True


def build_shaped_matcher(pattern: Grid, width: int, height: int) -> MatchFunc:
    return lambda grid: _grid_matcher(grid, pattern, width, height)


@dataclass
class Recipe:
    matches: MatchFunc
    output: Stack

    @classmethod
    def from_pattern(cls, pattern: Grid, output: Stack) -> "Recipe":
        _, width, height = strip_grid(pattern)
        return cls(build_shaped_matcher(pattern, width, height), output)

    @classmethod
    def from_single_item(cls, stack: Stack, output: Stack) -> "Recipe":
        return cls(build_single_matcher(stack), output)


def _craft(recipes: list[Recipe], grid: Grid) -> Stack | None:
    for recipe in recipes:
        if recipe.matches(grid):
            return recipe.output
    return None


def craft(grid: Grid) -> Stack | None:
    return _craft(RECIPES, grid)


RECIPES: list[Recipe] = [
    # --- basic materials ---
    Recipe.from_single_item((Item.LOG, 1), (Item.PLANKS, 4)),
    Recipe.from_pattern(
        [None, None, None, (Item.PLANKS, 1), None, None, (Item.PLANKS, 1), None, None],
        (Item.STICK, 4),
    ),  # --- torches ---
    Recipe.from_pattern(
        [None, None, None, (Item.COAL, 1), None, None, (Item.STICK, 1), None, None],
        (Item.TORCH, 4),
    ),
    Recipe.from_pattern(
        [None, None, None, (Item.AZURITE, 1), None, None, (Item.STICK, 1), None, None],
        (Item.COPPER_TORCH, 4),
    ),
    # --- pickaxe ---
    Recipe.from_pattern(
        [
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            None,
            (Item.STICK, 1),
            None,
            None,
            (Item.STICK, 1),
            None,
        ],
        (Item.PICKAXE, 1),
    ),
    # --- axe ---
    Recipe.from_pattern(
        [
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            None,
            (Item.IRON_INGOT, 1),
            (Item.STICK, 1),
            None,
            None,
            (Item.STICK, 1),
            None,
        ],
        (Item.AXE, 1),
    ),
    # --- sword ---
    Recipe.from_pattern(
        [
            None,
            None,
            None,
            (Item.IRON_INGOT, 1),
            None,
            None,
            (Item.STICK, 1),
            None,
            None,
        ],
        (Item.IRON_SWORD, 1),
    ),
    # --- bow ---
    Recipe.from_pattern(
        [
            None,
            (Item.IRON_INGOT, 1),
            None,
            (Item.STICK, 1),
            None,
            (Item.STICK, 1),
            None,
            (Item.IRON_INGOT, 1),
            None,
        ],
        (Item.BOW, 1),
    ),
    # --- arrow ---
    Recipe.from_pattern(
        [
            None,
            (Item.IRON_INGOT, 1),
            None,
            None,
            (Item.STICK, 1),
            None,
            None,
            None,
            None,
        ],
        (Item.ARROW, 4),
    ),
    # --- armor ---
    Recipe.from_pattern(
        [
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        (Item.HELMET, 1),
    ),
    Recipe.from_pattern(
        [
            (Item.IRON_INGOT, 1),
            None,
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
        ],
        (Item.CHESTPLATE, 1),
    ),
    Recipe.from_pattern(
        [
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            None,
            (Item.IRON_INGOT, 1),
            None,
            None,
            None,
        ],
        (Item.PANTS, 1),
    ),
    Recipe.from_pattern(
        [
            None,
            None,
            None,
            (Item.IRON_INGOT, 1),
            None,
            (Item.IRON_INGOT, 1),
            (Item.IRON_INGOT, 1),
            None,
            (Item.IRON_INGOT, 1),
        ],
        (Item.SHOES, 1),
    ),
]
