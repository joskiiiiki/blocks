import math
from collections.abc import Callable
from enum import Enum

from pygame import Vector2

BLOCK_SIZE = 1


class Axis(Enum):
    X = 0
    Y = 1


class Normal(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    TOP = (0, 1)
    BOTTOM = (0, -1)


class BoundingBox:
    position: Vector2
    size: Vector2

    def __init__(self, position: Vector2, size: Vector2):
        """
        Parameters
        ----------
        position : Vector2
            The position of the bottom-left corner of the bounding box in world coordinates.
        size : Vector2
            The size of the bounding box in world coordinates.
        """
        self.position = position
        self.size = size

    @staticmethod
    def new_from_tuples(
        position: tuple[float, float], size: tuple[float, float]
    ) -> "BoundingBox":
        return BoundingBox(Vector2(*position), Vector2(*size))

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y

    @property
    def w(self) -> float:
        return self.size.x

    @property
    def h(self) -> float:
        return self.size.y

    @property
    def left(self) -> float:
        return self.position.x

    @property
    def right(self) -> float:
        return self.position.x + self.size.x

    @property
    def top(self) -> float:
        return self.position.y + self.size.y

    @property
    def bottom(self) -> float:
        return self.position.y

    @property
    def center(self) -> Vector2:
        return self.position + self.size / 2


def index_range(bounding_box: BoundingBox):
    """
    Parameters
    ----------
    bounding_box : BoundingBox
        The bounding box to index.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        ((x_min, y_min), (x_max, y_max)) - the inclusive range of tile indices
    """
    x_min = int(math.floor(bounding_box.position.x))
    y_min = int(math.floor(bounding_box.position.y))
    x_max = int(math.floor(bounding_box.position.x + bounding_box.size.x - 1e-9))
    y_max = int(math.floor(bounding_box.position.y + bounding_box.size.y - 1e-9))

    return (x_min, y_min), (x_max, y_max)


def aabb_collision_block(
    bounding_box: BoundingBox, block_coords: tuple[int, int]
) -> bool:
    """
    Parameters
    ----------
    bounding_box1 : BoundingBox
        The first bounding box.
    block_coords : tuple[int, int]
        The coordinates of the block.

    Returns
    -------
    bool
        True if the bounding boxes are colliding, False otherwise.
    """

    block_x, block_y = block_coords

    return (
        bounding_box.left < block_x + BLOCK_SIZE
        and bounding_box.right > block_x
        and bounding_box.bottom < block_y + BLOCK_SIZE
        and bounding_box.top > block_y
    )


class CollisionResult:
    axis: Axis
    penetration: float
    normal: Normal

    def __init__(self, axis: Axis, penetration: float, normal: Normal):
        self.axis = axis
        self.penetration = penetration
        self.normal = normal


def resolve_collision_axis(
    bounding_box: BoundingBox, velocity: Vector2, block_coord: tuple[int, int]
) -> CollisionResult | None:
    """
    Resolve collision along a single axis.

    Parameters
    ----------
    bounding_box : BoundingBox
        The bounding box of the object.
    velocity : Vector2
        The velocity of the object.
    block_coord : tuple[int, int]
        The coordinates of the block.

    Returns
    -------
    CollisionResult | None
        The collision result if there is a collision, None otherwise.
    """
    if not aabb_collision_block(bounding_box, block_coord):
        return None

    block_x, block_y = block_coord

    penetration_left = bounding_box.right - block_x
    penetration_right = (block_x + BLOCK_SIZE) - bounding_box.left
    penetration_bottom = bounding_box.top - block_y
    penetration_top = (block_y + BLOCK_SIZE) - bounding_box.bottom

    min_x = min(math.fabs(penetration_left), math.fabs(penetration_right))
    min_y = min(math.fabs(penetration_top), math.fabs(penetration_bottom))

    axis: Axis | None = None
    penetration: float | None = None
    normal: Normal | None = None

    if abs(velocity.x) > abs(velocity.y):
        if min_x < min_y:
            axis = Axis.X
            if velocity.x > 0:
                penetration = penetration_left
                normal = Normal.LEFT
            else:
                penetration = penetration_right
                normal = Normal.RIGHT
        else:
            axis = Axis.Y
            if velocity.y > 0:
                penetration = penetration_bottom
                normal = Normal.BOTTOM
            else:
                penetration = penetration_top
                normal = Normal.TOP
    else:
        if min_y < min_x:
            axis = Axis.Y
            if velocity.y > 0:
                penetration = penetration_bottom
                normal = Normal.BOTTOM
            else:
                penetration = penetration_top
                normal = Normal.TOP
        else:
            axis = Axis.X
            if velocity.x > 0:
                penetration = penetration_left
                normal = Normal.LEFT
            else:
                penetration = penetration_right
                normal = Normal.RIGHT

    return CollisionResult(axis, penetration, normal)


def sweep_collision(
    bounding_box: BoundingBox,
    velocity: Vector2,
    is_solid: Callable[[float, float], bool],
) -> tuple[Vector2, Vector2, bool, bool]:
    on_ground = False
    hit_ceiling = False

    vel = velocity.copy()

    # Start from current position
    current_pos = bounding_box.position.copy()

    # Test X movement
    if vel.x != 0:
        # Create test box at the NEW position
        test_pos = current_pos + Vector2(vel.x, 0)
        test_box = BoundingBox(test_pos, bounding_box.size)

        (x_min, y_min), (x_max, y_max) = index_range(test_box)
        x_collision = False

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if is_solid(x, y) and aabb_collision_block(test_box, (x, y)):
                    x_collision = True
                    break
            if x_collision:
                break

        if x_collision:
            # Stop X movement - don't update current_pos.x
            vel.x = 0
        else:
            # No collision - update position
            current_pos.x += vel.x

    # Test Y movement from the X-resolved position
    if vel.y != 0:
        # Create test box at the NEW position (includes x movement if it happened)
        test_pos = current_pos + Vector2(0, vel.y)
        test_box = BoundingBox(test_pos, bounding_box.size)

        (x_min, y_min), (x_max, y_max) = index_range(test_box)
        y_collision = False

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if is_solid(x, y) and aabb_collision_block(test_box, (x, y)):
                    y_collision = True
                    if vel.y < 0:
                        on_ground = True
                    elif vel.y > 0:
                        hit_ceiling = True
                    break
            if y_collision:
                break

        if y_collision:
            if vel.y < 0:  # Falling - snap to top of block below
                # Player's bottom should align with top of highest colliding block
                target_y = math.floor(current_pos.y)  # Snap to grid
                current_pos.y = target_y
                on_ground = True
            else:  # Jumping - snap to bottom of block above
                target_y = math.ceil(current_pos.y + bounding_box.size.y)
                current_pos.y = target_y - bounding_box.size.y
                hit_ceiling = True
            vel.y = 0
        else:
            # No collision - update position
            current_pos.y += vel.y

    return current_pos, vel, on_ground, hit_ceiling
