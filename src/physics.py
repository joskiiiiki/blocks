# --- physics helper (lives in your game loop or a physics module) ---
from src.blocks import BLOCK_ID_MASK, is_solid
from src.collision import sweep_collision
from src.entity import Entity, PhysicsResult
from src.interfaces import IWorld


def physics_step(entity: Entity, world: IWorld, dt: float) -> None:
    next_pos = (
        entity.position + entity.velocity.normalize()
        if entity.velocity.length() > 0
        else entity.position
    )

    position, _, on_ground, hit_ceiling, x_col, y_col = sweep_collision(
        bounding_box=entity.bounding_box,
        velocity=entity.velocity * dt,
        is_solid=world.is_solid,
    )

    # auto-jump over 1-block steps
    if entity.auto_jump and x_col and on_ground:
        block = world.get_block(next_pos.x, next_pos.y + 1)
        if block and is_solid(block.value & BLOCK_ID_MASK):
            entity.jump()

    entity.apply_physics_result(
        PhysicsResult(position, on_ground, hit_ceiling, x_col, y_col)
    )


def get_touching_blocks(entity: Entity, world: IWorld, inset: float = 0.1) -> set[int]:
    bb = entity.bounding_box
    touching: set[int] = set()
    for px, py in [
        (bb.left + inset, bb.bottom + inset),
        (bb.right - inset, bb.bottom + inset),
        (bb.left + inset, bb.top - inset),
        (bb.right - inset, bb.top - inset),
        (bb.center.x, bb.center.y),
    ]:
        block = world.get_block(px, py)
        if block is not None:
            touching.add(BLOCK_ID_MASK & block.value)
    return touching
