# --- physics helper (lives in your game loop or a physics module) ---
from src.blocks import BLOCK_ID_MASK, is_solid
from src.collision import sweep_collision
from src.entity.entity import Entity, PhysicsResult
from src.interfaces import IWorld


def physics_step(entity: Entity, world: IWorld, dt: float) -> None:
    position, _, on_ground, hit_ceiling, x_col, y_col = sweep_collision(
        bounding_box=entity.bounding_box,
        velocity=entity.velocity * dt,
        is_solid=world.is_solid,
    )

    entity.apply_physics_result(
        dt, PhysicsResult(position, on_ground, hit_ceiling, x_col, y_col)
    )

    if entity.auto_jump and x_col and on_ground and entity._auto_jump_cooldown <= 0:
        check_x = entity.bounding_box.right + 0.1 if entity.vel_x > 0 else entity.bounding_box.left - 0.1
        check_y = entity.bounding_box.bottom + 1.0
        block = world.get_block(check_x, check_y)
        if block and is_solid(block.value & BLOCK_ID_MASK):
            entity._auto_jump_cooldown = 0.3
            entity.jump()

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
