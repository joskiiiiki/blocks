from collections.abc import Iterable
from dataclasses import dataclass

from src.collision import aabb_overlap
from src.entity.mob import Mob
from src.player import Player


@dataclass
class CombatResult:
    player_damaged: bool


def process_combat(player: Player, mobs: Iterable[Mob], dt: float) -> CombatResult:
    player_damaged = False

    for mob in mobs:
        # if mob.attack() and aabb_overlap(mob.attack_bbox(), player.bounding_box):
        #     knockback = (player.position - mob.position).normalize() * mob.knockback
        #     player.take_damage(mob.damage, mob.damage * 0.5, knockback)
        #     player_damaged = True

        if player.attack() and aabb_overlap(player.attack_bbox(), mob.bounding_box):
            print("ATTACK")
            knockback = (mob.position - player.position).normalize() * player.knockback
            mob.take_damage(player.damage, player.damage * 0.5, knockback)

    return CombatResult(player_damaged)
