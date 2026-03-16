from dataclasses import dataclass


@dataclass
class EntityStats:
    maxhealth: int
    maxstagger: int
    walk_speed: float
    dmg: int
    attack_speed: float
    bbox_size: tuple[float, float]
    jump_power: float = 1
    gravity: float = -9.81
    armor_value: float = 0.0
    passivregen: float = 0.0
    sprint_speed: float = 0.0  # 0 = no sprinting
    attack_range: float = 1.0
    knockback: float = 0.0


PLAYER_STATS = EntityStats(
    maxhealth=200,
    maxstagger=100,
    walk_speed=5.0,
    dmg=10,
    attack_speed=1.0,
    bbox_size=(0.8, 1.8),
    jump_power=6.0,
    armor_value=0.0,
    passivregen=5.0,
    sprint_speed=8.0,
    attack_range=3.0,
    knockback=10,
)


ZOMBIE_STATS = EntityStats(
    maxhealth=100,
    maxstagger=60,
    walk_speed=2.5,
    dmg=8,
    attack_speed=1.5,
    bbox_size=(0.8, 1.8),
    jump_power=6.0,
    attack_range=2.0,
    knockback=5,
)
