from __future__ import annotations

import pygame

from src.blocks import BLOCK_ID_MASK, BLOCK_SPEED, Block
from src.collision import BoundingBox, sweep_collision
from src.world import World


class Entity:
    # --- stats ---
    maxhealth: int = 0
    maxstagger: int = 0
    default_walk_speed: float = 0.0
    default_dmg: int = 0
    default_attack_speed: float = 0.0
    badEntity: bool = False
    armor_value: float = 0.0

    # --- bounding box size (override per subclass) ---
    bbox_size: pygame.Vector2 = pygame.Vector2(1.0, 1.0)

    # --- runtime stat state ---
    health: float
    stagger: float
    walkspeed: float
    attackspeed: float

    # --- physics ---
    gravity: float = -9.81
    bounding_box: BoundingBox
    velocity: pygame.Vector2
    on_ground: bool = False
    hit_ceiling: bool = False
    in_water: bool = False

    # --- combat state ---
    is_hit: bool = False
    is_parrying: bool = False
    is_blocking: bool = False
    is_attacking: bool = False
    is_staggered: bool = False
    is_dead: bool = False

    # --- timers ---
    attack_cooldown: float = 0.0
    hit_timer: float = 0.0
    stagger_timer: float = 0.0
    attack_frameindex: int = 0

    def __init__(
        self,
        maxhealth: int,
        maxstagger: int,
        walkspeed: float,
        dmg: int,
        attackspeed: float,
        x: float,
        y: float,
        world: World,
        badEntity: bool = False,
    ) -> None:
        self.maxhealth = maxhealth
        self.health = float(maxhealth)

        self.maxstagger = maxstagger
        self.stagger = float(maxstagger)

        self.default_walk_speed = walkspeed
        self.walkspeed = walkspeed

        self.Defaultdmg = dmg

        self.default_attack_speed = attackspeed
        self.attackspeed = attackspeed

        self.badEntity = badEntity
        self.armor_value = 0.0
        self.world = world

        # physics — bounding box owns position
        self.bounding_box = BoundingBox(
            position=pygame.Vector2(x, y),
            size=pygame.Vector2(self.bbox_size),  # copy so subclasses don't share
        )
        self.velocity = pygame.Vector2(0, 0)
        self.on_ground = False
        self.hit_ceiling = False
        self.in_water = False

        # combat state
        self.is_hit = False
        self.is_parrying = False
        self.is_blocking = False
        self.is_attacking = False
        self.is_staggered = False
        self.is_dead = False

        # timers
        self.attack_cooldown = 0.0
        self.hit_timer = 0.0
        self.stagger_timer = 0.0
        self.attack_frameindex = 0

    # --- position properties (delegate to bounding_box) ---

    @property
    def position(self) -> pygame.Vector2:
        return self.bounding_box.position

    @position.setter
    def position(self, value: pygame.Vector2) -> None:
        self.bounding_box.position = value

    @property
    def x(self) -> float:
        return self.bounding_box.position.x

    @x.setter
    def x(self, value: float) -> None:
        self.bounding_box.position.x = value

    @property
    def y(self) -> float:
        return self.bounding_box.position.y

    @y.setter
    def y(self, value: float) -> None:
        self.bounding_box.position.y = value

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @xy.setter
    def xy(self, value: tuple[float, float]) -> None:
        self.x, self.y = value

    # --- velocity properties ---

    @property
    def vel_x(self) -> float:
        return self.velocity.x

    @vel_x.setter
    def vel_x(self, value: float) -> None:
        self.velocity.x = value

    @property
    def vel_y(self) -> float:
        return self.velocity.y

    @vel_y.setter
    def vel_y(self, value: float) -> None:
        self.velocity.y = value

    @property
    def vel_xy(self) -> tuple[float, float]:
        return (self.velocity.x, self.velocity.y)

    @vel_xy.setter
    def vel_xy(self, value: tuple[float, float]) -> None:
        self.velocity.x, self.velocity.y = value

    # --- physics ---

    def apply_gravity(self, dt: float, multiplier: float = 1.0) -> None:
        self.vel_y += self.gravity * dt * multiplier

    def apply_velocity(self, dt: float) -> None:
        """Moves entity via sweep collision against the world."""
        self.position, _, self.on_ground, self.hit_ceiling = sweep_collision(
            bounding_box=self.bounding_box,
            velocity=self.velocity * dt,
            is_solid=self.world.is_solid,
        )

        if self.on_ground and self.vel_y < 0:
            self.vel_y = 0.0
        elif self.hit_ceiling and self.vel_y > 0:
            self.vel_y = 0.0

    # --- update loop ---

    def update_entity(self, dt: float) -> None:
        if self.is_dead:
            return

        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt

        if self.hit_timer > 0:
            self.hit_timer -= dt
        else:
            self.is_hit = False

        self._update_stagger(dt)
        self._regen_stagger(dt)

        touching_blocks = self.get_touching_blocks()
        self.in_water = Block.WATER.value in touching_blocks

        self.apply_gravity(dt)

        if self.in_water:
            self.velocity *= BLOCK_SPEED[Block.WATER.value]

        # sweep collision (defined on Entity.apply_velocity)
        self.apply_velocity(dt)

    def get_touching_blocks(self, inset: float = 0.1) -> set[int]:
        touching_blocks: set[int] = set()
        check_points: list[tuple[float, float]] = [
            (
                self.bounding_box.left + inset,
                self.bounding_box.bottom + inset,
            ),  # bottom-left
            (
                self.bounding_box.right - inset,
                self.bounding_box.bottom + inset,
            ),  # bottom-right
            (
                self.bounding_box.left + inset,
                self.bounding_box.top - inset,
            ),  # top-left
            (
                self.bounding_box.right - inset,
                self.bounding_box.top - inset,
            ),  # top-right
            (self.bounding_box.center.x, self.bounding_box.center.y),  # center
        ]
        for px, py in check_points:
            block = self.world.chunk_manager.get_block(px, py)
            if block is not None:
                touching_blocks.add(BLOCK_ID_MASK & block)
        return touching_blocks

    # --- damage ---

    def TakeDamage(self, damage: float, stagger_damage: float) -> None:
        if self.is_dead:
            return

        self.is_hit = True
        self.hit_timer = 150.0

        mitigated_damage = damage * (1.0 - self.armor_value / 100.0)
        self.health -= mitigated_damage
        self.stagger -= stagger_damage

        if self.health <= 0:
            self.die()

    # --- stagger ---

    def _update_stagger(self, dt: float) -> None:
        if self.stagger <= 0 and not self.is_staggered:
            self.is_staggered = True
            self.walkspeed = 0.0
            self.attackspeed = 0.0
            self.stagger_timer = 1000.0

        if self.is_staggered:
            self.stagger_timer -= dt

            if self.stagger_timer <= 0:
                self.is_staggered = False
                self.stagger = float(self.maxstagger)
                self.walkspeed = self.default_walk_speed
                self.attackspeed = self.default_attack_speed

    def _regen_stagger(self, dt: float) -> None:
        if not self.is_hit and not self.is_staggered:
            if self.stagger < self.maxstagger:
                self.stagger += 0.02 * dt
                if self.stagger > self.maxstagger:
                    self.stagger = float(self.maxstagger)

    # --- combat ---

    def attack(self) -> None:
        if self.attack_cooldown <= 0 and not self.is_staggered:
            self.is_attacking = True
            self.attack_frameindex = 0
            self.attack_cooldown = self.attackspeed

    # --- death ---

    def die(self) -> None:
        self.is_dead = True
        self.walkspeed = 0.0
        self.vel_x = 0.0
        self.is_attacking = False

    # --- passive healing (opt-in via subclass) ---

    def regeneration(self, dt: float) -> None:
        """Regenerates health passively. Requires subclass to define `passivregen`."""
        if not self.is_hit and self.health < self.maxhealth:
            self.health += self.passivregen * dt / 1000.0  # type: ignore[attr-defined]
            if self.health > self.maxhealth:
                self.health = float(self.maxhealth)


# --- Player ---


class Player(Entity):
    # --- stats ---
    maxhealth: int = 200
    maxstagger: int = 100
    default_walk_speed: float = 150.0
    Defaultdmg: int = 10
    default_attack_speed: float = 1.0

    # --- physics ---
    bbox_size: pygame.Vector2 = pygame.Vector2(0.8, 1.8)
    gravity: float = -9.81
    jump_power: float = 12.0
    sprint_speed: float = 8.0

    passivregen: float = 5.0
    armor_value: float = 0.0

    def __init__(self, x: float, y: float, world: World) -> None:
        super().__init__(
            maxhealth=self.maxhealth,
            maxstagger=self.maxstagger,
            walkspeed=self.default_walk_speed,
            dmg=self.Defaultdmg,
            attackspeed=self.default_attack_speed,
            x=x,
            y=y,
            world=world,
            badEntity=False,
        )
        self.passivregen = Player.passivregen
        self.armor_value = Player.armor_value
        self.jump_power = Player.jump_power
        self.sprint_speed = Player.sprint_speed

    # --- movement ---

    def jump(self) -> None:
        if self.on_ground:
            self.vel_y += self.jump_power
            self.on_ground = False

    def swim_up(self) -> None:
        if self.in_water:
            self.vel_y += 1.0

    def swim_down(self) -> None:
        if self.in_water:
            self.vel_y -= 1.0

    # --- combat ---

    def parry(self) -> None:
        self.is_parrying = True
        self.walkspeed = self.default_walk_speed * 0.5

    def stop_parry(self) -> None:
        self.is_parrying = False
        if not self.is_staggered:
            self.walkspeed = self.default_walk_speed

    def block(self) -> None:
        self.is_blocking = True

    def stop_block(self) -> None:
        self.is_blocking = False

    def update_entity(self, dt: float) -> None:
        super().update_entity(dt)
        self.regeneration(dt)


# --- Mob ---


class Mob(Entity):
    # --- stats ---
    maxhealth: int = 100
    maxstagger: int = 60
    default_walk_speed: float = 90.0
    Defaultdmg: int = 8
    default_attack_speed: float = 1.5

    # --- physics ---
    bbox_size: pygame.Vector2 = pygame.Vector2(0.8, 1.8)

    # --- aggro ranges ---
    detect_range_x: float = 200.0
    detect_range_y: float = 120.0
    chase_range_x: float = 350.0
    chase_range_y: float = 200.0

    has_aggro: bool = False

    def __init__(self, x: float, y: float, world: World) -> None:
        super().__init__(
            maxhealth=self.maxhealth,
            maxstagger=self.maxstagger,
            walkspeed=self.default_walk_speed,
            dmg=self.Defaultdmg,
            attackspeed=self.default_attack_speed,
            x=x,
            y=y,
            world=world,
            badEntity=True,
        )
        self.detect_range_x = Mob.detect_range_x
        self.detect_range_y = Mob.detect_range_y
        self.chase_range_x = Mob.chase_range_x
        self.chase_range_y = Mob.chase_range_y
        self.has_aggro = False

    # --- aggro ---

    def focus_player(self, player: Player) -> None:
        dx = abs(player.x - self.x)
        dy = abs(player.y - self.y)

        if not self.has_aggro:
            if dx < self.detect_range_x and dy < self.detect_range_y:
                self.has_aggro = True
        else:
            if dx > self.chase_range_x or dy > self.chase_range_y:
                self.has_aggro = False

    # --- movement ---

    def move_towards_player(self, player: Player, dt: float) -> None:
        if not self.has_aggro or self.is_staggered:
            return

        # set horizontal velocity toward player, then let sweep collision resolve it
        if player.x > self.x:
            self.vel_x = self.walkspeed
        elif player.x < self.x:
            self.vel_x = -self.walkspeed
        else:
            self.vel_x = 0.0

        self.apply_gravity(dt)
        self.apply_velocity(dt)

    def update_focus(self, player: Player, dt: float) -> None:
        self.focus_player(player)
        self.move_towards_player(player, dt)
        super().update_entity(dt)
