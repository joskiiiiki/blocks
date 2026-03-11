import pygame

from src.entity import Entity, EntityStats, Player


class Mob(Entity):
    # --- stats ---
    maxhealth: int = 100
    maxstagger: int = 60
    default_walk_speed: float = 5
    Defaultdmg: int = 8
    default_attack_speed: float = 1.5

    # --- physics ---
    bbox_size: pygame.Vector2 = pygame.Vector2(0.8, 1.8)

    # --- aggro ranges ---
    detect_range_x: float = 200.0
    detect_range_y: float = 120.0
    chase_range_x: float = 350.0
    chase_range_y: float = 200.0

    def __init__(self, x: float, y: float, stats: EntityStats) -> None:
        super().__init__(stats=stats, x=x, y=y)
        self.has_aggro: bool = False

    def focus_player(self, player: Player) -> None:
        dx = abs(player.x - self.x)
        dy = abs(player.y - self.y)
        if not self.has_aggro:
            if dx < self.detect_range_x and dy < self.detect_range_y:
                self.has_aggro = True
        else:
            if dx > self.chase_range_x or dy > self.chase_range_y:
                self.has_aggro = False

    def update_focus(self, player: Player, dt: float) -> None:
        self.focus_player(player)

        if not self.has_aggro or self.is_staggered:
            self.vel_x = 0.0
            return

        dx = player.bounding_box.center.x - self.bounding_box.center.x
        dy = player.bounding_box.center.y - self.bounding_box.center.y
        if abs(dx) + abs(dy) > 1.0:
            self.vel_x = (
                self.walkspeed if dx > 0 else -self.walkspeed if dx < 0 else 0.0
            )

            if self.in_water:
                if self.x_collision or dy > 0:
                    self.swim_up()
                else:
                    self.swim_down()
        else:
            self.vel_x = 0.0
