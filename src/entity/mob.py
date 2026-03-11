# ---------------------------------------------------------------------------
# Mob
# ---------------------------------------------------------------------------
import pygame

from src.entity.entity import ENTITY_REGISTRY, Entity
from src.entity.stats import ZOMBIE_STATS, EntityStats


class Mob(Entity):
    detect_range_x: float = 200.0
    detect_range_y: float = 120.0
    chase_range_x: float = 350.0
    chase_range_y: float = 200.0

    attack_bbox_grow = pygame.Vector2(3.0, 0.5)

    def __init__(
        self,
        x: float,
        y: float,
        stats: EntityStats = ZOMBIE_STATS,
    ) -> None:
        super().__init__(stats, x, y)
        self.has_aggro = False

    @classmethod
    def from_data(cls, data: dict) -> "Mob":
        mob_cls = ENTITY_REGISTRY.get(data["type"])
        if mob_cls is None:
            mob = cls(x=data["x"], y=data["y"])  # cls is Mob, stats has default
        else:
            mob = mob_cls(x=data["x"], y=data["y"])
        mob.from_json(data)
        return mob  # type: ignore[return-value]

    def to_json(self) -> dict:
        return super().to_json() | {
            "type": "Mob",  # subclasses override this
            "has_aggro": self.has_aggro,
        }

    def from_json(self, data: dict) -> None:
        super().from_json(data)
        self.has_aggro = data.get("has_aggro", False)

    def focus_player(self, pos: tuple[float, float]) -> None:
        dx = abs(pos[0] - self.x)
        dy = abs(pos[1] - self.y)

        if not self.has_aggro:
            if dx < self.detect_range_x and dy < self.detect_range_y:
                self.has_aggro = True
        else:
            if dx > self.chase_range_x or dy > self.chase_range_y:
                self.has_aggro = False

    def move_towards_player(self, pos: tuple[float, float]) -> None:
        if not self.has_aggro or self.is_staggered:
            self.vel_x = 0.0
            return

        dx = pos[0] - self.bounding_box.center.x
        dy = pos[1] - self.bounding_box.center.y
        if abs(dx) + abs(dy) > 1.0:
            self.vel_x = (
                self.walk_speed if dx > 0 else -self.walk_speed if dx < 0 else 0.0
            )

            if self.in_water:
                if self.x_collision or dy > 0:
                    self.swim_up()
                else:
                    self.swim_down()
        else:
            self.vel_x = 0.0

    def update_focus(self, pos: tuple[float, float], dt: float, in_water: bool) -> None:
        self.focus_player(pos)
        self.move_towards_player(pos)
        super().update_entity(dt, in_water)

    def draw(
        self,
        surface: pygame.Surface,
        resolution: tuple[int, int],
        player_x: float,
        player_y: float,
    ):
        pass
