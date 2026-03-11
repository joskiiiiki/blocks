import pygame

from src.collision import BoundingBox


class BoundingBoxed:
    bounding_box: BoundingBox

    def __init__(self, bounding_box: BoundingBox) -> None:
        self.bounding_box = bounding_box

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
