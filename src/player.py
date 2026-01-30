from collections.abc import Callable
import pygame

PLAYER_SPRITE = pygame.Surface((32, 64))

class Player:
    vel_x: float = 0
    vel_y: float = 0
    speed: float = 5
    sprint_speed: float = 8
    jump_power: float = 14
    gravity: float = -0.8 / 100
    on_ground: bool = False
    sliding: bool = False
    slide_timer: float = 0
    x: float
    y: float

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


    def handle_input(self) -> None:
        keys = pygame.key.get_pressed()

        self.vel_x = 0

        speed = self.sprint_speed if keys[pygame.K_LSHIFT] else self.speed

        if keys[pygame.K_a]:
            self.vel_x = -speed
        if keys[pygame.K_d]:
            self.vel_x = speed

        # springen
        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y = self.jump_power
            self.on_ground = False

        # sliden (nur am Boden + mit Sprint)
        if keys[pygame.K_s] and keys[pygame.K_LSHIFT] and self.on_ground and not self.sliding:
            self.sliding = True
            self.slide_timer = 20
            self.vel_x *= 2

    def apply_gravity(self) -> None:
        self.vel_y += self.gravity

    def update(self) -> None:
        self.handle_input()

        if self.sliding:
            self.slide_timer -= 1
            self.vel_x *= 0.95
            if self.slide_timer <= 0:
                self.sliding = False

        self.apply_gravity()

        return self.


    def set_position(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def draw(self, screen: pygame.Surface) -> None:
        x = screen.width // 2 - 16
        y = screen.height // 2 - 32
        screen.blit(PLAYER_SPRITE, (x, y))
