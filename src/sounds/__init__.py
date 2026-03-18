import os
import pathlib
import random

import pygame
from pygame.math import clamp

from src.entity.entity import Entity
from src.entity.mob import Mob

_ROOT = pathlib.Path(__file__).parent.resolve()

WALK_FADE_OUT = 100
WALK_FADE_IN = 10


class SoundManager:
    def __init__(self):
        self.sfx_volume: float = 0.7
        self.music_volume: float = 0.3

        self.sounds: dict[str, pygame.mixer.Sound] = {
            "music": pygame.mixer.Sound(_ROOT / "music.ogg"),
            "walk": pygame.mixer.Sound(_ROOT / "walk.ogg"),
            "player_damage": pygame.mixer.Sound(_ROOT / "damage.ogg"),
            "break": pygame.mixer.Sound(_ROOT / "break.ogg"),
            "break_single": pygame.mixer.Sound(_ROOT / "break_single.ogg"),
            "zombie_grunt": pygame.mixer.Sound(_ROOT / "zombie.ogg"),
            "zombie_damage": pygame.mixer.Sound(_ROOT / "zombie_damage.ogg"),
        }

        for sound in self.sounds.values():
            sound.set_volume(self.sfx_volume)

        self.walk_channel = pygame.mixer.Channel(0)
        self.cooldowns = {}

        self._mob_channels: dict[int, pygame.mixer.Channel] = {}

    def play(self, name: str):
        if name in self.sounds:
            self.sounds[name].play()

    def play_non_overlapping(self, name: str):
        if name in self.sounds:
            cooldown = self.sounds[name].get_length() * 1000
            self.play_with_cooldown(name, cooldown)

    def play_with_cooldown(self, name: str, cooldown_ms: float = 300):
        now = pygame.time.get_ticks()
        last = self.cooldowns.get(name, 0)
        if now - last >= cooldown_ms:
            self.play(name)
            self.cooldowns[name] = now

    def play_walk(self, is_walking: bool):
        if is_walking:
            if not self.walk_channel.get_busy():
                self.walk_channel.play(
                    self.sounds["walk"], loops=-1, fade_ms=WALK_FADE_IN
                )
        else:
            self.walk_channel.fadeout(WALK_FADE_OUT)

    def play_zombie_grunt(self):
        if random.random() < 0.002:
            self.play_with_cooldown("zombie_grunt", cooldown_ms=1000)

    def play_music(self, path: os.PathLike = _ROOT / "music.ogg"):
        pygame.mixer.music.set_volume(self.music_volume)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(loops=-1)

    def stop_music(self):
        pygame.mixer.music.stop()

    def play_at(
        self,
        name: str,
        source: tuple[float, float],
        listener: tuple[float, float],
        max_dist: float = 20.0,
    ) -> None:
        sound = self.sounds.get(name)
        if sound is None:
            return

        dx = source[0] - listener[0]
        dy = source[1] - listener[1]
        dist = (dx**2 + dy**2) ** 0.5

        if dist > max_dist:
            return

        volume = 1.0 - (dist / max_dist)
        pan = clamp(dx / max_dist, -1.0, 1.0)

        left = volume * (1.0 - max(0.0, pan))
        right = volume * (1.0 + min(0.0, pan))

        channel = sound.play()
        if channel:
            channel.set_volume(left * self.sfx_volume, right * self.sfx_volume)

    def update_mob_walk(
        self,
        mob: Entity,
        player_pos: tuple[float, float],
        max_dist: float = 20.0,
    ) -> None:
        if mob.is_dead or not mob.walking:
            self.stop_mob_walk(mob)
            return

        sound = self.sounds.get("walk")
        if sound is None:
            return

        mob_id = mob.id
        channel = self._mob_channels.get(mob_id)

        if channel is None or not channel.get_busy():
            channel = sound.play(loops=-1, fade_ms=WALK_FADE_IN)
            if channel is None:
                return
            self._mob_channels[mob_id] = channel

        dx = mob.x - player_pos[0]
        dy = mob.y - player_pos[1]
        dist = (dx**2 + dy**2) ** 0.5
        volume = max(0.0, 1.0 - dist / max_dist) * self.sfx_volume
        pan = max(-1.0, min(1.0, dx / max_dist))
        channel.set_volume(
            volume * (1.0 - max(0.0, pan)),
            volume * (1.0 + min(0.0, pan)),
        )

    def stop_mob_walk(self, mob: Entity) -> None:
        channel = self._mob_channels.pop(mob.id, None)
        if channel:
            channel.fadeout(WALK_FADE_OUT)

    def stop_all_mob_walks(self) -> None:
        for channel in self._mob_channels.values():
            channel.stop()
        self._mob_channels.clear()


# ── Nutzung in der Game-Loop ──────────────────────────────────────────────────
#
# Einmalig VOR pygame.init():
#   pygame.mixer.pre_init(44100, -16, 2, 512)
#   pygame.init()
#
# Einmalig nach dem Init:
#   sound_manager = SoundManager()
#   sound_manager.play_music()
#
# In der Loop:
#   moving = keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]
#   sound_manager.play_walk(moving)
#
#   if player.took_damage:
#       sound_manager.play("player_damage")
#
#   if player.is_mining:
#       sound_manager.play_with_cooldown("player_mine", cooldown_ms=400)
#
#   for zombie in zombies:
#       sound_manager.play_zombie_grunt()
#       if zombie.took_damage:
#           sound_manager.play("zombie_damage")
