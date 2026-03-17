import pathlib

import pygame
import random

_ROOT = pathlib.Path(__file__).parent.resolve()

class SoundManager:
    def __init__(self):
        self.sfx_volume = 0.7
        self.music_volume = 0.5

        self.sounds = {
            "walk":   pygame.mixer.Sound(_ROOT / "walk.ogg"),  # z.B. "assets/sounds/player_walk.wav"
            "player_damage": pygame.mixer.Sound(_ROOT / "damage.ogg"),  # z.B. "assets/sounds/player_damage.wav"
            "break":   pygame.mixer.Sound(_ROOT / "break.ogg"),  # z.B. "assets/sounds/player_mine.wav"
            "zombie_grunt":  pygame.mixer.Sound(_ROOT / "zombie.ogg"),  # z.B. "assets/sounds/zombie_grunt.wav"
            "zombie_damage": pygame.mixer.Sound(_ROOT / "zombie_damage.ogg"),  # z.B. "assets/sounds/zombie_damage.wav"
        }

        for sound in self.sounds.values():
            sound.set_volume(self.sfx_volume)

        self.walk_channel = pygame.mixer.Channel(0)
        self.cooldowns = {}

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()

    def play_with_cooldown(self, name, cooldown_ms=300):
        now = pygame.time.get_ticks()
        last = self.cooldowns.get(name, 0)
        if now - last >= cooldown_ms:
            self.play(name)
            self.cooldowns[name] = now

    def play_walk(self, is_walking):
        if is_walking:
            if not self.walk_channel.get_busy():
                self.walk_channel.play(self.sounds["player_walk"], loops=-1)
        else:
            self.walk_channel.stop()

    def play_zombie_grunt(self):
        if random.random() < 0.002:
            self.play_with_cooldown("zombie_grunt", cooldown_ms=1000)

    def play_music(self, path="hier datei"):  # z.B. "assets/music/background.ogg"
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(self.music_volume)
        pygame.mixer.music.play(loops=-1)

    def stop_music(self):
        pygame.mixer.music.stop()


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
