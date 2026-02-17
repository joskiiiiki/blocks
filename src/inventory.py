from collections.abc import Iterable

import pygame
from pygame.key import ScancodeWrapper

from src.assets import HOTBAR, HOTBAR_RIM, HOTBAR_SELECTED
from src.blocks import Block, Item

type Stack = tuple[Item, int]

HOTBAR_SLOTS = 9


class Inventory:
    stack_size: int = 100
    slots: dict[int, Stack] = {}
    slot_count: int = HOTBAR_SLOTS * 4
    used_slots: int = 0

    def add_stack(self, stack: Stack) -> int:
        item, count = stack
        for slot in self.slots:
            if self.slots[slot][0].value == item.value and count > 0:
                in_slot = self.slots[slot][1]

                new = min(in_slot + count, self.stack_size)

                self.slots[slot] = (item, new)

                count -= new - in_slot

        if count <= 0:
            return 0

        for slot_id in range(self.slot_count):
            if slot_id not in self.slots:
                self.slots[slot_id] = (item, count)
                new = min(count, self.stack_size)
                self.slots[slot_id] = (item, new)
                self.used_slots += 1

                count -= new
                if count <= 0:
                    break

        return count

    def get_slot(self, index: int) -> Stack | None:
        return self.slots.get(index)

    def set_slot(self, index: int, stack: Stack | None):
        if stack is None or stack[1] == 0:
            self.slots.pop(index, None)
        else:
            self.slots[index] = stack
        self.used_slots += -1 if stack is None or stack[1] == 0 else 1

    def set_slot_count(self, index: int, count: int) -> int:
        if index not in self.slots:
            return 0
        if count < 0 or count > self.stack_size:
            return 0
        item = self.slots[index][0]
        if count == 0:
            self.slots.pop(index, None)
            self.used_slots -= 1
        else:
            self.slots[index] = (item, count)
        return count

    def increment_slot_count(self, index: int, count: int) -> int:
        if index not in self.slots:
            return 0
        current_count = self.slots[index][1]
        new_count = min(self.stack_size, current_count + count)
        if new_count == 0:
            self.slots.pop(index, None)
            self.used_slots -= 1
        else:
            self.slots[index] = (self.slots[index][0], new_count)

        return new_count

    def get_hotbar_slots(self, slot_count: int = HOTBAR_SLOTS) -> list[Stack | None]:
        return [self.slots.get(i) for i in range(slot_count)]

    def get_block_in_slot(self, index: int) -> Block | None:
        stack = self.get_slot(index)
        return stack[0].get_block() if stack else None


KEY_TO_SLOT: dict[int, int] = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
    pygame.K_5: 4,
    pygame.K_6: 5,
    pygame.K_7: 6,
    pygame.K_8: 7,
    pygame.K_9: 8,
}


class Hotbar:
    selected_slot: int = 0
    slot_count: int = HOTBAR_SLOTS
    slot_size_px: tuple[int, int]
    rim_size: int = HOTBAR_RIM
    font: pygame.font.Font

    def position(self, resolution: tuple[int, int]) -> tuple[int, int]:
        x = (resolution[0] - HOTBAR.width) // 2
        y = resolution[1] - HOTBAR.height
        return x, y

    def __init__(self):
        self.slot_size_px = (HOTBAR.width // self.slot_count, HOTBAR.height)
        self.font = pygame.font.Font(None, 16)

    def draw(self, screen: pygame.Surface, slots: Iterable[Stack | None]):
        position = self.position(screen.get_size())
        screen.blit(HOTBAR, position)

        for i, slot in enumerate(slots):
            x = position[0] + self.slot_size_px[0] * i
            y = position[1]

            if i == self.selected_slot:
                screen.blit(HOTBAR_SELECTED, (x, y))

            if slot is None:
                continue

            item, count = slot
            texture = item.get_texture()

            if texture is not None:
                item_x = x + (self.slot_size_px[0] - texture.width) // 2
                item_y = y + (self.slot_size_px[1] - texture.height) // 2
                screen.blit(texture, (item_x, item_y))

            if count > 1:
                text = self.font.render(str(count), True, (255, 255, 255))
                screen.blit(text, (x + self.rim_size, y + self.rim_size))

    def handle_keys(self, keys: ScancodeWrapper):
        for key, slot in KEY_TO_SLOT.items():
            if keys[key]:
                self.selected_slot = slot

    def handle_scroll(self, direction):
        self.selected_slot += -1 if direction > 0 else 1
        # Wrap-Around
        self.selected_slot %= self.slot_count
