from __future__ import annotations

import pygame

from src.inventory import HOTBAR_SLOTS, Stack

SLOT_SIZE = 40
SLOT_PADDING = 4
COLS = HOTBAR_SLOTS
ROWS = 4

COLOR_BG = (30, 30, 30, 200)
COLOR_SLOT = (60, 60, 60, 220)
COLOR_SLOT_HOVER = (90, 90, 90, 240)
COLOR_SLOT_HELD = (120, 100, 40, 240)
COLOR_BORDER = (100, 100, 100, 255)
COLOR_COUNT = (220, 220, 220)


class InventoryRenderer:
    """
    Stateless renderer + minimal UI state (held stack, hovered slot).
    Resolution is always passed explicitly — never cached.

    Usage
    -----
    renderer = InventoryRenderer()

    # event loop (only when open):
    renderer.handle_event(event, inventory.slots, resolution)

    # draw loop (only when open):
    renderer.draw(surface, inventory.slots, resolution)

    # on close:
    renderer.close(inventory.slots)
    """

    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 16)
        self.held: Stack | None = None
        self.held_src_slot: int | None = None
        self.hovered_slot: int | None = None

    # ── geometry (pure) ───────────────────────────────────────────────────────

    def panel_rect(self, resolution: tuple[int, int]) -> pygame.Rect:
        panel_w = COLS * (SLOT_SIZE + SLOT_PADDING) + SLOT_PADDING
        panel_h = ROWS * (SLOT_SIZE + SLOT_PADDING) + SLOT_PADDING
        x = (resolution[0] - panel_w) // 2
        y = (resolution[1] - panel_h) // 2
        return pygame.Rect(x, y, panel_w, panel_h)

    def slot_rect(
        self, slot_idx: int, resolution: tuple[int, int]
    ) -> pygame.Rect | None:
        if slot_idx < 0 or slot_idx >= ROWS * COLS:
            return None
        row, col = divmod(slot_idx, COLS)
        r = self.panel_rect(resolution)
        x = r.x + SLOT_PADDING + col * (SLOT_SIZE + SLOT_PADDING)
        y = r.y + SLOT_PADDING + row * (SLOT_SIZE + SLOT_PADDING)
        return pygame.Rect(x, y, SLOT_SIZE, SLOT_SIZE)

    def slot_at(self, pos: tuple[int, int], resolution: tuple[int, int]) -> int | None:
        for slot_idx in range(ROWS * COLS):
            r = self.slot_rect(slot_idx, resolution)
            if r and r.collidepoint(pos):
                return slot_idx
        return None

    # ── events ────────────────────────────────────────────────────────────────

    def handle_event(
        self,
        event: pygame.event.Event,
        slots: dict[int, Stack],
        resolution: tuple[int, int],
    ) -> bool:
        """Returns True if the event was consumed."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered_slot = self.slot_at(event.pos, resolution)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            slot_idx = self.slot_at(event.pos, resolution)
            if slot_idx is None:
                return False
            if event.button == 1:
                self._left_click(slot_idx, slots)
            elif event.button == 3:
                self._right_click(slot_idx, slots)
            return True

        return False

    def close(self, slots: dict[int, Stack]) -> None:
        """Return held stack to source slot and reset UI state."""
        if self.held is not None and self.held_src_slot is not None:
            slots[self.held_src_slot] = self.held
        self.held = None
        self.held_src_slot = None
        self.hovered_slot = None

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(
        self,
        surface: pygame.Surface,
        slots: dict[int, Stack],
        resolution: tuple[int, int],
    ) -> None:
        surface.fill(pygame.Color(0, 0, 0, 127))
        r = self.panel_rect(resolution)

        bg = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
        bg.fill(COLOR_BG)
        surface.blit(bg, r.topleft)

        for slot_idx in range(ROWS * COLS):
            sr = self.slot_rect(slot_idx, resolution)
            if sr is None:
                continue

            color = (
                COLOR_SLOT_HELD
                if slot_idx == self.held_src_slot
                else COLOR_SLOT_HOVER
                if slot_idx == self.hovered_slot
                else COLOR_SLOT
            )
            slot_surf = pygame.Surface((SLOT_SIZE, SLOT_SIZE), pygame.SRCALPHA)
            slot_surf.fill(color)
            pygame.draw.rect(slot_surf, COLOR_BORDER, slot_surf.get_rect(), 1)
            surface.blit(slot_surf, sr.topleft)

            stack = slots.get(slot_idx)
            if stack is not None:
                self._draw_stack(surface, stack, sr.x, sr.y)

        if self.held is not None:
            mx, my = pygame.mouse.get_pos()
            self._draw_stack(
                surface, self.held, mx - SLOT_SIZE // 2, my - SLOT_SIZE // 2
            )

    # ── click logic ───────────────────────────────────────────────────────────

    def _left_click(self, slot_idx: int, slots: dict[int, Stack]) -> None:
        slot_stack = slots.get(slot_idx)

        if self.held is None:
            if slot_stack is not None:
                self.held = slot_stack
                self.held_src_slot = slot_idx
                slots.pop(slot_idx)
        else:
            held_item, held_count = self.held
            if slot_stack is None:
                slots[slot_idx] = self.held
                self.held = None
                self.held_src_slot = None
            elif slot_stack[0] == held_item:
                from src.inventory import Inventory

                total = slot_stack[1] + held_count
                slots[slot_idx] = (held_item, min(total, Inventory.stack_size))
                leftover = total - Inventory.stack_size
                self.held = (held_item, leftover) if leftover > 0 else None
                if self.held is None:
                    self.held_src_slot = None
            else:
                slots[slot_idx] = self.held
                self.held = slot_stack
                self.held_src_slot = slot_idx

    def _right_click(self, slot_idx: int, slots: dict[int, Stack]) -> None:
        slot_stack = slots.get(slot_idx)

        if self.held is None:
            if slot_stack is not None:
                item, count = slot_stack
                take = max(1, count // 2)
                leave = count - take
                self.held = (item, take)
                self.held_src_slot = slot_idx
                if leave > 0:
                    slots[slot_idx] = (item, leave)
                else:
                    slots.pop(slot_idx)
        else:
            held_item, held_count = self.held
            if slot_stack is None or slot_stack[0] == held_item:
                current = slot_stack[1] if slot_stack else 0
                slots[slot_idx] = (held_item, current + 1)
                remaining = held_count - 1
                self.held = (held_item, remaining) if remaining > 0 else None
                if self.held is None:
                    self.held_src_slot = None

    # ── stack drawing ─────────────────────────────────────────────────────────

    def _draw_stack(
        self, surface: pygame.Surface, stack: Stack, x: int, y: int
    ) -> None:
        item, count = stack
        texture = item.get_texture()
        if texture is not None:
            tex_surf = texture.surface()
            if tex_surf is not None:
                scaled = pygame.transform.scale(
                    tex_surf, (SLOT_SIZE - 8, SLOT_SIZE - 8)
                )
                surface.blit(scaled, (x + 4, y + 4))
        if count > 1:
            text = self.font.render(str(count), True, COLOR_COUNT)
            surface.blit(
                text,
                (
                    x + SLOT_SIZE - text.get_width() - 2,
                    y + SLOT_SIZE - text.get_height() - 2,
                ),
            )
