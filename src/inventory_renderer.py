from __future__ import annotations

from collections.abc import Callable

import pygame

from src.inventory import HOTBAR_SLOTS, Inventory, Stack

SLOT_SIZE    = 40
SLOT_PADDING = 4
COLS         = HOTBAR_SLOTS
ROWS         = 4
CRAFT_COLS   = 3
CRAFT_ROWS   = 3
CRAFT_GAP    = 16
ARROW_W      = 32

COLOR_BG         = (30,  30,  30,  200)
COLOR_SLOT       = (60,  60,  60,  220)
COLOR_SLOT_HOVER = (90,  90,  90,  240)
COLOR_SLOT_HELD  = (120, 100, 40,  240)
COLOR_SLOT_OUT   = (40,  70,  40,  220)
COLOR_BORDER     = (100, 100, 100, 255)
COLOR_COUNT      = (220, 220, 220)
COLOR_ARROW      = (180, 180, 180)

CRAFT_SLOT_BASE   = 1000
CRAFT_OUTPUT_SLOT = 1009

CraftingRecipe = Callable[[list[Stack | None]], Stack | None]


class InventoryRenderer:
    """
    Minimal UI state (held stack, hovered slot).
    Resolution is always passed explicitly — never cached.

    Crafting grid: slot indices 1000-1008 (3×3), output at 1009.
    Pass a recipe: (grid: list[Stack | None]) -> Stack | None
    """

    def __init__(self, recipe: CraftingRecipe | None = None) -> None:
        self.font         = pygame.font.Font(None, 16)
        self.held:          Stack | None = None
        self.held_src_slot: int   | None = None
        self.hovered_slot:  int   | None = None
        self._pressed_slot: int   | None = None
        self.recipe      = recipe
        self.craft_slots: dict[int, Stack] = {}

    # ── geometry ──────────────────────────────────────────────────────────────

    def panel_rect(self, resolution: tuple[int, int]) -> pygame.Rect:
        w = COLS * (SLOT_SIZE + SLOT_PADDING) + SLOT_PADDING
        h = ROWS * (SLOT_SIZE + SLOT_PADDING) + SLOT_PADDING
        return pygame.Rect((resolution[0] - w) // 2, (resolution[1] - h) // 2, w, h)

    def craft_panel_rect(self, resolution: tuple[int, int]) -> pygame.Rect:
        inv = self.panel_rect(resolution)
        w = CRAFT_COLS * (SLOT_SIZE + SLOT_PADDING) + SLOT_PADDING + ARROW_W + SLOT_SIZE + SLOT_PADDING
        h = CRAFT_ROWS * (SLOT_SIZE + SLOT_PADDING) + SLOT_PADDING
        return pygame.Rect(inv.right + CRAFT_GAP, inv.y + (inv.height - h) // 2, w, h)

    def slot_rect(self, slot_idx: int, resolution: tuple[int, int]) -> pygame.Rect | None:
        if CRAFT_SLOT_BASE <= slot_idx <= CRAFT_OUTPUT_SLOT:
            return self._craft_slot_rect(slot_idx, resolution)
        if not (0 <= slot_idx < ROWS * COLS):
            return None
        row, col = divmod(slot_idx, COLS)
        ox, oy = self.panel_rect(resolution).topleft
        return pygame.Rect(
            ox + SLOT_PADDING + col * (SLOT_SIZE + SLOT_PADDING),
            oy + SLOT_PADDING + row * (SLOT_SIZE + SLOT_PADDING),
            SLOT_SIZE, SLOT_SIZE,
        )

    def _craft_slot_rect(self, slot_idx: int, resolution: tuple[int, int]) -> pygame.Rect | None:
        cp = self.craft_panel_rect(resolution)
        if slot_idx == CRAFT_OUTPUT_SLOT:
            return pygame.Rect(
                cp.x + SLOT_PADDING + CRAFT_COLS * (SLOT_SIZE + SLOT_PADDING) + ARROW_W,
                cp.y + SLOT_PADDING + (CRAFT_ROWS // 2) * (SLOT_SIZE + SLOT_PADDING),
                SLOT_SIZE, SLOT_SIZE,
            )
        local = slot_idx - CRAFT_SLOT_BASE
        if not (0 <= local < CRAFT_COLS * CRAFT_ROWS):
            return None
        row, col = divmod(local, CRAFT_COLS)
        return pygame.Rect(
            cp.x + SLOT_PADDING + col * (SLOT_SIZE + SLOT_PADDING),
            cp.y + SLOT_PADDING + row * (SLOT_SIZE + SLOT_PADDING),
            SLOT_SIZE, SLOT_SIZE,
        )

    def slot_at(self, pos: tuple[int, int], resolution: tuple[int, int]) -> int | None:
        for slot_idx in range(ROWS * COLS):
            r = self.slot_rect(slot_idx, resolution)
            if r and r.collidepoint(pos):
                return slot_idx
        for i in range(CRAFT_COLS * CRAFT_ROWS):
            slot_idx = CRAFT_SLOT_BASE + i
            r = self.slot_rect(slot_idx, resolution)
            if r and r.collidepoint(pos):
                return slot_idx
        r = self.slot_rect(CRAFT_OUTPUT_SLOT, resolution)
        if r and r.collidepoint(pos):
            return CRAFT_OUTPUT_SLOT
        return None

    # ── events ────────────────────────────────────────────────────────────────

    def handle_event(
        self,
        event: pygame.event.Event,
        slots: dict[int, Stack],
        resolution: tuple[int, int],
    ) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered_slot = self.slot_at(event.pos, resolution)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            self._pressed_slot = self.slot_at(event.pos, resolution)
            return False

        if event.type != pygame.MOUSEBUTTONUP:
            return False

        slot_idx = self.slot_at(event.pos, resolution)
        if slot_idx is None or slot_idx != self._pressed_slot:
            self._pressed_slot = None
            return False

        self._pressed_slot = None
        if event.button == 1:
            self._left_click(slot_idx, slots)
        elif event.button == 3:
            self._right_click(slot_idx, slots)
        return True

    def close(self, slots: dict[int, Stack]) -> None:
        if self.held is not None and self.held_src_slot is not None:
            slots[self.held_src_slot] = self.held
        self.held          = None
        self.held_src_slot = None
        self.hovered_slot  = None
        for stack in self.craft_slots.values():
            _add_to_slots(slots, stack)
        self.craft_slots.clear()

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(
        self,
        surface: pygame.Surface,
        slots: dict[int, Stack],
        resolution: tuple[int, int],
    ) -> None:
        dim = pygame.Surface(resolution, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        surface.blit(dim, (0, 0))

        self._draw_panel(surface, self.panel_rect(resolution))
        for slot_idx in range(ROWS * COLS):
            sr = self.slot_rect(slot_idx, resolution)
            if sr:
                self._draw_slot(surface, sr, slot_idx, slots)

        cp = self.craft_panel_rect(resolution)
        self._draw_panel(surface, cp)
        for i in range(CRAFT_COLS * CRAFT_ROWS):
            slot_idx = CRAFT_SLOT_BASE + i
            sr = self.slot_rect(slot_idx, resolution)
            if sr:
                self._draw_slot(surface, sr, slot_idx, self.craft_slots)

        arrow = self.font.render("->", True, COLOR_ARROW)
        surface.blit(arrow, (
            cp.x + SLOT_PADDING + CRAFT_COLS * (SLOT_SIZE + SLOT_PADDING) + ARROW_W // 2 - arrow.get_width() // 2,
            cp.y + cp.height // 2 - arrow.get_height() // 2,
        ))

        osr = self.slot_rect(CRAFT_OUTPUT_SLOT, resolution)
        if osr:
            output = self._compute_output()
            color = COLOR_SLOT_HOVER if self.hovered_slot == CRAFT_OUTPUT_SLOT else COLOR_SLOT_OUT
            self._draw_slot_bg(surface, osr, color)
            if output is not None:
                self._draw_stack(surface, output, osr.x, osr.y)

        if self.held is not None:
            mx, my = pygame.mouse.get_pos()
            self._draw_stack(surface, self.held, mx - SLOT_SIZE // 2, my - SLOT_SIZE // 2)

    def _draw_panel(self, surface: pygame.Surface, r: pygame.Rect) -> None:
        bg = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
        bg.fill(COLOR_BG)
        surface.blit(bg, r.topleft)

    def _draw_slot(
        self,
        surface: pygame.Surface,
        sr: pygame.Rect,
        slot_idx: int,
        slots: dict[int, Stack],
    ) -> None:
        color = (
            COLOR_SLOT_HELD  if slot_idx == self.held_src_slot else
            COLOR_SLOT_HOVER if slot_idx == self.hovered_slot  else
            COLOR_SLOT
        )
        self._draw_slot_bg(surface, sr, color)
        stack = slots.get(slot_idx)
        if stack is not None:
            self._draw_stack(surface, stack, sr.x, sr.y)

    def _draw_slot_bg(self, surface: pygame.Surface, sr: pygame.Rect, color: tuple) -> None:
        s = pygame.Surface((SLOT_SIZE, SLOT_SIZE), pygame.SRCALPHA)
        s.fill(color)
        pygame.draw.rect(s, COLOR_BORDER, s.get_rect(), 1)
        surface.blit(s, sr.topleft)

    def _draw_stack(self, surface: pygame.Surface, stack: Stack, x: int, y: int) -> None:
        item, count = stack
        texture = item.get_texture()
        if texture is not None:
            tex_surf = texture.surface()
            if tex_surf is not None:
                surface.blit(pygame.transform.scale(tex_surf, (SLOT_SIZE - 8, SLOT_SIZE - 8)), (x + 4, y + 4))
        if count > 1:
            text = self.font.render(str(count), True, COLOR_COUNT)
            surface.blit(text, (x + SLOT_SIZE - text.get_width() - 2, y + SLOT_SIZE - text.get_height() - 2))

    # ── crafting ──────────────────────────────────────────────────────────────

    def _compute_output(self) -> Stack | None:
        if self.recipe is None:
            return None
        return self.recipe([self.craft_slots.get(CRAFT_SLOT_BASE + i) for i in range(CRAFT_COLS * CRAFT_ROWS)])

    def _consume_craft_inputs(self) -> None:
        for i in range(CRAFT_COLS * CRAFT_ROWS):
            slot_idx = CRAFT_SLOT_BASE + i
            stack = self.craft_slots.get(slot_idx)
            if stack is None:
                continue
            item, count = stack
            if count <= 1:
                self.craft_slots.pop(slot_idx)
            else:
                self.craft_slots[slot_idx] = (item, count - 1)

    # ── click logic ───────────────────────────────────────────────────────────

    def _target_slots(self, slot_idx: int, inv_slots: dict[int, Stack]) -> dict[int, Stack]:
        return self.craft_slots if CRAFT_SLOT_BASE <= slot_idx < CRAFT_OUTPUT_SLOT else inv_slots

    def _left_click(self, slot_idx: int, slots: dict[int, Stack]) -> None:
        if slot_idx == CRAFT_OUTPUT_SLOT:
            self._left_click_output()
            return

        target      = self._target_slots(slot_idx, slots)
        slot_stack  = target.get(slot_idx)

        if self.held is None:
            if slot_stack is None:
                return
            self.held          = slot_stack
            self.held_src_slot = slot_idx
            target.pop(slot_idx)
            return

        held_item, held_count = self.held

        if slot_stack is None:
            target[slot_idx]   = self.held
            self.held          = None
            self.held_src_slot = None
            return

        if slot_stack[0] != held_item:
            target[slot_idx]   = self.held
            self.held          = slot_stack
            self.held_src_slot = slot_idx
            return

        # same item — merge
        total              = slot_stack[1] + held_count
        target[slot_idx]   = (held_item, min(total, Inventory.stack_size))
        leftover           = total - Inventory.stack_size
        self.held          = (held_item, leftover) if leftover > 0 else None
        if self.held is None:
            self.held_src_slot = None

    def _left_click_output(self) -> None:
        output = self._compute_output()
        if output is None:
            return

        if self.held is None:
            self.held          = output
            self.held_src_slot = None
            self._consume_craft_inputs()
            return

        if self.held[0] != output[0]:
            return
        if self.held[1] + output[1] > Inventory.stack_size:
            return

        self.held = (self.held[0], self.held[1] + output[1])
        self._consume_craft_inputs()

    def _right_click(self, slot_idx: int, slots: dict[int, Stack]) -> None:
        if slot_idx == CRAFT_OUTPUT_SLOT:
            return

        target     = self._target_slots(slot_idx, slots)
        slot_stack = target.get(slot_idx)

        if self.held is None:
            if slot_stack is None:
                return
            item, count        = slot_stack
            take               = max(1, count // 2)
            leave              = count - take
            self.held          = (item, take)
            self.held_src_slot = slot_idx
            if leave > 0:
                target[slot_idx] = (item, leave)
            else:
                target.pop(slot_idx)
            return

        held_item, held_count = self.held
        if slot_stack is not None and slot_stack[0] != held_item:
            return

        current          = slot_stack[1] if slot_stack else 0
        target[slot_idx] = (held_item, current + 1)
        remaining        = held_count - 1
        self.held        = (held_item, remaining) if remaining > 0 else None
        if self.held is None:
            self.held_src_slot = None


def _add_to_slots(slots: dict[int, Stack], stack: Stack) -> None:
    item, count = stack
    for idx, s in slots.items():
        if s[0] != item or s[1] >= Inventory.stack_size:
            continue
        take       = min(Inventory.stack_size - s[1], count)
        slots[idx] = (item, s[1] + take)
        count     -= take
        if count <= 0:
            return
    for idx in range(ROWS * COLS):
        if idx not in slots:
            slots[idx] = (item, count)
            return
