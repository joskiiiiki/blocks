# Blocks

A 2D sandbox RPG platformer built with Python, pygame-ce and ModernGL.

---

## Development

### NixOS / Nix (recommended)

```sh
nix develop
cd blocks
python3 -m src.main
```

### Manual install

**Python 3.13** is required.

```sh
pip install pygame-ce numpy moderngl PyOpenGL pygame-gui matplotlib platformdirs pyfastnoiselite
```

On Linux you also need the following system libraries available to pygame:

- `libGL`, `libGLU`
- `libX11`, `libXext`, `libXrandr`, `libXi`

On Ubuntu/Debian:
```sh
sudo apt install libgl1 libglu1 libx11-6 libxext6 libxrandr2 libxi6
```

### Running

```sh
cd blocks
python3 -m src.main
```

---

## How to Play

### Movement

| Key | Action |
|-----|--------|
| `A` / `D` | Move left / right |
| `Space` | Jump |
| `Shift` | Sprint |
| `W` / `Space` | Swim up (in water) |
| `S` | Swim down (in water) |
| `Escape` | Quit |

### Mining & Building

| Input | Action |
|-------|--------|
| Left click | Mine block (hold to break) |
| Right click | Place block from hotbar |
| Scroll wheel | Cycle hotbar selection |
| `1`–`9` | Select hotbar slot |

Breaking speed depends on the block and your held tool. A pickaxe is faster on stone and ores, an axe is faster on wood.

### Combat

| Key / Input | Action |
|-------------|--------|
| `R` | Attack |

Enemies drop stagger when hit. Once stagger is depleted they are briefly stunned. Watch your own stagger bar — if it empties you will be stunned too.

Fall damage applies above ~15 blocks of fall height.

### Inventory

| Key | Action |
|-----|--------|
| `E` | Open / close inventory |
| Left click slot | Pick up / place stack |
| Right click slot | Pick up half / place one |

The crafting grid (3×3) is on the right side of the inventory. Place items to see the result appear in the output slot. Click the output to collect it — inputs are consumed automatically.

---

## Crafting Recipes

All recipes are shapeless within the 3×3 grid unless marked as shaped.

### Materials

| Output | Ingredients |
|--------|-------------|
| Planks ×4 | 1 Log (anywhere) |
| Sticks ×4 | 2 Planks (vertical, shaped) |
| Torch ×4 | 1 Coal + 1 Stick (vertical) |
| Copper Torch ×4 | 1 Azurite + 1 Stick (vertical) |

### Tools

| Output | Recipe (shaped) |
|--------|-----------------|
| Pickaxe | 3 Iron Ingots across top row, 2 Sticks below centre |
| Axe | Flipped L: 2 Ingots top row + 1 Ingot below left |
| Sword | 1 Iron Ingot then 1 Stick below (vertical, 1×2) |
| Bow | Iron Ingots top and bottom centre, Sticks on left and right |
| Arrow ×4 | 1 Iron Ingot then 1 Stick below (vertical) |

### Armor

| Output | Recipe |
|--------|--------|
| Helmet | 2 Iron Ingots side by side (1×2) |
| Chestplate | 3×3 Iron Ingots except top-centre |
| Pants | Full top row + outer columns of middle row |
| Shoes | Outer two columns of bottom two rows |

---

## Ores

| Ore | Depth range | Drop |
|-----|------------|------|
| Coal Ore | Surface → 240 | Coal |
| Iron Ore | Near surface → 180 | Iron Ingot |
| Azurite Ore | 5 → 120 | Azurite |
| Emerald Ore | 5 → 80 | Emerald |
| IronQuartz Ore | 0 → 60 | IronQuartz Ingot |
| Diamond Ore | 0 → 30 | Diamond |
| Shadow Ore | 0 → 40 | Shadow Gem |
| Void Ore | 0 → 15 | Void Shard |

Depth 0 is the bottom of the world. Rarer ores spawn deeper and in smaller veins.

---

## World Data

Worlds are saved to your local data directory:

- **Linux:** `~/.local/share/blocks/world-1/`
- **Windows:** `%APPDATA%\blocks\world-1\`
- **macOS:** `~/Library/Application Support/blocks/world-1/`

Each world contains chunk files organised by region, and a `player.json` for player state.

### Resetting the world

Delete the world directory:

```sh
# Linux
rm -rf ~/.local/share/blocks/world-1/

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:APPDATA\blocks\world-1"
```

### Removing a stale lock

If the game crashed or was killed, a `.lock` file may prevent the world from loading. Remove it manually:

```sh
# Linux
rm ~/.local/share/blocks/world-1/.lock

# Windows (PowerShell)
Remove-Item "$env:APPDATA\blocks\world-1\.lock"
```
