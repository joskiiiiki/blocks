import math
def world_to_camera(cam_x: float, cam_y: float, world_x: float, world_y: float) -> tuple[float, float]:
    """Convert world space coordinates to camera space coordinates"""

    return world_x - cam_x, world_y - cam_y

# block coordinates with origin from 0 0
def camera_to_world(cam_x: float, cam_y: float, sx: float, sy: float) -> tuple[float, float]:
    """Convert world space coordinates relative to camera to world coordinates from 0 0"""

    return sx + cam_x, sy + cam_y

def world_to_screen(cam_x: float, cam_y: float, world_x: float, world_y: float, screen_width: float, screen_height: float, tile_size: int) -> tuple[float, float]:
    """Convert world space coordinates (blocks) to screen space coordinates (pixels)"""

    camera_x, camera_y = world_to_camera(cam_x, cam_y, world_x, world_y)
    screen_x = camera_x * tile_size + screen_width // 2
    screen_y = screen_height // 2 - camera_y * tile_size

    return screen_x, screen_y

def screen_to_world(cam_x: float, cam_y: float, screen_x: float, screen_y: float, screen_width: float, screen_height: float, tile_size: int) -> tuple[float, float]:
    """Convert screen space coordinates (pixels) to world space coordinates (blocks)"""

    from_camera_x = (screen_x - screen_width // 2) / tile_size
    from_camera_y = (screen_height // 2 - screen_y) / tile_size

    return camera_to_world(cam_x, cam_y, from_camera_x, from_camera_y)

def to_block(x: float, y: float) -> tuple[int, int]:
    return int(math.floor(x)), int(math.floor(y))
