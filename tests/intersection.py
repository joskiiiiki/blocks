from src.blocks import Block
from src.world import Chunk, ChunkManager, IntersectionDirection


def test_ortho_y():
    chunk_manager = ChunkManager()

    chunk = chunk_manager.empty_chunk()
    chunk[::, 10] = Block.STONE.value

    chunk_manager.chunk_cache[0] = chunk

    def downwards():
        ctx = chunk_manager.intersect_ortho_line_y(1, 11, 1, 10)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.DOWN
        assert x == 1
        assert y >= 11 and y < 11.5

    def upwards():
        ctx = chunk_manager.intersect_ortho_line_y(1, 9, 1, 10)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.UP
        assert x == 1
        assert y < 10 and y > 9.5

    downwards()
    upwards()


def test_ortho_x():
    chunk_manager = ChunkManager()
    chunk = chunk_manager.empty_chunk()
    chunk[10, ::] = Block.STONE.value
    chunk_manager.chunk_cache[0] = chunk

    def leftward():
        ctx = chunk_manager.intersect_ortho_line_x(9, 0, 10, 0)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.RIGHT
        assert x < 10 and x > 9.5
        assert y == 0

    def rightward():
        ctx = chunk_manager.intersect_ortho_line_x(11, 0, 10, 0)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.LEFT
        assert x >= 11 and x < 11.5
        assert y == 0

    leftward()


def display_chunk(chunk: Chunk, x_min: int, x_max: int, y_min: int, y_max: int):
    for y in range(y_max, y_min - 1, -1):
        for x in range(x_min, x_max + 1):
            print(chunk[x, y], end=" ")
        print()


def test_diagonal_horizontal_wall():
    chunk_manager = ChunkManager()
    chunk = chunk_manager.empty_chunk()
    chunk[::, 10] = Block.STONE.value
    chunk_manager.chunk_cache[0] = chunk

    def downward():
        ctx = chunk_manager.intersect_diagonal_horizontal_wall(9, 11, 10, 9)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.DOWN
        assert x >= 9 and x <= 10
        assert y < 12 and y >= 11

    def upward():
        ctx = chunk_manager.intersect_diagonal_horizontal_wall(9, 9, 10, 11)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.UP
        assert x >= 9 and x <= 10
        assert y >= 9 and y < 10

    downward()
    upward()


def test_diagonal_vertical_wall():
    chunk_manager = ChunkManager()
    chunk = chunk_manager.empty_chunk()
    chunk[10, ::] = Block.STONE.value
    chunk_manager.chunk_cache[0] = chunk

    def leftward():
        ctx = chunk_manager.intersect_diagonal_vertical_wall(9, 11, 10, 10)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        print(direction)
        assert direction == IntersectionDirection.RIGHT
        print(x)
        assert x >= 9 and x <= 10
        assert y == 10

    def rightward():
        ctx = chunk_manager.intersect_diagonal_vertical_wall(11, 9, 10, 11)
        assert ctx
        block, direction, x, y, _ = ctx
        assert block == Block.STONE.value
        assert direction == IntersectionDirection.LEFT
        assert x >= 11 and x <= 12
        assert y < 12 and y >= 11

    leftward()
    rightward()


if __name__ == "__main__":
    test_ortho_y()
    test_ortho_x()
    test_diagonal_horizontal_wall()
    test_diagonal_vertical_wall()
