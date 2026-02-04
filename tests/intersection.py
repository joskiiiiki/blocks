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
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.DOWN
        assert ctx.intersect[0] == 1
        assert ctx.intersect[1] == 11

    def upwards():
        ctx = chunk_manager.intersect_ortho_line_y(1, 9, 1, 10)
        assert ctx
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.UP
        assert ctx.intersect[0] == 1
        assert ctx.intersect[1] == 10

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
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.RIGHT
        assert ctx.intersect[0] == 10
        assert ctx.intersect[1] == 0

    def rightward():
        ctx = chunk_manager.intersect_ortho_line_x(11, 0, 10, 0)
        assert ctx
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.LEFT
        assert ctx.intersect[0] == 11
        assert ctx.intersect[1] == 0

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
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.DOWN
        assert ctx.intersect[1] == 11

    def upward():
        ctx = chunk_manager.intersect_diagonal_horizontal_wall(9, 9, 10, 11)
        assert ctx
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.UP
        assert ctx.intersect[1] == 10

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
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.RIGHT
        assert ctx.intersect[0] == 10

    def rightward():
        ctx = chunk_manager.intersect_diagonal_vertical_wall(11, 9, 10, 11)
        assert ctx
        assert ctx.block == Block.STONE.value
        assert ctx.direction == IntersectionDirection.LEFT
        assert ctx.start[0] == 11

    leftward()
    rightward()


if __name__ == "__main__":
    test_ortho_y()
    test_ortho_x()
    test_diagonal_horizontal_wall()
    test_diagonal_vertical_wall()
