from src.world import ChunkManager, WorldGenContext


def test_conversion(cm: ChunkManager):
    for x in range(-1000, 1000):
        for y in range(-1000, 1000):
            coords = chunk_manager._world_to_chunk(x, y)
            if coords is None:
                continue
            chunk_x, local_x, local_y = coords

            assert local_x < 32
            assert local_x >= 0
            assert local_y < cm.height
            assert local_y >= 0


if __name__ == "__main__":
    gen_ctx = WorldGenContext(0)
    chunk_manager = ChunkManager(gen_ctx)

    test_conversion(chunk_manager)
