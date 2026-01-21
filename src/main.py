from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

Chunk: TypeAlias = NDArray[np.int8]

class ChunkManager():
    width: int
    height: int
    chunks: dict[int, Chunk]
    min_id: int
    max_id: int
    
    def __init__(self, width: int, height: int):
        self.height = height
        self.width = width

    def generate_chunk(self) -> Chunk:
        chunk: Chunk = np.zeros(shape=(self.width, self.height), dtype=np.int8)
        return chunk

    def generate_chunk_left(self) -> int:
        self.min_id -= 1
        id = self.min_id
        self.chunks[id] = self.generate_chunk()
        return id

    def generate_chunk_right(self) -> int:
        self.min_id -= 1
        id = self.min_id
        self.chunks[id] = self.generate_chunk()
        return id

    def get_chunk(self, id: int) -> Chunk:
        if id in self.chunks.keys:a     
        
    
