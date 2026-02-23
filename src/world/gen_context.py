from pyfastnoiselite.pyfastnoiselite import (  # ty:ignore[unresolved-import]
    FastNoiseLite,
    FractalType,
    NoiseType,
)

import numpy as np
import numpy.typing as npt

class WorldGenContext:
    seed: int
    generator: np.random.RandomState
    noise: FastNoiseLite
    base_height: int = 256

    def __init__(self, seed: int):
        self.seed = seed
        self.generator = np.random.RandomState(seed=seed)
        print(seed)

        self.noise = FastNoiseLite(seed=self.seed)

        # Configure base noise settings
        self.noise.noise_type = NoiseType.NoiseType_OpenSimplex2
        self.noise.fractal_type = FractalType.FractalType_FBm

    def fractal_noise_1d(
        self,
        x_coords: npt.NDArray[np.int64],
        chunk_x: int,
        width: int,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        scale=0.02,
    ):
        """Generate 1D fractal noise using FastNoiseLite built-in fractal"""
        # Configure fractal parameters
        
        self.noise.noise_type = NoiseType.NoiseType_OpenSimplex2
        self.noise.fractal_octaves = octaves
        self.noise.fractal_lacunarity = lacunarity
        self.noise.fractal_gain = persistence  # gain = persistence
        self.noise.frequency = scale

        # Calculate world coordinates
        x_world = (x_coords + chunk_x * width).astype(np.float32)
        y_world = np.zeros_like(x_world, dtype=np.float32)

        # Stack into shape (2, N) - this is what gen_from_coords expects
        coords = np.stack([x_world, y_world], axis=0)

        # Generate noise
        return self.noise.gen_from_coords(coords)
