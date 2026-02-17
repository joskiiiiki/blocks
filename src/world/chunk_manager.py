import json
import math
import os
import pathlib
import queue
import threading
from typing import Iterable, Optional

import numpy as np

from src.blocks import Block
from src.world.chunk import Chunk, ChunkStatus
from src.world.utils import smooth_caves_with_neighbors, smooth_mask_ce


class ChunkManager:
    """Thread-safe chunk manager with coordinated generation and decoration"""

    def __init__(
        self,
        gen_ctx,  # WorldGenContext
        path: Optional[os.PathLike] = None,
        width: int = 32,
        height: int = 512,
        region_size: int = 32,
    ):
        self.height = height
        self.width = width
        self.region_size = region_size
        self.gen_ctx = gen_ctx

        # Thread-safe cache with RWLock pattern
        self._chunk_cache: dict[int, Chunk] = {}
        self._cache_lock = threading.RLock()  # Reentrant lock for nested calls

        # Track chunks being processed to avoid duplicate work
        self._generating: set[int] = set()
        self._decorating: set[int] = set()

        # Condition variables for coordination
        self._generation_cv = threading.Condition(self._cache_lock)

        # Persistent storage
        self.world_dir = pathlib.Path(path) if path else None
        if self.world_dir:
            self.world_dir.mkdir(exist_ok=True, parents=True)

        # Worker threads
        self._running = False
        self._generation_queue: queue.Queue[int] = queue.Queue()
        self._save_queue: queue.Queue[tuple[int, Chunk]] = queue.Queue()
        self._decoration_pending: set[int] = set()  # Chunks waiting for decoration

        self._generation_thread: Optional[threading.Thread] = None
        self._save_thread: Optional[threading.Thread] = None

    def start(self):
        """Start background worker threads"""
        if self._running:
            return

        self._running = True
        if self.world_dir:
            self._generation_thread = threading.Thread(
                target=self._generation_worker, daemon=False, name="ChunkGenerator"
            )
            self._save_thread = threading.Thread(
                target=self._save_worker, daemon=False, name="ChunkSaver"
            )
            self._generation_thread.start()
            self._save_thread.start()

    def shutdown(self):
        """Gracefully shutdown all workers"""
        print("Shutting down ChunkManager...")

        # Save all cached chunks
        with self._cache_lock:
            chunks_to_save = list(self._chunk_cache.keys())

        print(f"Queueing {len(chunks_to_save)} chunks for saving")
        for chunk_x in chunks_to_save:
            self._queue_save(chunk_x)

        # Wait for save queue to empty
        self._save_queue.join()

        # Signal threads to stop
        self._running = False

        # Wake up any waiting threads
        with self._generation_cv:
            self._generation_cv.notify_all()

        # Wait for threads with timeout
        if self._generation_thread:
            self._generation_thread.join(timeout=10)
            if self._generation_thread.is_alive():
                print("WARNING: Generation thread did not terminate")

        if self._save_thread:
            self._save_thread.join(timeout=10)
            if self._save_thread.is_alive():
                print("WARNING: Save thread did not terminate")

        print("ChunkManager shutdown complete")

    # ==================== WORKER THREADS ====================

    def _generation_worker(self):
        """Background thread that generates and decorates chunks"""
        while self._running:
            try:
                chunk_x = self._generation_queue.get(timeout=0.5)
                self._process_chunk_generation(chunk_x)
                self._generation_queue.task_done()
            except queue.Empty:
                # Check if any pending decorations can proceed
                self._try_pending_decorations()
            except Exception as e:
                print(f"Error in generation worker: {e}")
                import traceback

                traceback.print_exc()

    def _save_worker(self):
        """Background thread that saves chunks to disk"""
        while self._running:
            try:
                chunk_x, chunk = self._save_queue.get(timeout=0.5)
                self._write_chunk_to_disk(chunk_x, chunk)
                self._save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving chunk {chunk_x}: {e}")
                self._save_queue.task_done()

        # Process remaining saves
        while not self._save_queue.empty():
            try:
                chunk_x, chunk = self._save_queue.get_nowait()
                self._write_chunk_to_disk(chunk_x, chunk)
                self._save_queue.task_done()
            except Exception as e:
                print(f"Error in final save: {e}")

    # ==================== GENERATION PIPELINE ====================

    def _process_chunk_generation(self, chunk_x: int):
        """Generate terrain and attempt decoration (thread-safe)"""

        # Check if already exists or is being processed
        with self._cache_lock:
            if chunk_x in self._chunk_cache:
                return  # Already loaded
            if chunk_x in self._generating:
                return  # Already being generated

            # Mark as being generated
            self._generating.add(chunk_x)

        try:
            # Check disk first
            chunk = self._load_from_disk(chunk_x)

            if chunk is None:
                # Generate new chunk
                chunk = Chunk(chunk_x=chunk_x)
                chunk.generate(self.gen_ctx)  # This is NOT thread-safe with neighbors

            # Add to cache atomically
            with self._cache_lock:
                self._chunk_cache[chunk_x] = chunk
                self._generating.remove(chunk_x)

                # If not decorated, mark as pending
                if chunk.status != ChunkStatus.DECORATED:
                    self._decoration_pending.add(chunk_x)

                # Notify anyone waiting for this chunk
                self._generation_cv.notify_all()

            self._smooth_chunk_caves(chunk_x)

            # Try to decorate this chunk and trigger neighbor decoration
            self._attempt_decoration(chunk_x)

            # Also check if neighbors can now be decorated
            for neighbor_x in [chunk_x - 1, chunk_x + 1]:
                with self._cache_lock:
                    if neighbor_x in self._decoration_pending:
                        self._attempt_decoration(neighbor_x)

        except Exception as e:
            print(f"Error generating chunk {chunk_x}: {e}")
            with self._cache_lock:
                self._generating.discard(chunk_x)
                self._generation_cv.notify_all()
            raise e

    def _smooth_chunk_caves(
        self,
        chunk_x: int,
        iterations: int = 3,
        death_limit: int = 3,
        birth_limit: int = 4,
    ):
        """
        Smooth caves in a chunk, considering neighbors if available.
        Called after terrain generation.
        """
        with self._cache_lock:
            chunk = self._chunk_cache.get(chunk_x)
            if chunk is None:
                return

            left_chunk = self._chunk_cache.get(chunk_x - 1)
            right_chunk = self._chunk_cache.get(chunk_x + 1)

        # If both neighbors exist, do multi-chunk smoothing
        if left_chunk is not None and right_chunk is not None:
            for _ in range(iterations):
                changed = smooth_caves_with_neighbors(
                    chunk,
                    left_chunk,
                    right_chunk,
                    death_limit=death_limit,
                    birth_limit=birth_limit,
                )
                if not changed:
                    break  # Converged early
        else:
            # Neighbors not ready yet - do single-chunk smoothing
            # (will be re-smoothed when neighbors are generated)
            self._smooth_chunk_caves_solo(chunk, iterations, death_limit, birth_limit)

    def _smooth_chunk_caves_solo(
        self,
        chunk: Chunk,
        iterations: int = 3,
        death_limit: int = 3,
        birth_limit: int = 4,
        min_y: int = 20,
        max_y: int = 250,
    ):
        """
        Smooth caves in a single chunk (fallback when neighbors unavailable).
        Treats edges as solid.
        """
        region = chunk.blocks[:, min_y : max_y + 1]
        cave_mask = region == Block.AIR.value

        for _ in range(iterations):
            changed = smooth_mask_ce(cave_mask, death_limit, birth_limit)
            if not changed:
                break

        # Apply changes back
        region[cave_mask] = Block.AIR.value
        region[~cave_mask & (region == Block.AIR.value)] = Block.STONE.value

    def _attempt_decoration(self, chunk_x: int):
        """Try to decorate a chunk if neighbors are ready"""

        with self._cache_lock:
            # Check if chunk exists and needs decoration
            chunk = self._chunk_cache.get(chunk_x)
            if chunk is None or chunk.status == ChunkStatus.DECORATED:
                self._decoration_pending.discard(chunk_x)
                return

            if chunk.status != ChunkStatus.TERRAIN_GENERATED:
                return

            # Check if already being decorated
            if chunk_x in self._decorating:
                return

            # Get neighbor chunks
            left_chunk = self._chunk_cache.get(chunk_x - 1)
            right_chunk = self._chunk_cache.get(chunk_x + 1)

            # Check if neighbors are ready (terrain generated or decorated)
            left_ready = left_chunk is not None and left_chunk.status in [
                ChunkStatus.TERRAIN_GENERATED,
                ChunkStatus.DECORATED,
            ]
            right_ready = right_chunk is not None and right_chunk.status in [
                ChunkStatus.TERRAIN_GENERATED,
                ChunkStatus.DECORATED,
            ]

            if not (left_ready and right_ready):
                # Can't decorate yet - neighbors not ready
                return

            # Mark as being decorated
            self._decorating.add(chunk_x)

        # Decorate outside the lock (the chunk.decorate method needs access to neighbors)
        try:
            left_status = left_chunk.status if left_chunk else None
            right_status = right_chunk.status if right_chunk else None

            success = chunk.decorate(left_status, right_status, self.gen_ctx)

            with self._cache_lock:
                self._decorating.remove(chunk_x)
                if success:
                    self._decoration_pending.discard(chunk_x)
                    # Notify threads waiting for decoration
                    self._generation_cv.notify_all()

        except Exception as e:
            print(f"Error decorating chunk {chunk_x}: {e}")
            with self._cache_lock:
                self._decorating.remove(chunk_x)

    def _try_pending_decorations(self):
        """Attempt to decorate all pending chunks"""
        with self._cache_lock:
            pending = list(self._decoration_pending)

        for chunk_x in pending:
            self._attempt_decoration(chunk_x)

    # ==================== PUBLIC API ====================

    def queue_generation(self, chunk_x: int):
        """Request generation of a chunk (non-blocking)"""
        self._generation_queue.put(chunk_x)

    def load_chunk(self, chunk_x: int, generate_if_not_exists: bool = True):
        """Load a chunk from cache or disk, optionally generating if missing"""
        with self._cache_lock:
            if chunk_x in self._chunk_cache:
                return  # Already loaded

        # Try loading from disk
        chunk = self._load_from_disk(chunk_x)
        if chunk is not None:
            with self._cache_lock:
                self._chunk_cache[chunk_x] = chunk
                if chunk.status != ChunkStatus.DECORATED:
                    self._decoration_pending.add(chunk_x)
                    self._attempt_decoration(chunk_x)
            return

        # Generate if requested
        if generate_if_not_exists:
            self.queue_generation(chunk_x)

    def load_chunks_only(self, chunks: Iterable[int]):
        """Load only the specified chunks, unloading others"""
        chunks_set = set(chunks)

        with self._cache_lock:
            # Find chunks to unload
            to_unload = [x for x in self._chunk_cache.keys() if x not in chunks_set]

        # Unload chunks
        for chunk_x in to_unload:
            self.unload_chunk(chunk_x)

        # Load desired chunks
        for chunk_x in chunks_set:
            self.load_chunk(chunk_x)

    def unload_chunk(self, chunk_x: int):
        """Unload a chunk from cache, saving it first"""
        with self._cache_lock:
            chunk = self._chunk_cache.pop(chunk_x, None)
            self._decoration_pending.discard(chunk_x)

        if chunk is not None:
            self._queue_save(chunk_x, chunk)

    def get_chunk_from_cache(self, chunk_x: int) -> Chunk | None:
        """Thread-safe cache access"""
        with self._cache_lock:
            return self._chunk_cache.get(chunk_x)

    def get_chunk(self, chunk_x: int) -> Chunk | None:
        """Get chunk from cache or disk (does not generate)"""
        chunk = self.get_chunk_from_cache(chunk_x)
        if chunk:
            return chunk

        return self._load_from_disk(chunk_x)

    # ==================== BLOCK OPERATIONS ====================

    def get_block(self, x: float, y: float):
        """Thread-safe block access"""
        coords = self._world_to_chunk(x, y)
        if not coords:
            return None

        chunk_x, local_x, local_y = coords

        with self._cache_lock:
            chunk = self._chunk_cache.get(chunk_x)
            if chunk is None:
                return None

            xn = int(np.floor(local_x))
            yn = int(np.floor(local_y))
            return chunk.blocks[xn, yn]

    def set_block(self, x: float, y: float, block, allow_replace: bool = False) -> bool:
        """Thread-safe block modification"""
        coords = self._world_to_chunk(x, y)
        if coords is None:
            return False

        chunk_x, local_x, local_y = coords

        with self._cache_lock:
            chunk = self._chunk_cache.get(chunk_x)
            if chunk is None:
                return False

            xn = int(np.floor(local_x))
            yn = int(np.floor(local_y))

            # Check if replacement is allowed
            if not allow_replace and chunk.blocks[xn, yn] != 0:  # Assuming 0 = AIR
                return False

            chunk.blocks[xn, yn] = block
            return True

    def destroy_block(self, x: float, y: float):
        """Thread-safe block destruction"""
        coords = self._world_to_chunk(x, y)
        if coords is None:
            return None

        chunk_x, local_x, local_y = coords

        with self._cache_lock:
            chunk = self._chunk_cache.get(chunk_x)
            if chunk is None:
                return None

            xn = int(np.floor(local_x))
            yn = int(np.floor(local_y))

            old_block = chunk.blocks[xn, yn]
            chunk.blocks[xn, yn] = 0  # AIR
            return old_block

    # ==================== DISK I/O ====================

    def _queue_save(self, chunk_x: int, chunk: Optional[Chunk] = None):
        """Queue a chunk for saving"""
        if chunk is None:
            with self._cache_lock:
                chunk = self._chunk_cache.get(chunk_x)

        if chunk is not None:
            self._save_queue.put((chunk_x, chunk.copy()))

    def _write_chunk_to_disk(self, chunk_x: int, chunk: Chunk):
        """Write chunk to disk (called by save worker)"""
        if not self.world_dir:
            return

        region = chunk_x // self.region_size
        region_path = self.world_dir / str(region)
        region_path.mkdir(parents=True, exist_ok=True)

        chunk_path = region_path / f"{chunk_x}.npy"
        data_path = region_path / f"{chunk_x}.json"

        # Atomic write
        tmp_chunk = chunk_path.with_suffix(".tmp")
        with tmp_chunk.open("wb") as f:
            np.save(f, chunk.blocks, allow_pickle=False)
        tmp_chunk.replace(chunk_path)

        tmp_data = data_path.with_suffix(".tmp")
        with tmp_data.open("w") as f:
            json.dump(chunk.data(), f)
        tmp_data.replace(data_path)

    def _load_from_disk(self, chunk_x: int) -> Chunk | None:
        """Load chunk from disk"""
        if not self.world_dir:
            return None

        region = chunk_x // self.region_size
        chunk_path = self.world_dir / str(region) / f"{chunk_x}.npy"
        data_path = self.world_dir / str(region) / f"{chunk_x}.json"

        if not (chunk_path.exists() and data_path.exists()):
            return None

        try:
            blocks = np.load(chunk_path, allow_pickle=False)
            with data_path.open() as f:
                data = json.load(f)
            return Chunk.from_data(chunk_x=chunk_x, blocks=blocks, data=data)
        except Exception as e:
            print(f"Error loading chunk {chunk_x}: {e}")
            return None

    # ==================== HELPERS ====================

    def _world_to_chunk(self, x: float, y: float) -> tuple[int, float, float] | None:
        """Convert world coordinates to chunk coordinates"""
        chunk_x = math.floor(x) // self.width
        local_x = x % self.width

        assert local_x < 32
        assert local_x >= 0

        if y < 0 or y >= self.height:
            return None

        return chunk_x, local_x, y

    def get_chunk_x(self, x: float) -> int:
        """Get chunk X coordinate from world X"""
        return int(x) // self.width
