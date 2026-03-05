import heapq
import itertools
from dataclasses import dataclass
from typing import Callable, Generator


@dataclass(frozen=True)
class State:
    x: int
    y: int
    vy: int


def astar(
    start: State,
    goal: State,
    get_neighbours: Callable[[State], Generator[tuple[State, float], None, None]],
    heuristic: Callable[[State], float],
) -> list[State] | None:
    counter = itertools.count()  # unique, always-increasing tiebreaker

    open_set: list[tuple[float, float, int, State, list[State]]] = [
        (heuristic(start), 0.0, next(counter), start, [start])
    ]
    visited: set[State] = set()

    while open_set:
        f, g, _, node, path = heapq.heappop(open_set)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for neighbour, cost in get_neighbours(node):
            if neighbour not in visited:
                new_g: float = g + cost
                new_f: float = new_g + heuristic(neighbour)
                heapq.heappush(
                    open_set,
                    (new_f, new_g, next(counter), neighbour, path + [neighbour]),
                )

    return None
