import itertools
from collections.abc import Callable, Generator, Iterable
import numpy as np
import numpy.typing as npt
import heapq
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    x: int
    y: int
    vy: int


def astar(
    start: State,
    goal: State,
    get_neighbours: Callable[[State], Iterable[tuple[State, float]]],
    heuristic: Callable[[State], float],
) -> list[State] | None:
    counter = itertools.count()  # unique, always-increasing tiebreaker  # noqa: F821

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

        neighbours = list(get_neighbours(node))

        for neighbour, cost in neighbours:
            if neighbour not in visited:
                new_g: float = g + cost
                new_f: float = new_g + heuristic(neighbour)
                heapq.heappush(
                    open_set,
                    (new_f, new_g, next(counter), neighbour, path + [neighbour]),
                )

    return None

def get_neighbours(arr: npt.NDArray[np.bool_], state: State) -> Generator[tuple[State, float]]:
    dim = arr.shape
    for dx, dy in itertools.product((-1, 0, 1), repeat=2):
        if dx == 0 and dy == 0:
            continue
        x = state.x + dx
        y = state.y + dy
        if x < 0 or x >= dim[0]:
            continue
        if y < 0 or y >= dim[1]:
            continue
        if arr[x, y]:
            continue
        yield (State(x, y, 0), 1)
        
def heuristic(s1:State, s2:State) -> float:
    return abs(s2.x - s1.x) + abs(s2.y - s1.y)

arr = np.zeros((20, 20), dtype=np.bool_)
arr[10, 5:15] = True

print(arr)

start = State(0, 0, 0)
end = State(19, 19, 0)
res = astar(start=start, goal=end, get_neighbours=lambda s: get_neighbours(arr, s), heuristic=lambda s: heuristic(s, end))
res: list[State] = res if res is not None else []

debug = [["X" if arr[x, y] else " " for x in range(0, arr.shape[0])] for y in range(0, arr.shape[1])]



for state in res:
    debug[state.y][state.x] = "#"

for row in debug:
    print(*row, sep=" ")
