import numpy as np


def extract_kernel_sizes(op_name: str) -> list[tuple[int, int]]:
    elems = op_name.split("_")
    candidates = [x.lower().split("x") for x in elems if "x" in x.lower()]
    res = []
    for c in candidates:
        if all([x.isnumeric() for x in c]):
            res.append((int(c[0]), int(c[1])))
    return res


def generate_recursive_indices(
        initial_size: int, num_group: int, size: int, seed=12345) -> list[tuple]:
    """ Generate indices from initial indices that allows later entries to use past indices

    For example:
    generate_recursive_indices(2, 2, 3)
    Initial size = 2
    Possible indices for:
        First entry: [0,1]
        Second entry: [0,1,2]
        Third entry: [0,1,2,3]

    Output: [(0,1), (2,1), (3,0)]
    """

    rng = np.random.default_rng(seed)
    possible_inputs = list(range(initial_size))
    out: list[tuple[int, int]] = []
    for counter in range(size):
        out.append(tuple(rng.choice(possible_inputs, size=num_group)))
        possible_inputs.append(counter + initial_size)

    return out
