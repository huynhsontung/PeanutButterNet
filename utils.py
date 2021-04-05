def extract_kernel_sizes(op_name: str) -> list[tuple[int, int]]:
    elems = op_name.split("_")
    candidates = [x.lower().split("x") for x in elems if "x" in x.lower()]
    res = []
    for c in candidates:
        if all([x.isnumeric() for x in c]):
            res.append((int(c[0]), int(c[1])))
    return res