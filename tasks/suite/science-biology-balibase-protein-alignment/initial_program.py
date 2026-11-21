"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Progressive multiple alignment: pairwise identity guide tree plus profile Needleman-Wunsch.'''

GAP_OPEN = -11.0
GAP_EXTEND = -1.0
MATCH = 2.0
MISMATCH = -1.0


def _score(a, b):
    if a == b:
        return MATCH
    return MISMATCH


def _profile_align(left, right):
    '''Align two blocks of equal-length rows with affine gaps on average column scores.'''
    n, m = len(left[0]), len(right[0])
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0] = GAP_OPEN + (i - 1) * GAP_EXTEND
        move[i][0] = 1
    for j in range(1, m + 1):
        best[0][j] = GAP_OPEN + (j - 1) * GAP_EXTEND
        move[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            column = 0.0
            pairs = 0
            for a in left:
                for b in right:
                    if a[i - 1] != "-" and b[j - 1] != "-":
                        column += _score(a[i - 1], b[j - 1])
                        pairs += 1
            diagonal = best[i - 1][j - 1] + (column / pairs if pairs else 0.0)
            up = best[i - 1][j] + (GAP_EXTEND if move[i - 1][j] == 1 else GAP_OPEN)
            leftward = best[i][j - 1] + (GAP_EXTEND if move[i][j - 1] == 2 else GAP_OPEN)
            options = ((diagonal, 0), (up, 1), (leftward, 2))
            value, direction = max(options)
            best[i][j], move[i][j] = value, direction
    i, j = n, m
    out_left = ["" for _ in left]
    out_right = ["" for _ in right]
    while i > 0 or j > 0:
        direction = move[i][j] if (i > 0 and j > 0) else (1 if j == 0 else 2)
        if direction == 0:
            for k, row in enumerate(left):
                out_left[k] = row[i - 1] + out_left[k]
            for k, row in enumerate(right):
                out_right[k] = row[j - 1] + out_right[k]
            i -= 1
            j -= 1
        elif direction == 1:
            for k, row in enumerate(left):
                out_left[k] = row[i - 1] + out_left[k]
            for k in range(len(right)):
                out_right[k] = "-" + out_right[k]
            i -= 1
        else:
            for k in range(len(left)):
                out_left[k] = "-" + out_left[k]
            for k, row in enumerate(right):
                out_right[k] = row[j - 1] + out_right[k]
            j -= 1
    return out_left + out_right


def _identity(a, b):
    rows = _profile_align([a], [b])
    same = sum(1 for x, y in zip(rows[0], rows[1]) if x == y and x != "-")
    return same / max(1, min(len(a), len(b)))


def align(sequences):
    blocks = [([i], [sequences[i]]) for i in range(len(sequences))]
    while len(blocks) > 1:
        best = None
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                value = _identity(blocks[a][1][0], blocks[b][1][0])
                if best is None or value > best[0]:
                    best = (value, a, b)
        _, a, b = best
        merged_rows = _profile_align(blocks[a][1], blocks[b][1])
        merged = (blocks[a][0] + blocks[b][0], merged_rows)
        blocks = [block for k, block in enumerate(blocks) if k not in (a, b)] + [merged]
    order, rows = blocks[0]
    out = [""] * len(sequences)
    for index, row in zip(order, rows):
        out[index] = row
    return out
# EVOLVE-BLOCK-END
