"""Reject incomplete games before ranks can enter formal statistics."""

def validate_game_rows(rows, *, expected_seeds=None):
    if not rows:
        raise ValueError("benchmark contains no games")
    seen = set()
    for row in rows:
        seed = row["seed"]
        if seed in seen:
            raise ValueError(f"duplicate game seed: {seed}")
        seen.add(seed)
        if row.get("error") is not None or row.get("completed") is not True:
            raise ValueError(f"incomplete/error game: {seed}")
        if row.get("illegal_actions", 0) != 0:
            raise ValueError(f"illegal action in game: {seed}")
        if sorted(row.get("final_ranks", [])) != [0, 1, 2, 3]:
            raise ValueError(f"invalid final ranks in game: {seed}")
        if "seat" in row and row["seat"] not in range(4):
            raise ValueError(f"invalid candidate seat: {seed}")
    if expected_seeds is not None and seen != set(expected_seeds):
        raise ValueError("game seed coverage does not match registration")
