"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


def build_model(data):
    n, width, length = data["n_containers"], data["width"], data["length"]
    klass, separation = data["container_class"], data["separation"]
    west = cp.intvar(0, data["deck_width"], shape=n, name="west")
    east = cp.intvar(0, data["deck_width"], shape=n, name="east")
    north = cp.intvar(0, data["deck_length"], shape=n, name="north")
    south = cp.intvar(0, data["deck_length"], shape=n, name="south")
    model = cp.Model()
    for c in range(n):
        model += (((east[c] - west[c] == width[c]) & (south[c] - north[c] == length[c])) |
                  ((east[c] - west[c] == length[c]) & (south[c] - north[c] == width[c])))
    for c1 in range(n):
        for c2 in range(n):
            if c1 == c2:
                continue
            model += (west[c1] != west[c2]) | (north[c1] != north[c2])
            gap_x = cp.max([west[c1], west[c2]]) - cp.min([east[c1], east[c2]])
            gap_y = cp.max([north[c1], north[c2]]) - cp.min([south[c1], south[c2]])
            model += (gap_x >= 0) | (gap_y >= 0)
            sep = separation[klass[c1]][klass[c2]]
            model += (gap_x >= sep) | (gap_y >= sep)

    def extractor():
        return {"west": west.value().tolist(), "east": east.value().tolist(),
                "north": north.value().tolist(), "south": south.value().tolist()}
    return model, extractor
# EVOLVE-BLOCK-END
