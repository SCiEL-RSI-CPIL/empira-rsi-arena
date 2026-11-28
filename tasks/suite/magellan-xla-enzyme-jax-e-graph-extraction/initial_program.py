"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math


def extract(egraph):
    """Bottom-up extraction (as extraction-gym's bottom-up extractor): repeat passes over the e-classes;
    a node's cost is its own cost plus the current best cost of each child e-class, and every e-class
    keeps its cheapest node, until nothing changes. Optimal for tree cost, not for DAG cost."""
    nodes = egraph["nodes"]
    classes = {}
    for node_id, node in nodes.items():
        classes.setdefault(node["eclass"], []).append(node_id)
    best, choice = {}, {}
    changed = True
    while changed:
        changed = False
        for class_id, members in classes.items():
            for node_id in members:
                node = nodes[node_id]
                children = 0.0
                for child in node["children"]:
                    children += best.get(nodes[child]["eclass"], math.inf)
                total = node["cost"] + children
                if total < best.get(class_id, math.inf):
                    best[class_id], choice[class_id] = total, node_id
                    changed = True
    return choice
# EVOLVE-BLOCK-END
