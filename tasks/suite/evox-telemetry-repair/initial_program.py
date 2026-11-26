"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def run_repair(telemetry, topology):
    """Seed with the behaviour of the ADRS initial program: no repair yet.

    Every counter is reported as repaired to 0 with full confidence and the
    status is passed through; an evolved program should use the topology
    (my rx should match the peer's tx, and vice versa) to detect and repair
    dropped or scaled counters and to calibrate its confidence.
    """
    result = {}
    for interface_id, data in telemetry.items():
        status = data.get("interface_status", "unknown")
        result[interface_id] = {
            "rx_rate": (0.0, 0.0, 1.0),
            "tx_rate": (0.0, 0.0, 1.0),
            "interface_status": (status, status, 1.0),
            "connected_to": data.get("connected_to"),
            "local_router": data.get("local_router"),
            "remote_router": data.get("remote_router"),
        }
    return result
# EVOLVE-BLOCK-END
