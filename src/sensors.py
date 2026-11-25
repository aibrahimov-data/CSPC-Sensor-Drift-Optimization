def sensor_features(sensor_ids):
    """Sensor IDs (1-16) -> their 8 feature columns each."""
    cols = []
    for s in sensor_ids:
        cols += [f"f{8 * (s - 1) + i:03d}" for i in range(1, 9)]
    return cols