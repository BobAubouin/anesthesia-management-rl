def apply_safety_contraints(bis: float) -> float:
    """Penalise unsafe BIS values"""
    if bis < 30 or bis > 70:
        return -10
    return 0.0