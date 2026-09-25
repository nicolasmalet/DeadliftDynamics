from config import t


def differentiate(f: list[float]) -> list[float]:
    """Computes the discrete derivative of a list of values over time t."""
    return [(f[i + 1] - f[i]) / t for i in range(len(f) - 1)]


def integrate(f: list[float]) -> list[float]:
    """Computes the discrete integral of a list of values over time t using the Euler method."""
    F = [0.0]
    for i in range(len(f)):
        F.append(F[-1] + t * f[i])
    return F
