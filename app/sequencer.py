import re
from typing import List

try:
    from app.analyzer import TrackProfile
except ModuleNotFoundError:
    from analyzer import TrackProfile


def parse_camelot(camelot_str: str):
    """Devuelve (número, letra) de una clave Camelot como '8A'."""
    match = re.match(r"(\d+)([AB])", camelot_str)
    if not match:
        return (8, "A")
    return int(match.group(1)), match.group(2)


def camelot_distance(c1: str, c2: str) -> float:
    """Calcula la penalización armónica entre dos tonalidades Camelot."""
    n1, l1 = parse_camelot(c1)
    n2, l2 = parse_camelot(c2)

    # Distancia en el reloj (1..12)
    clock_diff = abs(n1 - n2)
    clock_dist = min(clock_diff, 12 - clock_diff)

    if l1 == l2:
        # Misma modalidad (ambas A o ambas B)
        return float(clock_dist)
    else:
        # Cambio de modalidad (ej: 8A -> 8B es relativo directo, coste bajo)
        if clock_dist == 0:
            return 0.5
        return clock_dist + 1.5


def optimize_track_sequence(tracks: List[TrackProfile]) -> List[TrackProfile]:
    """Ordena las pistas minimizando saltos bruscos de BPM y disonancia armónica."""
    if len(tracks) <= 2:
        return sorted(tracks, key=lambda t: t.bpm)

    remaining = tracks.copy()
    # Empezar por el tema de BPM más bajo (arranque progresivo típico de un set de DJ)
    remaining.sort(key=lambda t: t.bpm)
    ordered = [remaining.pop(0)]

    while remaining:
        current = ordered[-1]
        best_candidate = None
        best_score = float("inf")
        best_idx = -1

        for i, candidate in enumerate(remaining):
            # Penalización por BPM (procurando aumentos suaves o cambios pequeños)
            bpm_diff = candidate.bpm - current.bpm
            # Favorecer que el BPM aumente ligeramente o se mantenga igual
            bpm_penalty = abs(bpm_diff) if bpm_diff >= 0 else abs(bpm_diff) * 1.5
            # Penalización armónica
            arm_penalty = camelot_distance(current.camelot, candidate.camelot)

            score = (bpm_penalty * 1.0) + (arm_penalty * 4.0)
            if score < best_score:
                best_score = score
                best_candidate = candidate
                best_idx = i

        ordered.append(remaining.pop(best_idx))

    return ordered
