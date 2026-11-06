from pathlib import Path
from detect_subidones_spectral import analyze_track_subidones

f1 = Path('/mnt/ssd/MúSiCa/TeMaZoS/Sequential One - Imagination (Main DJ Version).flac')
f2 = Path('/mnt/ssd/MúSiCa/TeMaZoS/Gaetano Parisio - Outdated (A1).flac')

for f in [f1, f2]:
    info = analyze_track_subidones(f)
    print(f"\n========================================================")
    print(f"ANÁLISIS ESPECTRAL DE: {info['name']}")
    print(f"BPM: {info['bpm']:.2f} | Duración: {int(info['duration']//60):02d}:{int(info['duration']%60):02d} | Frases de 32: {len(info['phrases'])}")
    print(f"--- Subidones detectados (Drops tras parón) ---")
    if not info['drops']:
        print("  (Pista continua / techno lineal sin grandes parones de graves)")
    else:
        for d in info['drops']:
            print(f"  🚀 SUBIDÓN en [{d['time_str']}] (Frase {d['phrase']}, Beat {d['beat']}): Graves suben de {d['low_prev']:.3f} a {d['low_curr']:.3f} (x{d['low_ratio']:.2f}) [Score={d['score']:.2f}]")
            
    print(f"--- Desglose por Frases ---")
    for ph in info['phrases']:
        print(f"  Frase {ph['phrase']:02d} [{ph['time_str']}]: Low={ph['low']:.3f} | Mid={ph['mid']:.3f} | High={ph['high']:.3f} | Tot={ph['tot']:.3f}")
