#!/usr/bin/env python3
"""
Position du nodal point (entrance pupil) par la méthode de parallaxe.

Contrôle croisé du nodal offset calculé dans le Lens File Unreal / REtracker :
ce script donne la position de l'entrance pupil LE LONG DE L'AXE OPTIQUE
uniquement (il ne connaît pas la pose du tracker).

Usage :
    python nodal_point_parallax.py mesures.csv
    python nodal_point_parallax.py mesures.csv --fx 2950 --z-near 0.8 --z-far 15

Format du CSV (séparateur virgule, en-tête obligatoire) :
    slide_mm,pan_deg,dx_px
    0,12,-41.5
    0,-12,40.8
    10,12,-22.0
    10,-12,21.7
    ...
  - slide_mm : graduation de la glissière nodale
  - pan_deg  : angle de pan, POSITIF = caméra tournée vers la DROITE
  - dx_px    : x_cible_proche - x_cible_lointaine, en pixels natifs

Protocole :
  1. Cible proche (~0.5-1 m) et cible lointaine (>10 m) alignées au centre
     de l'image à pan = 0.
  2. Pour chaque position de glissière, mesurer dx à +θ et -θ (θ ≈ 10-15°).
  3. Au moins 2 positions de glissière (idéalement 3-5 encadrant le point)
     -> graduation où la parallaxe s'annule = entrance pupil sur l'axe de
        rotation. Mesurer ensuite au réglet axe <-> repère plan capteur.
  4. Avec une seule position : fournir --fx (px, calib de ce repo),
     --z-near et --z-far (m, depuis l'entrance pupil) pour estimer
     directement la distance pupille <-> axe de rotation.

Modèle (vue de dessus, pupille à d mm DEVANT l'axe de rotation) :
    dx(θ) ≈ a - fx · d · (1/Zn - 1/Zf) · sin θ / cos² θ
'a' absorbe le défaut d'alignement initial des deux cibles.
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

# --- seuil de décision : parallaxe résiduelle (px) à ±θ considérée nulle
SEUIL_PARALLAXE_PX = 0.5


def load_measures(path):
    by_slide = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_slide[float(row["slide_mm"])].append(
                (float(row["pan_deg"]), float(row["dx_px"])))
    return dict(sorted(by_slide.items()))


def fit_parallax(samples):
    """Ajuste dx = a + b·g(θ), g = sinθ/cos²θ. Retourne (b, parallaxe px à θmax)."""
    theta = np.radians([s[0] for s in samples])
    dx = np.array([s[1] for s in samples])
    if np.ptp(theta) == 0:
        sys.exit("Il faut au moins deux angles de pan différents par position (ex. +θ et -θ).")
    g = np.sin(theta) / np.cos(theta) ** 2
    A = np.column_stack([np.ones_like(g), g])
    (_, b), *_ = np.linalg.lstsq(A, dx, rcond=None)
    return b, b * np.max(np.abs(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("csv")
    ap.add_argument("--fx", type=float, help="focale en pixels (calibration)")
    ap.add_argument("--z-near", type=float, help="distance cible proche (m)")
    ap.add_argument("--z-far", type=float, help="distance cible lointaine (m)")
    args = ap.parse_args()

    measures = load_measures(args.csv)
    if not measures:
        sys.exit(f"Aucune mesure dans {args.csv}")

    have_geom = args.fx and args.z_near and args.z_far
    k = None
    if have_geom:
        # px par mm de décalage pupille/axe et par unité de g(θ)
        k = args.fx * (1.0 / (args.z_near * 1000) - 1.0 / (args.z_far * 1000))

    slides, slopes = [], []
    print(f"{'glissière (mm)':>15} {'parallaxe à θmax (px)':>22} {'pupille devant l’axe (mm)':>27}")
    for slide, samples in measures.items():
        b, p_max = fit_parallax(samples)
        slides.append(slide)
        slopes.append(b)
        d_txt = f"{-b / k:+.1f}" if k else "—"
        flag = "  <- OK" if abs(p_max) < SEUIL_PARALLAXE_PX else ""
        print(f"{slide:>15.1f} {p_max:>+22.2f} {d_txt:>27}{flag}")

    if len(slides) >= 2:
        slope, intercept = np.polyfit(slides, slopes, 1)
        if slope == 0:
            sys.exit("Parallaxe identique à toutes les positions : vérifier les mesures.")
        s0 = -intercept / slope
        print(f"\nPoint sans parallaxe : glissière = {s0:.1f} mm")
        if not min(slides) <= s0 <= max(slides):
            print("  (extrapolé : refaire des mesures qui encadrent cette position)")
        if k:
            # pente théorique attendue : ±k par mm de glissière
            print(f"  cohérence fx/distances : pente mesurée / attendue = {abs(slope) / k:.2f} (idéal ≈ 1)")
        print("-> Placer la glissière à cette graduation, puis mesurer au réglet\n"
              "   la distance axe de rotation <-> repère plan capteur (⦵) :\n"
              "   c'est la position de l'entrance pupil devant le capteur.")
    elif k:
        print("\nd > 0 : pupille DEVANT l'axe -> reculer la caméra de d sur la glissière.\n"
              "d < 0 : pupille DERRIÈRE l'axe -> avancer la caméra de |d|.")
    else:
        print("\nUne seule position : ajouter des positions de glissière, ou fournir "
              "--fx --z-near --z-far pour estimer la distance pupille <-> axe.")


if __name__ == "__main__":
    main()
