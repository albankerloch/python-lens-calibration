#!/usr/bin/env python3
"""
Étape 1 — Test rapide de distorsion.

Mesure la courbure résiduelle d'une ligne DROITE de la mire, sur une frame
extraite dans le format EXACT de production (même mode, même résolution,
même crop, ADC dans l'état qui sera tourné).

Usage :
    python mesure_courbure.py frame.png

Protocole :
  1. Clic gauche sur 5 à 9 points le long d'UNE ligne de la mire, en allant
     d'un bord de l'image à l'autre. Choisir la ligne la plus PÉRIPHÉRIQUE
     possible (le cadre extérieur en haut ou en bas) : c'est là que la
     distorsion radiale est la plus forte.
  2. Touche 'z' = annuler le dernier point, 'r' = tout effacer.
  3. Touche ENTRÉE = calculer.  'q' = quitter.

Le script trace la droite idéale entre le premier et le dernier point cliqué,
puis mesure l'écart perpendiculaire maximal des points intermédiaires.
"""

import sys
import numpy as np
import cv2

# --- seuils de décision (cf. rapport, section "Seuils qui changent la décision")
SEUIL_NEGLIGEABLE = 1.0   # px : en dessous, distorsion considérée neutralisée
SEUIL_A_CALIBRER = 3.0    # px : au dessus, calibration complète indispensable

if len(sys.argv) < 2:
    sys.exit("Usage: python mesure_courbure.py frame.png")

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit(f"Impossible de lire {sys.argv[1]}")
h, w = img.shape[:2]
pts = []

# affichage redimensionné, mais mesures faites en pixels natifs
scale = min(1.0, 1600 / w)
disp_w, disp_h = int(w * scale), int(h * scale)


def redraw():
    vis = cv2.resize(img, (disp_w, disp_h)) if scale < 1.0 else img.copy()
    for i, (x, y) in enumerate(pts):
        p = (int(x * scale), int(y * scale))
        cv2.circle(vis, p, 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i + 1), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if len(pts) >= 2:
        a = (int(pts[0][0] * scale), int(pts[0][1] * scale))
        b = (int(pts[-1][0] * scale), int(pts[-1][1] * scale))
        cv2.line(vis, a, b, (0, 255, 0), 1)
    cv2.putText(vis, f"{len(pts)} points | ENTREE=calculer  z=annuler  r=reset  q=quitter",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow("mire", vis)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x / scale, y / scale))
        redraw()


def analyse():
    if len(pts) < 3:
        print("Il faut au moins 3 points (2 extrémités + 1 intermédiaire).")
        return
    P = np.array(pts, float)
    a, b = P[0], P[-1]
    d = b - a
    L = np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) / L          # normale unitaire à la corde
    ecarts = (P[1:-1] - a) @ n               # écart signé de chaque point interne

    i = int(np.argmax(np.abs(ecarts)))
    emax = ecarts[i]
    sagitta = abs(emax)

    # sens de la courbure par rapport au centre image
    milieu = P[len(P) // 2]
    centre = np.array([w / 2.0, h / 2.0])
    vers_centre = (centre - milieu) @ n
    sens = "barillet (la ligne bombe VERS L'EXTÉRIEUR)" if emax * vers_centre < 0 \
        else "coussinet (la ligne creuse VERS LE CENTRE)"

    print("\n" + "=" * 62)
    print(f"Image                 : {w} x {h} px")
    print(f"Longueur de corde     : {L:.0f} px ({100*L/w:.0f} % de la largeur)")
    print(f"Écarts intermédiaires : " + ", ".join(f"{e:+.2f}" for e in ecarts) + " px")
    print(f"FLÈCHE MAXIMALE       : {sagitta:.2f} px  ({100*sagitta/h:.3f} % de la hauteur)")
    if sagitta > 0.3:
        print(f"Sens                  : {sens}")
    print("-" * 62)
    if sagitta < SEUIL_NEGLIGEABLE:
        print("→ LIGNES DROITES. Distorsion neutralisée en interne.")
        print("  Ne calibrer QUE les intrinsèques (fx, fy, cx, cy) + le nodal offset.")
        print("  Ne PAS appliquer de dé-distorsion supplémentaire dans UE.")
    elif sagitta < SEUIL_A_CALIBRER:
        print("→ ZONE GRISE. Distorsion résiduelle faible mais mesurable.")
        print("  Calibration ChArUco complète recommandée ; un modèle k1,k2 suffira")
        print("  probablement (CALIB_FIX_K3, CALIB_ZERO_TANGENT_DIST).")
    else:
        print("→ LIGNES COURBES. Distorsion à calibrer entièrement.")
        print("  ChArUco + Brown-Conrady 5 coefficients, puis application sur")
        print("  l'inner frustum via le Lens File UE.")
    print("=" * 62 + "\n")
    print("Rappel : refaire cette mesure pour CHAQUE format candidat")
    print("(R3D NE / N-RAW vs H.265 ADC ON vs H.265 ADC OFF).\n")


cv2.namedWindow("mire", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("mire", on_mouse)
redraw()
while True:
    k = cv2.waitKey(20) & 0xFF
    if k in (13, 10):
        analyse()
    elif k == ord('z') and pts:
        pts.pop(); redraw()
    elif k == ord('r'):
        pts.clear(); redraw()
    elif k == ord('q') or k == 27:
        break
cv2.destroyAllWindows()
