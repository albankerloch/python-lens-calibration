"""Matching d'éclairage : acteur keyé (Fusion/Resolve) <-> décor Unreal <-> image de référence.

Le programme compare des *rapports* plutôt que des valeurs absolues. La référence
et le comp sont mesurés de la même façon : image de sortie (Rec.709 gamma 2.4,
look compris) décodée en linéaire d'affichage. La correction, elle, est appliquée
avant Look_Film ; la boucle compense la non-linéarité du look.

L'anneau de décor autour de l'acteur est coupé en deux à la verticale de l'acteur :
le demi-anneau le plus lumineux est le côté key, l'autre le côté fill.

    key_stops      log2(p95 luminance acteur) - log2(médiane du décor côté key)
    hi_stops       log2(p95 luminance acteur) - log2(p95 du décor autour, hors pixels écrêtés) :
                   hautes lumières contre hautes lumières, indépendant de ce que l'anneau
                   contient de sombre (étagère, meuble)
    contrast_stops log2(p90 / p20) de la luminance de l'acteur (ratio key/fill)
    warmth_r/b     log2(R/G), log2(B/G) des hautes lumières de l'acteur
                   MOINS celles du décor autour (teinte relative acteur/décor)
    light_r/b      idem, mais contre les hautes lumières du décor côté key (couleur de la lumière)
    black_stops    log2(ombres de l'acteur) - log2(ombres du décor côté fill)
    sat_stops      log2(saturation des tons moyens de l'acteur / celle du décor autour)
    shadow_stops   log2(médiane de la partie de l'acteur dans l'ombre portée (matte Sink_Shadow))
                   - log2(médiane du décor autour pris dans la même ombre)
    bg_stops, bg_r/b  exposition et teinte du décor lui-même (image entière)
    key_side       position horizontale des hautes lumières dans l'acteur
                   (-1 = gauche, +1 = droite) : information, non corrigée
    key_fill_stops écart key/fill de l'acteur moins celui du décor : information, non corrigée
                   (> 0 : l'acteur est plus contrasté d'un côté à l'autre que le décor)

La partie de l'acteur dans l'ombre portée est exclue des autres mesures : sinon ses
5 % les plus sombres seraient l'ombre ajoutée, et les noirs de tout l'acteur seraient relevés.

Deux priorités (--priority) :

    integration (défaut)  l'acteur est d'abord calé sur le décor CGI qui l'entoure : mêmes
                          noirs, même couleur de lumière. L'exposition et la saturation suivent
                          la référence, mais à --margin stops au plus de ce que montre le décor.
                          La fidélité à la référence (exposition, teinte) est ensuite visée
                          sur l'image entière, après le merge : le raccord acteur/décor reste.
    reference             ancien comportement : les rapports acteur/décor de la référence
                          sont imposés à l'acteur seul.
    harmony               sans référence : l'acteur est seulement harmonisé avec le décor qui
                          l'entoure (contre l'effet de collage). Hautes lumières, couleur de la
                          lumière, saturation, noirs et ombre portée visent ceux du décor
                          (écart nul) ; l'exposition de l'acteur est réglée par hi_stops
                          (critère match). --reference est alors facultatif.

Corrections :

    acteur (Person_Match, avant le merge)   out = G_c * in ** k_c   (par canal), puis saturation
                                            integration : calage deux points, G_c sur les
                                            hautes lumières, k_c sur les ombres (noirs) ;
                                            reference : k commun = contraste
    ombre  (Sink_Shadow, avant le merge)    gain de la zone d'ombre portée
    décor  (BG_Match, avant le merge)       out = G * in ** k, calage deux points du décor :
                                            sa médiane sur celle de la référence (bg_stops),
                                            ses hautes lumières sur l'écart hi_stops
    image  (Ref_Grade, après le merge)      out = G_c * in

Plusieurs frames (--frames) : sans --animate, une correction unique calée sur la moyenne
des frames ; avec --animate, une correction par frame, posée en keyframes.

Usage :
    # 1. mesurer la référence (boîte de l'acteur en pixels, ou masque PNG)
    python src/light_match.py reference --image ref.png --subject-box 480,150,750,650 -o ref_metrics.json

    # 2. mesurer le comp (calques exportés par resolve/light_match_resolve.py) et calculer la correction
    python src/light_match.py solve --layers <dossier> --reference ref_metrics.json -o correction.json

    # 3. boucle complète via l'API de Resolve (scripting externe activé dans Resolve)
    python src/light_match.py loop --reference ref_metrics.json --work <dossier> [--max-iter 6]         [--frames 394,640,884,1130,1373 [--animate]]

    # options de 2 et 3 : --priority integration|reference, --margin 0.25,
    #   --correct exposure,light,blacks,saturation,shadow (acteur), --global exposure,warmth|none (image)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Rec.709 / sRGB : même primaires, coefficients de luminance
LUMA = np.array([0.2126, 0.7152, 0.0722])
EPS = 1e-6

# tolérances de convergence (stops / log2)
TOL = {"key_stops": 0.10, "contrast_stops": 0.15, "warmth_r": 0.03, "warmth_b": 0.03,
       "light_r": 0.03, "light_b": 0.03, "black_stops": 0.15,
       "sat_stops": 0.08, "shadow_stops": 0.20, "hi_stops": 0.15,
       "bg_stops": 0.10, "bg_r": 0.03, "bg_b": 0.03}

# limites par itération, pour éviter qu'une mesure aberrante fasse diverger la boucle
MAX_STEP_STOPS = 1.5
K_RANGE = (0.6, 1.6)
DAMPING = 0.7
# bornes de la correction cumulée : au-delà, la cible est hors d'atteinte (ex. look saturé)
K_TOTAL = (0.4, 2.5)
GAIN_TOTAL = (2 ** -6, 2 ** 6)
SAT_TOTAL = (0.3, 2.0)
BG_K_TOTAL = (0.5, 3.0)
# le gain du décor compense le gamma autour d'une médiane très basse (~2^-8) : il lui faut
# bien plus de marge que les ±6 stops de l'acteur (k = 2 -> environ 8 stops de gain)
BG_GAIN_TOTAL = (2 ** -20, 2 ** 20)
SHADOW_GAIN_TOTAL = (0.005, 1.0)

# écart toléré entre l'acteur et le décor quand la référence demande plus (priorité integration)
MARGIN_STOPS = 0.25

# priorité harmony : écart visé entre les hautes lumières de l'acteur et celles du décor.
# Un T-shirt blanc peut être un peu plus clair que le carrelage éclairé : viser 0 assombrissait
# l'acteur jusqu'au niveau d'un CGI sous-exposé (-3,6 stops sur Take_41)
HI_TARGET_STOPS = 1.0

# en dessous (luminance linéaire, ~ code 10/255 en gamma 2.4), les ombres d'un still 8 bits
# sont écrasées ou quantifiées : un rapport contre elles n'a pas de sens (noirs, ombre portée)
BLACK_FLOOR = 2 ** -11
# code value au-delà duquel un canal est considéré écrêté (par le look ou le format)
CLIP_CODE = 0.97

# les images sont réduites à cette largeur avant la mesure : les rapports en stops n'en
# dépendent quasiment pas, et l'anneau (dilatation de 10 % de la largeur) devient ~16x plus rapide
MEASURE_WIDTH = 1008
# érosion des masques, en pixels à MEASURE_WIDTH (9 px en 4032 de large)
ERODE_PX = 3

# critères corrigés sur l'acteur, et les mesures qui jugent leur convergence
CRITERIA = {"exposure": ("key_stops",), "contrast": ("contrast_stops",),
            "warmth": ("warmth_r", "warmth_b"), "light": ("light_r", "light_b"),
            "blacks": ("black_stops",), "saturation": ("sat_stops",), "shadow": ("shadow_stops",),
            "decor": ("hi_stops", "bg_stops"), "match": ("hi_stops",)}
# critères corrigés sur l'image entière (après le merge), jugés sur le décor
GLOBAL_CRITERIA = {"exposure": ("bg_stops",), "warmth": ("bg_r", "bg_b")}
DEFAULTS = {"integration": {"correct": ("decor", "light", "blacks", "saturation", "shadow"),
                            "global": ("exposure", "warmth")},
            "reference": {"correct": ("exposure", "contrast", "warmth"), "global": ()},
            "harmony": {"correct": ("match", "light", "saturation", "blacks", "shadow"), "global": ()}}
CORRECTABLE = DEFAULTS["integration"]["correct"]
CORRECT_HELP = (f"critères corrigés sur l'acteur parmi {', '.join(CRITERIA)} "
                "(défaut : decor,light,blacks,saturation,shadow ;"
                " en priorité reference : exposure,contrast,warmth ;"
                " en priorité harmony : match,light,saturation,blacks,shadow)")
GLOBAL_HELP = (f"critères corrigés sur l'image entière parmi {', '.join(GLOBAL_CRITERIA)}, ou none "
               "(défaut : exposure,warmth ; en priorité reference : none)")


def parse_criteria(text: str, allowed: dict) -> tuple[str, ...]:
    if text.strip() == "none":
        return ()
    items = tuple(t.strip() for t in text.split(",") if t.strip())
    unknown = [t for t in items if t not in allowed]
    if unknown or not items:
        raise SystemExit(f"critères possibles : {', '.join(allowed)} (reçu : {text})")
    return items


def criteria_from_args(a) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Critères acteur et image entière, avec les défauts de la priorité choisie."""
    d = DEFAULTS[a.priority]
    correct = parse_criteria(a.correct, CRITERIA) if a.correct else d["correct"]
    glob = parse_criteria(a.global_criteria, GLOBAL_CRITERIA) if a.global_criteria else d["global"]
    if {"warmth", "light"} <= set(correct):
        raise SystemExit("warmth et light corrigent tous deux la teinte de l'acteur : en choisir un")
    if "decor" in correct and "exposure" in glob:
        raise SystemExit("decor cale déjà le niveau médian du décor (bg_stops) : ne pas y ajouter"
                         " --global exposure")
    if {"exposure", "decor"} <= set(correct):
        raise SystemExit("exposure (acteur) et decor (hautes lumières du décor) règlent tous deux l'écart"
                         " acteur/décor : en choisir un")
    if {"contrast", "blacks"} <= set(correct):
        raise SystemExit("contrast et blacks règlent tous deux le gamma de l'acteur : en choisir un")
    if "match" in correct and {"exposure", "decor"} & set(correct):
        raise SystemExit("match règle déjà l'écart de hautes lumières acteur/décor : ne pas y ajouter"
                         " exposure ni decor")
    return correct, glob


def load_reference(a) -> dict | None:
    """Mesures de la référence ; facultatives en priorité harmony (aucune cible n'en vient)."""
    if not a.reference:
        if a.priority != "harmony":
            raise SystemExit("--reference est requis, sauf en priorité harmony")
        return None
    return complete(json.loads(Path(a.reference).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- images

def read_image(path: Path) -> np.ndarray:
    """Retourne un tableau float64 HxWxC (3 ou 4 canaux, ordre RGB[A])."""
    path = Path(path)
    with path.open("rb") as f:
        is_exr = f.read(4) == b"\x76\x2f\x31\x01"
    if is_exr:
        import OpenEXR
        with OpenEXR.File(str(path)) as exr:
            ch = exr.channels()
            if "RGBA" in ch:
                return ch["RGBA"].pixels.astype(np.float64)
            if "RGB" in ch:
                return ch["RGB"].pixels.astype(np.float64)
            return np.stack([ch[c].pixels for c in "RGBA" if c in ch], -1).astype(np.float64)
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    im = im.astype(np.float64) / scale
    if im.ndim == 2:
        return np.repeat(im[..., None], 3, -1)
    return im[..., [2, 1, 0, 3]] if im.shape[2] == 4 else im[..., ::-1]


def decode(rgb: np.ndarray, transfer: str) -> np.ndarray:
    """Code value -> linéaire (display-linear pour une image étalonnée)."""
    rgb = np.clip(rgb, 0.0, None)
    if transfer == "linear":
        return rgb
    if transfer == "gamma24":
        return rgb ** 2.4
    if transfer == "srgb":
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    raise ValueError(transfer)


def small(im: np.ndarray, width: int = MEASURE_WIDTH) -> np.ndarray:
    """Image réduite à `width` de large (moyenne par zone), inchangée si déjà plus petite."""
    h, w = im.shape[:2]
    if w <= width:
        return im
    return cv2.resize(im, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    m = np.zeros_like(mask)
    h, w = mask.shape
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    m[yd, xd] = mask[ys, xs]
    return m


def ring_around(subject: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Anneau de décor autour de l'acteur, entre `inner` et `outer` pixels."""
    u8 = subject.astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return (cv2.dilate(u8, k(outer)) > 0) & ~(cv2.dilate(u8, k(inner)) > 0)


# --------------------------------------------------------------------------- mesures

def split_ring(bg_mask: np.ndarray, subject_mask: np.ndarray, lum: np.ndarray, key_dir: int | None = None):
    """Coupe l'anneau de décor à la verticale du centre de l'acteur.

    Retourne (côté key, côté fill, key_dir) ; key_dir = +1 si la key vient de droite.
    Sans `key_dir`, le côté key est le demi-anneau dont la médiane est la plus lumineuse.
    Si un demi-anneau est trop petit (acteur au bord du cadre), l'anneau entier sert aux deux.
    """
    cx = np.nonzero(subject_mask)[1].mean()
    right = bg_mask & (np.arange(bg_mask.shape[1])[None, :] >= cx)
    left = bg_mask & ~right
    if right.sum() < 500 or left.sum() < 500:
        return bg_mask, bg_mask, key_dir or 0
    if key_dir is None:
        key_dir = 1 if np.median(lum[right]) >= np.median(lum[left]) else -1
    return (right, left, key_dir) if key_dir > 0 else (left, right, key_dir)


def saturation(rgb: np.ndarray, lum: np.ndarray, white: np.ndarray) -> float:
    """Saturation moyenne (max - min) / max des tons moyens (entre p20 et p90 de luminance),
    après neutralisation de la dominante : chaque canal est divisé par celui de `white`
    (couleur des hautes lumières de la zone). Un gain par canal (réchauffer, refroidir)
    ne change donc pas la mesure : seule la saturation propre de la zone compte."""
    lo, hi = np.percentile(lum, [20, 90])
    mid = rgb[(lum >= lo) & (lum <= hi)] / np.maximum(white, EPS)
    mx = np.maximum(mid.max(-1), EPS)
    return float(np.mean((mx - mid.min(-1)) / mx))


def measure(subject_rgb: np.ndarray, subject_mask: np.ndarray,
            bg_rgb: np.ndarray, bg_mask: np.ndarray,
            shadow_mask: np.ndarray | None = None, bg_shadow_mask: np.ndarray | None = None,
            key_dir: int | None = None, clipped: np.ndarray | None = None) -> dict:
    """Métriques de rapport entre l'acteur et le décor qui l'entoure (images linéaires).

    `shadow_mask` : partie de l'acteur dans l'ombre portée, `bg_shadow_mask` : décor autour
    pris dans la même ombre ; tous deux facultatifs (shadow_stops absent sinon).
    `clipped` : pixels écrêtés (un canal au-delà de CLIP_CODE), exclus des mesures de teinte
    des hautes lumières : leur rapport R/G ne bouge plus quand on corrige.
    """
    if clipped is None:
        clipped = np.zeros(subject_mask.shape, bool)
    if subject_mask.sum() < 500 or bg_mask.sum() < 500:
        raise ValueError("masque acteur ou décor trop petit (< 500 px)")
    s = subject_rgb[subject_mask]
    ls = np.maximum(s @ LUMA, EPS)
    ys, xs = np.nonzero(subject_mask)
    lit = ls >= np.percentile(ls, 80)
    width = max(xs.max() - xs.min(), 1)
    # décalage des hautes lumières par rapport au centre de l'acteur, ±1 = un quart de sa largeur
    key_side = float(np.clip((xs[lit].mean() - xs.mean()) / (width / 4), -1, 1))
    # le côté key est celui d'où la lumière frappe l'acteur ; à défaut (lumière de face),
    # le demi-anneau de décor le plus lumineux
    if key_dir is None and abs(key_side) >= 0.1:
        key_dir = 1 if key_side > 0 else -1
    lum_bg_img = np.maximum(bg_rgb @ LUMA, EPS)
    key_m, fill_m, key_dir = split_ring(bg_mask, subject_mask, lum_bg_img, key_dir)
    b = bg_rgb[bg_mask]
    bk = bg_rgb[key_m]
    bf = bg_rgb[fill_m]
    lb = np.maximum(b @ LUMA, EPS)
    lbk = np.maximum(bk @ LUMA, EPS)
    lbf = np.maximum(bf @ LUMA, EPS)

    p20, p90, p95 = np.percentile(ls, [20, 90, 95])
    bp20, bp90 = np.percentile(lb, [20, 90])
    b_ok = ~clipped[bg_mask]
    bg_p95 = np.percentile(lb[b_ok] if b_ok.sum() >= 100 else lb, 95)
    kp95 = np.percentile(lbk, 95)
    s_ok = ~clipped[subject_mask]
    if s_ok.sum() < 100:
        raise ValueError("acteur presque entièrement écrêté : baisser l'exposition avant de mesurer")
    # hautes lumières de l'acteur (T-shirt), sans les pixels écrêtés
    hi = s[s_ok & (ls >= np.percentile(ls[s_ok], 85))].mean(0)
    bgm = b[lb >= np.percentile(lb, 25)].mean(0)       # sans les quasi-noirs, dominés par le bruit
    bk_ok = ~clipped[key_m]
    if bk_ok.sum() < 100:
        bk_ok = np.ones_like(bk_ok)
    # surfaces du décor les plus éclairées, côté key, sans les pixels écrêtés
    bg_hi = bk[bk_ok & (lbk >= np.percentile(lbk[bk_ok], 85))].mean(0)
    shadow = s[ls <= np.percentile(ls, 5)].mean(0)
    bg_shadow = bf[lbf <= np.percentile(lbf, 5)].mean(0)   # ombres du décor, côté fill

    def chroma(c):
        c = np.maximum(c, EPS)
        return math.log2(c[0] / c[1]), math.log2(c[2] / c[1])

    sr, sb = chroma(hi)
    br, bb = chroma(bgm)
    hr, hb = chroma(bg_hi)

    # écart key/fill : moitié de l'acteur côté key contre l'autre, comparé au décor
    on_key = (xs >= xs.mean()) if key_dir >= 0 else (xs < xs.mean())
    key_fill = (math.log2(np.median(ls[on_key])) - math.log2(np.median(ls[~on_key]))
                - (math.log2(np.median(lbk)) - math.log2(np.median(lbf))))

    m = {
        "key_stops": math.log2(p95) - math.log2(np.median(lbk)),
        "hi_stops": math.log2(p95) - math.log2(bg_p95),
        "contrast_stops": math.log2(p90 / p20),
        "warmth_r": sr - br,
        "warmth_b": sb - bb,
        "light_r": sr - hr,
        "light_b": sb - hb,
        "black_stops": math.log2(max(shadow @ LUMA, EPS)) - math.log2(max(bg_shadow @ LUMA, EPS)),
        "_bg_shadow_crushed": bool(bg_shadow @ LUMA < BLACK_FLOOR),
        "sat_stops": math.log2(max(saturation(s, ls, hi), EPS) / max(saturation(b, lb, bgm), EPS)),
        "bg_stops": math.log2(np.median(lb)),
        "bg_r": br,
        "bg_b": bb,
        "key_side": key_side,
        "key_fill_stops": key_fill,
        "oriented": True,
        # valeurs brutes, utiles pour la correction et le diagnostic
        "_key_dir": key_dir,
        "_subject_p95": float(p95),
        "_subject_wb": [sr, sb],
        "_subject_hi": hi.tolist(),
        "_subject_shadow": shadow.tolist(),
        "_bg_median": float(np.median(lb)),
        "_bg_p95": float(bg_p95),
        "_bg_key_median": float(np.median(lbk)),
        "_bg_wb": [br, bb],
        "_bg_light_wb": [hr, hb],
        "_bg_shadow": bg_shadow.tolist(),
        "_bg_key": math.log2(kp95) - math.log2(np.median(lbk)),   # même mesure que key_stops, sur le décor
        "_bg_contrast": math.log2(bp90 / bp20),
        "_subject_px": int(subject_mask.sum()),
        "_subject_clipped": float(1 - s_ok.mean()),
        "_bg_px": int(bg_mask.sum()),
    }
    if shadow_mask is not None and shadow_mask.sum() >= 500 and bg_shadow_mask.sum() >= 500:
        lss = np.maximum(subject_rgb[shadow_mask] @ LUMA, EPS)
        lbs = np.maximum(bg_rgb[bg_shadow_mask] @ LUMA, EPS)
        if np.median(lbs) < BLACK_FLOOR:
            m["_shadow_crushed"] = True
        else:
            m["shadow_stops"] = math.log2(np.median(lss)) - math.log2(np.median(lbs))
        m["_shadow_px"] = int(shadow_mask.sum())
    return m


def complete(m: dict) -> dict:
    """Ajoute les mesures absentes des fichiers de référence écrits avant leur introduction."""
    if not m.get("oriented") or "hi_stops" not in m:
        raise SystemExit("mesures de référence d'une version antérieure (anneau non orienté, sans"
                         " saturation) : relancer « light_match.py reference » sur la référence")
    m = dict(m)
    m.setdefault("bg_stops", math.log2(m["_bg_median"]))
    m.setdefault("bg_r", m["_bg_wb"][0])
    m.setdefault("bg_b", m["_bg_wb"][1])
    return m


def parse_box(text: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in text.split(","))
    return x0, y0, x1, y1


def active_area(code: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """Masque de l'image utile, sans les bandes noires (letterbox / pillarbox)."""
    peak = code.max(-1)
    rows = peak.max(1) > floor
    cols = peak.max(0) > floor
    return rows[:, None] & cols[None, :]


def grabcut_subject(code: np.ndarray, box) -> np.ndarray:
    """Détoure l'acteur à l'intérieur de `box` (x0, y0, x1, y1) par GrabCut."""
    x0, y0, x1, y1 = box
    bgr = (np.clip(code[..., ::-1], 0, 1) * 255).astype(np.uint8)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    m = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    # on garde la plus grande composante connexe, érodée pour éviter les bords
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n > 1:
        m = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.erode(m.astype(np.uint8), k) > 0


def measure_reference(image: Path, transfer: str, subject_box=None, subject_mask=None,
                      ring=(0.02, 0.10), save_mask: Path | None = None) -> dict:
    """Référence : image unique, acteur défini par un masque PNG (blanc = acteur)
    ou par une boîte affinée par GrabCut."""
    code = small(read_image(image)[..., :3])
    rgb = decode(code, transfer)
    h, w = rgb.shape[:2]
    if subject_mask is not None:
        m = read_image(subject_mask)[..., 0] > 0.5
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    elif subject_box is not None:
        m = grabcut_subject(code, subject_box)
    else:
        raise ValueError("--subject-box ou --subject-mask requis pour la référence")
    if save_mask is not None:
        cv2.imwrite(str(save_mask), m.astype(np.uint8) * 255)
    bg = ring_around(m, int(ring[0] * w), int(ring[1] * w)) & active_area(code)
    return measure(rgb, m, rgb, bg, clipped=code.max(-1) >= CLIP_CODE)


def check_subject_mask(subject: np.ndarray, max_fill: float = 0.75) -> None:
    """Refuse un masque d'acteur invraisemblable.

    Une silhouette remplit 50 à 70 % de son rectangle englobant (63 % mesuré sur Take_41).
    Si CorridorKey a échoué, toute la zone de garbage passe opaque et le « masque »
    devient un rectangle plein (85 % mesuré) : le mesurer donnerait une correction fausse.
    """
    ys, xs = np.nonzero(subject)
    if len(xs) < 500:
        raise ValueError("masque d'acteur vide : key absent ou frame sans acteur")
    fill = len(xs) / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    if fill > max_fill:
        raise ValueError(f"masque d'acteur rectangulaire (remplissage {fill:.0%}) :"
                         " le key a probablement échoué sur cette frame")


def _matte(layers: Path, meta: dict, key: str, shape) -> np.ndarray:
    """Matte exporté en still (blanc linéaire 1.0), décodé comme la frame et mis à la taille."""
    m = decode(read_image(layers / meta[key])[..., :3], meta.get("transfer", "gamma24")).mean(-1)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return m


def measure_layers(layers: Path, ring=(0.02, 0.10), exclude_bottom: float = 0.0) -> dict:
    """Comp : calques exportés par light_match_resolve (export_stills ou export_layers).

    Les pixels viennent du still de la timeline (sortie finale, look compris), décodé
    comme la référence. Le masque vient soit du matte exporté en still (déjà à sa place
    dans l'image), soit de l'alpha de l'acteur rendu dans Fusion (à décaler et redimensionner).
    Si le matte de l'ombre portée est exporté (shadow.png), la partie de l'acteur qu'il couvre
    est retirée des mesures de l'acteur et mesurée à part (shadow_stops).
    Si la frame est aussi exportée sans le look (prelook), noirs et ombre portée sont mesurés
    dessus : le look écrase les ombres du décor à 0, et les corrections (Person_Match,
    Sink_Shadow) agissent de toute façon avant lui.
    `exclude_bottom` : part du bas du cadre exclue du décor (raccord de sol non représentatif).
    """
    meta = json.loads((layers / "layers.json").read_text(encoding="utf-8"))
    code = small(read_image(layers / meta["still"])[..., :3])
    rgb = decode(code, meta.get("transfer", "gamma24"))
    h, w = rgb.shape[:2]

    if "matte" in meta:
        alpha = _matte(layers, meta, "matte", (h, w))
    else:
        alpha = np.clip(read_image(layers / meta["person"])[..., 3], 0.0, 1.0)
        ah, aw = alpha.shape
        # l'acteur est décalé par le Merge (Center) : on place son masque là où il apparaît
        dx = int(round((meta["merge_center"][0] - 0.5) * aw))
        dy = int(round(-(meta["merge_center"][1] - 0.5) * ah))   # y Fusion vers le haut
        alpha = shift(alpha, dx, dy)
        # comp 4032x2268 -> timeline 3840x2160 : mise à l'échelle uniforme (scale to fit)
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    active = active_area(code)
    opaque = (cv2.erode((alpha > 0.98).astype(np.uint8), k) > 0) & active
    check_subject_mask(opaque)
    bg_mask = ring_around(alpha > 0.5, int(ring[0] * w), int(ring[1] * w)) & active
    if exclude_bottom > 0:
        bg_mask[int(h * (1 - exclude_bottom)):] = False

    subject, shadow_subject, bg_shadow = opaque, None, None
    if "shadow" in meta:
        sh = _matte(layers, meta, "shadow", (h, w))
        # marge autour du bord doux de l'ombre : ni tout à fait éclairé, ni tout à fait dans l'ombre
        touched = cv2.dilate((sh > 0.03).astype(np.uint8), k) > 0
        subject = opaque & ~touched
        shadow_subject = opaque & (cv2.erode((sh > 0.9).astype(np.uint8), k) > 0)
        bg_shadow = bg_mask & (sh > 0.9)
        bg_mask = bg_mask & ~touched
    m = measure(rgb, subject, rgb, bg_mask, shadow_subject, bg_shadow, clipped=code.max(-1) >= CLIP_CODE)
    if "prelook" in meta:
        pre_code = small(read_image(layers / meta["prelook"])[..., :3])
        pre_rgb = decode(pre_code, meta.get("transfer", "gamma24"))
        pre = measure(pre_rgb, subject, pre_rgb, bg_mask, shadow_subject, bg_shadow,
                      key_dir=m["_key_dir"], clipped=pre_code.max(-1) >= CLIP_CODE)
        for key in ("shadow_stops", "_shadow_crushed", "_shadow_px"):
            m.pop(key, None)
        m.update({key: pre[key] for key in ("black_stops", "_subject_shadow", "_bg_shadow", "_bg_shadow_crushed",
                                            "shadow_stops", "_shadow_crushed", "_shadow_px") if key in pre})
        # écart hautes lumières / ombres de l'acteur dans le même domaine que ses ombres (calage deux points)
        m["_pre_subject_hi"] = pre["_subject_hi"]
        m["_pre_bg_median"], m["_pre_bg_p95"] = pre["_bg_median"], pre["_bg_p95"]
        m["_dark_source"] = "prelook"
    return m


# --------------------------------------------------------------------------- correction

IDENTITY = {"k": [1.0, 1.0, 1.0], "gain": [1.0, 1.0, 1.0], "global_gain": [1.0, 1.0, 1.0],
            "sat": 1.0, "shadow_gain": None, "bg_k": 1.0, "bg_gain": 1.0}


def normalized(corr: dict | None) -> dict:
    """Correction complétée par l'identité ; un k unique (anciens fichiers) vaut pour les 3 canaux."""
    c = {**IDENTITY, **(corr or {})}
    if not isinstance(c["k"], (list, tuple)):
        c["k"] = [float(c["k"])] * 3
    c.pop("lift", None)
    return c


def targets(current: dict, reference: dict, priority: str, margin: float = MARGIN_STOPS,
            hi_target: float = HI_TARGET_STOPS) -> dict:
    """Valeur visée pour chaque mesure de TOL.

    hi_stops (hautes lumières acteur / décor) vise l'écart de la référence : c'est le décor
    qui est corrigé (BG_Match), l'exposition de l'acteur reste en place.
    Noirs et ombre portée visent le décor du comp (écart nul) ; la couleur de la lumière vise
    l'écart de la référence (borné à `margin` en integration) ; le décor
    de l'image entière vise celui de la référence. Exposition, contraste et saturation de
    l'acteur suivent la référence ; en priorité integration, ils restent à `margin` stops au
    plus de ce que le décor montre lui-même (ses hautes lumières, son contraste, sa saturation).
    En priorité harmony, la référence est ignorée : les rapports acteur/décor visent 0, et les
    mesures non corrigées gardent leur valeur actuelle (écart nul, simple information).
    """
    if priority == "harmony":
        t = {key: current[key] for key in (*TOL, "key_side") if key in current}
        t.update(hi_stops=hi_target, light_r=0.0, light_b=0.0, sat_stops=0.0, black_stops=0.0, shadow_stops=0.0)
        return t
    t = {key: reference[key] for key in
         ("key_stops", "hi_stops", "contrast_stops", "warmth_r", "warmth_b", "sat_stops",
          "bg_stops", "bg_r", "bg_b", "key_side")}
    # couleur de la lumière : l'écart de teinte acteur / hautes lumières du décor de la référence
    # (sur Take_41, T-shirt plus froid que le mur chaud : light_b +0,42)
    t.update(light_r=reference["light_r"], light_b=reference["light_b"], black_stops=0.0, shadow_stops=0.0)
    if priority == "integration":
        t["sat_stops"] = float(np.clip(t["sat_stops"], -margin, margin))
        # au plus à `margin` de la couleur de la lumière du décor
        t["light_r"] = float(np.clip(t["light_r"], -margin, margin))
        t["light_b"] = float(np.clip(t["light_b"], -margin, margin))
        t["key_stops"] = float(np.clip(t["key_stops"], current["_bg_key"] - margin, current["_bg_key"] + margin))
        t["contrast_stops"] = float(np.clip(t["contrast_stops"], current["_bg_contrast"] - margin,
                                            current["_bg_contrast"] + margin))
    return t


def solve(current: dict, target: dict, previous: dict | None = None,
          correct=CORRECTABLE) -> dict:
    """Correction cumulée de l'acteur out = gain_c * in ** k_c (par canal) pour amener
    `current` sur `target` (issu de targets()).

    Avec `blacks` : calage deux points par canal. Les hautes lumières de l'acteur visent
    l'exposition (key_stops) et la teinte demandées, ses ombres celles du décor (niveau et
    teinte) ; gain et gamma de chaque canal s'en déduisent exactement.
    Sans `blacks` : k commun aux trois canaux, réglé par le contraste (contrast_stops).

    `correct` : critères corrigés parmi CRITERIA ; les autres sont laissés en l'état
    (ex. le contraste, quand l'écart vient de la direction de la lumière et non du grade).
    Les clés de `previous` non concernées (global_gain) sont conservées.
    """
    prev = normalized(previous)

    # la mesure est faite après Look_Film (non linéaire) alors que la correction est
    # appliquée avant : le pas analytique n'est qu'approché, on l'amortit
    dr = db = 0.0
    tint = "warmth" if "warmth" in correct else "light" if "light" in correct else None
    if tint:
        dr = float(np.clip(DAMPING * (target[f"{tint}_r"] - current[f"{tint}_r"]), -0.5, 0.5))
        db = float(np.clip(DAMPING * (target[f"{tint}_b"] - current[f"{tint}_b"]), -0.5, 0.5))
    tint_step = np.array([dr, 0.0, db])

    # match : gain de l'acteur pour caler ses hautes lumières sur celles du décor (hi_stops)
    e_match = 0.0
    if "match" in correct:
        e_match = float(np.clip(DAMPING * (target["hi_stops"] - current["hi_stops"]),
                                -MAX_STEP_STOPS, MAX_STEP_STOPS))

    if "blacks" in correct:
        e = e_match
        if "exposure" in correct:
            e = float(np.clip(DAMPING * (target["key_stops"] - current["key_stops"]),
                              -MAX_STEP_STOPS, MAX_STEP_STOPS))
        # hautes lumières et ombres de l'acteur mesurées sur la même image (sans le look si
        # elle a été exportée) : l'écart entre les deux fixe le gamma
        l_hi = np.log2(np.maximum(current.get("_pre_subject_hi", current["_subject_hi"]), EPS))
        l_sh = np.log2(np.maximum(current["_subject_shadow"], EPS))
        aim = np.log2(np.maximum(current["_bg_shadow"], EPS)) + target["black_stops"]
        hi_t = l_hi + e + tint_step
        sh_t = l_sh + DAMPING * np.clip(aim - l_sh, -MAX_STEP_STOPS, MAX_STEP_STOPS)
        k_step = np.clip((hi_t - sh_t) / np.maximum(l_hi - l_sh, 0.5), *K_RANGE)
        log_g = hi_t - k_step * l_hi
    else:
        k_step = 1.0
        if "contrast" in correct:
            k_step = target["contrast_stops"] / max(current["contrast_stops"], 0.05)
            k_step = float(np.clip(k_step ** DAMPING, *K_RANGE))
        g = e_match
        if "exposure" in correct:
            # avec out = G * in**k, log2(p95_out) = log2 G + k * log2(p95_in)
            lp95 = math.log2(current["_subject_p95"])
            g = target["key_stops"] + math.log2(current["_bg_median"]) - k_step * lp95
            g = float(np.clip(DAMPING * g, -MAX_STEP_STOPS, MAX_STEP_STOPS))
        k_step = np.full(3, k_step)
        log_g = g + tint_step

    # composition avec la correction déjà en place : G2 * (G1 * x**k1)**k2
    k_raw = np.asarray(prev["k"]) * k_step
    gain_raw = 2 ** log_g * np.asarray(prev["gain"]) ** k_step
    k = np.clip(k_raw, *K_TOTAL)
    gain = np.clip(gain_raw, *GAIN_TOTAL)
    limited = bool((k != k_raw).any() or (gain != gain_raw).any())

    # saturation : la saturation (max - min) / max des tons moyens varie comme le facteur appliqué
    sat, ds = prev["sat"], 0.0
    if "saturation" in correct:
        ds = float(np.clip(DAMPING * (target["sat_stops"] - current["sat_stops"]), -0.5, 0.5))
        sat_raw = sat * 2 ** ds
        sat = float(np.clip(sat_raw, *SAT_TOTAL))
        limited = limited or sat != sat_raw
    # ombre portée : gain de Sink_Shadow (la zone d'ombre de l'acteur varie comme ce gain)
    shadow_gain, dsh = prev["shadow_gain"], 0.0
    if "shadow" in correct and "shadow_stops" in current and shadow_gain is not None:
        dsh = float(np.clip(DAMPING * (target["shadow_stops"] - current["shadow_stops"]),
                            -MAX_STEP_STOPS, MAX_STEP_STOPS))
        shadow_gain = float(np.clip(shadow_gain * 2 ** dsh, *SHADOW_GAIN_TOTAL))
    # décor : calage deux points, out = G * in ** k. La médiane vise celle de la référence
    # (bg_stops), les hautes lumières l'écart hi_stops avec celles de l'acteur.
    bg_k, bg_gain, dk = prev["bg_k"], prev["bg_gain"], 1.0
    if "decor" in correct:
        l_med = math.log2(max(current.get("_pre_bg_median", current["_bg_median"]), EPS))
        l_p95 = math.log2(max(current.get("_pre_bg_p95", current["_bg_p95"]), EPS))
        d_med = float(np.clip(DAMPING * (target["bg_stops"] - current["bg_stops"]),
                              -MAX_STEP_STOPS, MAX_STEP_STOPS))
        d_p95 = float(np.clip(DAMPING * (current["hi_stops"] - target["hi_stops"]),
                              -MAX_STEP_STOPS, MAX_STEP_STOPS))
        dk = float(np.clip(1 + (d_p95 - d_med) / max(l_p95 - l_med, 0.5), *K_RANGE))
        bg_k_raw = bg_k * dk
        bg_k = float(np.clip(bg_k_raw, *BG_K_TOTAL))
        # gain calculé pour le gamma réellement appliqué : s'il est borné, la médiane doit
        # quand même arriver où elle est visée
        dk = bg_k / prev["bg_k"]
        bg_gain_raw = 2 ** (l_med + d_med - dk * l_med) * bg_gain ** dk
        bg_gain = float(np.clip(bg_gain_raw, *BG_GAIN_TOTAL))
        limited = limited or bg_k != bg_k_raw or bg_gain != bg_gain_raw
    return {**prev, "k": k.tolist(), "gain": gain.tolist(), "sat": sat, "shadow_gain": shadow_gain,
            "bg_k": bg_k, "bg_gain": bg_gain, "limited": limited,
            "step": {"k": k_step.tolist(), "gain_stops": log_g.tolist(), "dr": dr, "db": db,
                     "sat_stops": ds, "shadow_stops": dsh, "bg_k": dk}}


def solve_global(current: dict, target: dict, previous: dict | None = None, criteria=()) -> dict:
    """Gain cumulé de l'image entière (après le merge) pour amener le décor sur celui de la
    référence. Un gain commun à l'acteur et au décor ne change pas leur raccord."""
    prev = normalized(previous)
    g = dr = db = 0.0
    if "exposure" in criteria:
        g = float(np.clip(DAMPING * (target["bg_stops"] - current["bg_stops"]), -MAX_STEP_STOPS, MAX_STEP_STOPS))
    if "warmth" in criteria:
        dr = float(np.clip(DAMPING * (target["bg_r"] - current["bg_r"]), -0.5, 0.5))
        db = float(np.clip(DAMPING * (target["bg_b"] - current["bg_b"]), -0.5, 0.5))
    step = [2 ** (g + dr), 2 ** g, 2 ** (g + db)]
    gg = np.clip(np.asarray(step) * np.asarray(prev["global_gain"]), *GAIN_TOTAL).tolist()
    return {**prev, "global_gain": gg, "global_step": {"gain_stops": g, "dr": dr, "db": db}}


def errors(current: dict, target: dict) -> dict:
    """Écarts à la cible ; shadow_stops n'existe que si le matte d'ombre a été exporté et que
    le décor dans l'ombre n'est pas écrasé ; black_stops est ignoré si les ombres du décor le sont."""
    err = {key: current[key] - target[key] for key in TOL if key in current and key in target}
    if current.get("_bg_shadow_crushed"):
        err.pop("black_stops", None)
    return err


def watched_keys(correct=CORRECTABLE, glob=()) -> list[str]:
    return [key for c in correct for key in CRITERIA[c]] + [key for c in glob for key in GLOBAL_CRITERIA[c]]


def converged(err: dict, correct=CORRECTABLE, glob=()) -> bool:
    """Convergence sur les seuls critères corrigés."""
    return all(abs(err[key]) <= TOL[key] for key in watched_keys(correct, glob) if key in err)


def report(label: str, m: dict, target: dict | None = None, watched=None) -> str:
    """Mesures (et écarts à la cible). `watched` : mesures corrigées, marquées d'un * ;
    les autres sont affichées pour information."""
    keys = ["key_stops", "hi_stops", "contrast_stops", "warmth_r", "warmth_b", "sat_stops"]
    if target is not None:
        keys += ["light_r", "light_b", "black_stops", "shadow_stops", "bg_stops", "bg_r", "bg_b"]
    lines = [label]
    for key in [*keys, "key_side", "key_fill_stops"]:
        if key not in m:
            continue
        mark = "*" if watched and key in watched else " "
        line = f" {mark}{key:15s} {m[key]:+.3f}"
        if target is not None and key in target:
            line += f"   cible {target[key]:+.3f}   écart {m[key] - target[key]:+.3f}"
        lines.append(line)
    if target is not None and m["key_side"] * target["key_side"] < 0:
        lines.append("  ! la key n'arrive pas du même côté que dans la référence :"
                     " à corriger à l'éclairage (plateau ou lights UE), pas en étalonnage")
    if m.get("_subject_clipped", 0) > 0.05:
        lines.append(f"  ! {m['_subject_clipped']:.0%} de l'acteur écrêté : exclu des mesures de teinte"
                     " (un roll-off des hautes lumières lui rendrait du modelé)")
    if m.get("_bg_shadow_crushed"):
        lines.append("  ! ombres du décor écrasées (sous 2^-11) : black_stops non mesurable, noirs non corrigés")
    if m.get("_shadow_crushed"):
        lines.append("  ! décor écrasé dans l'ombre portée : shadow_stops non mesurable, ombre non corrigée")
    if abs(m.get("key_fill_stops", 0.0)) > 0.5:
        lines.append(f"  ! écart key/fill de l'acteur différent du décor ({m['key_fill_stops']:+.2f} stop) :"
                     " non corrigeable par un grade uniforme (gradient ou éclairage)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def cmd_reference(a):
    out = Path(a.output)
    m = measure_reference(a.image, a.transfer, parse_box(a.subject_box) if a.subject_box else None,
                          a.subject_mask, save_mask=out.with_name(out.stem + "_mask.png"))
    Path(a.output).write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(report(f"Référence {a.image}", m))


def step_all(current: dict, target: dict, corr: dict | None, correct, glob) -> dict:
    if current.get("_bg_shadow_crushed"):
        correct = tuple(c for c in correct if c != "blacks")
    corr = solve(current, target, corr, correct)
    return solve_global(current, target, corr, glob) if glob else corr


def describe(corr: dict, glob) -> str:
    text = (f"k={[round(k, 3) for k in corr['k']]} gain={[round(g, 4) for g in corr['gain']]}"
            f" sat={corr['sat']:.3f}")
    if corr.get("shadow_gain") is not None:
        text += f" ombre={corr['shadow_gain']:.4f}"
    if corr.get("bg_k", 1.0) != 1.0 or corr.get("bg_gain", 1.0) != 1.0:
        text += f" | décor : k={corr['bg_k']:.3f} gain={corr['bg_gain']:.4f}"
    if glob:
        text += f" | image : gain={[round(g, 4) for g in corr['global_gain']]}"
    if corr.get("limited"):
        text += (f"\n  ! correction bornée (acteur : k {K_TOTAL}, gain ±6 stops ; saturation {SAT_TOTAL} ;"
                 f" décor : k {BG_K_TOTAL}) : cible hors"
                 " d'atteinte en étalonnage (contraste de l'acteur trop différent du décor ?"
                 " essayer --margin plus grand, ou sans blacks)")
    return text


def cmd_solve(a):
    correct, glob = criteria_from_args(a)
    reference = load_reference(a)
    current = measure_layers(Path(a.layers), exclude_bottom=a.exclude_bottom)
    target = targets(current, reference, a.priority, a.margin, a.hi_target)
    prev = json.loads(Path(a.previous).read_text(encoding="utf-8")) if a.previous else None
    err = errors(current, target)
    corr = step_all(current, target, prev, correct, glob)
    corr.update({"metrics": current, "targets": target, "errors": err,
                 "converged": converged(err, correct, glob), "priority": a.priority,
                 "corrected": list(correct), "corrected_global": list(glob)})
    Path(a.output).write_text(json.dumps(corr, indent=2), encoding="utf-8")
    print(report(f"Comp (priorité {a.priority})", current, target, watched_keys(correct, glob)))
    print(f"correction cumulée : {describe(corr, glob)}  {'(convergé)' if corr['converged'] else ''}")


# mesures brutes d'intensité : moyennées en log (moyenne géométrique), les autres en linéaire
LOG_AVERAGED = ("_subject_p95", "_subject_hi", "_subject_shadow", "_bg_median", "_bg_key_median", "_bg_shadow",
                "_bg_p95", "_pre_bg_median", "_pre_bg_p95", "_pre_subject_hi")


def average_metrics(ms: list[dict]) -> dict:
    """Moyenne de mesures de plusieurs frames (clés présentes dans toutes les frames)."""
    out = dict(ms[0])
    for key in ms[0]:
        if not all(key in m for m in ms):
            out.pop(key)
            continue
        vals = [m[key] for m in ms]
        if isinstance(vals[0], bool) or not isinstance(vals[0], (int, float, list)):
            continue
        arr = np.asarray(vals, dtype=float)
        avg = 2 ** np.log2(np.maximum(arr, EPS)).mean(0) if key in LOG_AVERAGED else arr.mean(0)
        out[key] = avg.tolist() if isinstance(vals[0], list) else float(avg)
    return out


def spread_warnings(ms: list[dict], keys) -> list[str]:
    """Mesures qui varient trop d'une frame à l'autre pour une correction unique."""
    out = []
    for key in keys:
        vals = [m[key] for m in ms if key in m]
        if len(vals) > 1 and max(vals) - min(vals) > 2 * TOL[key]:
            out.append(f"{key} varie de {max(vals) - min(vals):.2f} entre les frames")
    return out


def parse_frames(text: str | None, single: str | None) -> list:
    """Frames de la boucle : timecodes (01:00:26:14) ou temps du comp (884) séparés par des virgules."""
    if not text:
        return [single]
    return [f if ":" in f else int(f) for f in (t.strip() for t in text.split(",")) if f]


def stagnated(err: dict, previous: dict | None, keys) -> bool:
    """Vrai si aucun critère encore hors tolérance n'a réduit son écart d'au moins la moitié de
    sa tolérance depuis l'itération précédente : continuer ne ferait que tourner pour rien."""
    if previous is None:
        return False
    open_keys = [k for k in keys if k in err and k in previous and abs(err[k]) > TOL[k]]
    return bool(open_keys) and all(abs(previous[k]) - abs(err[k]) < TOL[k] / 2 for k in open_keys)


def cmd_loop(a):
    sys.path.insert(0, str(Path(__file__).parent / "resolve"))
    import light_match_resolve as lmr

    correct, glob = criteria_from_args(a)
    reference = load_reference(a)
    watched = watched_keys(correct, glob)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    shot = lmr.get_comp(a.timeline, a.track)
    frames = [lmr.timecode_for(shot, f) if isinstance(f, int) else f for f in parse_frames(a.frames, a.frame)]
    animate = a.animate and len(frames) > 1
    times = {tc: lmr.comp_time(shot, tc) for tc in frames}
    # le still sans look ne sert qu'aux noirs, à l'ombre portée et au pivot du décor
    prelook = bool({"blacks", "shadow", "decor"} & set(correct))
    prev_err = None
    corrs = {tc: lmr.read_correction(shot, times[tc] if animate else None) for tc in frames}
    print(f"Priorité {a.priority} : acteur {','.join(correct)} ; image {','.join(glob) or 'aucun'} ;"
          f" {len(frames)} frame(s){' en keyframes' if animate else ''}")
    for it in range(1, a.max_iter + 1):
        measured = {}
        for tc in frames:
            layers = work / f"iter_{it:02d}" / (f"f{int(times[tc]):04d}" if tc else "tete")
            meta = lmr.export_stills(shot, layers, tc, prelook=prelook)
            measured[tc] = measure_layers(layers, exclude_bottom=a.exclude_bottom)
            if len(frames) > 1:
                print(f"  frame {meta['timecode']} (comp {meta['comp_time']}) mesurée", flush=True)
        if animate:
            pending = {}
            for tc, current in measured.items():
                target = targets(current, reference, a.priority, a.margin, a.hi_target)
                err = errors(current, target)
                print(report(f"Itération {it}, frame {tc}", current, target, watched), flush=True)
                if not converged(err, correct, glob):
                    pending[tc] = step_all(current, target, corrs[tc], correct, glob)
            if not pending:
                print("Convergé sur toutes les frames.")
                break
            waited = lmr.wait_ready()
            for tc, corr in pending.items():
                corrs[tc] = corr
                lmr.apply_correction(shot, corr, with_global=bool(glob), time=times[tc])
                print(f"  -> {tc} : {describe(corr, glob)}", flush=True)
            print(f"  (attente Resolve {waited:.0f} s)", flush=True)
            continue
        current = average_metrics(list(measured.values()))
        target = targets(current, reference, a.priority, a.margin, a.hi_target)
        err = errors(current, target)
        label = f"Itération {it}" + (f" (moyenne de {len(frames)} frames)" if len(frames) > 1 else "")
        print(report(label, current, target, watched), flush=True)
        for w in spread_warnings(list(measured.values()), watched):
            print(f"  ! {w} : envisager --animate", flush=True)
        if converged(err, correct, glob):
            print("Convergé.")
            break
        if stagnated(err, prev_err, watched):
            print("Arrêt : plus aucun critère ne progresse (stagnation). Voir les avertissements ci-dessus.")
            break
        prev_err = err
        corr = step_all(current, target, corrs[frames[0]], correct, glob)
        corrs = {tc: corr for tc in frames}
        waited = lmr.wait_ready()
        lmr.apply_correction(shot, corr, with_global=bool(glob))
        print(f"  -> {describe(corr, glob)} (attente Resolve {waited:.0f} s)", flush=True)
    else:
        print(f"Pas de convergence après {a.max_iter} itérations : vérifier masque, key_side et spill.")
    out = {tc or "tete": c for tc, c in corrs.items()} if animate else corrs[frames[0]]
    (work / "correction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def add_priority_args(p):
    p.add_argument("--priority", choices=list(DEFAULTS), default="integration",
                   help="integration : l'acteur d'abord calé sur le décor CGI (défaut) ;"
                        " reference : rapports acteur/décor de la référence imposés à l'acteur ;"
                        " harmony : acteur harmonisé avec le décor, sans référence")
    p.add_argument("--margin", type=float, default=MARGIN_STOPS,
                   help="priorité integration : écart max (stops) entre l'acteur et le décor"
                        " pour l'exposition, le contraste et la saturation")
    p.add_argument("--exclude-bottom", type=float, default=0.0,
                   help="part du bas du cadre exclue du décor (ex. 0.08 pour un raccord de sol)")
    p.add_argument("--hi-target", type=float, default=HI_TARGET_STOPS,
                   help="priorité harmony : écart visé (stops) entre les hautes lumières de l'acteur"
                        " et celles du décor (défaut 1.0)")
    p.add_argument("--correct", help=CORRECT_HELP)
    p.add_argument("--global", dest="global_criteria", help=GLOBAL_HELP)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reference", help="mesurer l'image de référence")
    r.add_argument("--image", required=True, type=Path)
    r.add_argument("--transfer", default="gamma24", choices=["gamma24", "srgb", "linear"])
    r.add_argument("--subject-box", help="x0,y0,x1,y1 en pixels")
    r.add_argument("--subject-mask", type=Path, help="PNG, blanc = acteur")
    r.add_argument("-o", "--output", default="ref_metrics.json")
    r.set_defaults(func=cmd_reference)

    s = sub.add_parser("solve", help="mesurer des calques exportés et calculer la correction")
    s.add_argument("--layers", required=True)
    s.add_argument("--reference", help="mesures de la référence (facultatif en priorité harmony)")
    s.add_argument("--previous", help="correction.json déjà appliquée (pour cumuler)")
    s.add_argument("-o", "--output", default="correction.json")
    add_priority_args(s)
    s.set_defaults(func=cmd_solve)

    lp = sub.add_parser("loop", help="boucle export -> mesure -> correction via l'API de Resolve")
    lp.add_argument("--reference", help="mesures de la référence (facultatif en priorité harmony)")
    lp.add_argument("--work", required=True)
    lp.add_argument("--timeline", default="Take_41")
    lp.add_argument("--track", type=int, help="piste du clip acteur (défaut : recherche du comp contenant BG_Merge)")
    lp.add_argument("--frame", help="timecode de la frame mesurée, ex. 01:00:26:14 (défaut : tête de lecture)")
    lp.add_argument("--frames", help="plusieurs frames, timecodes ou temps du comp séparés par des virgules"
                                     " (ex. 394,640,884,1130,1373) : correction calée sur leur moyenne")
    lp.add_argument("--animate", action="store_true",
                    help="avec --frames : une correction par frame, posée en keyframes")
    lp.add_argument("--max-iter", type=int, default=6)
    add_priority_args(lp)
    lp.set_defaults(func=cmd_loop)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
