"""Visualisation de la distorsion d'objectif à partir de fichiers .ulens."""

import json
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# streamlit_image_coordinates 0.4.0 importe un symbole interne renommé dans les versions
# récentes de Streamlit (>=1.5x). Ce symbole n'est utilisé que comme annotation de type
# (jamais évalué au runtime, grâce à `from __future__ import annotations` dans ce module),
# donc un simple stub suffit à débloquer l'import sans toucher au package installé.
import streamlit.elements.image as _st_image_module
if not hasattr(_st_image_module, "UseColumnWith"):
    _st_image_module.UseColumnWith = str
from streamlit_image_coordinates import streamlit_image_coordinates

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
CALIBRATION_INPUT_DIR = Path(__file__).resolve().parent.parent / "calibration_input"

DECKLINK_DEVICE_NAME = "Blackmagic WDM Capture"

SEUIL_NEGLIGEABLE = 1.0  # px : en dessous, distorsion considérée neutralisée
SEUIL_A_CALIBRER = 3.0   # px : au dessus, calibration complète indispensable

REPROJ_ERROR_BON = 0.5   # px : erreur de reprojection par image, seuil "bon" (cf. doc)
REPROJ_ERROR_MAUVAIS = 1.0  # px : au dessus, image à retirer/reprendre


def format_reprojection_error(err: float) -> str:
    if err < REPROJ_ERROR_BON:
        return f"✅ {err:.3f} px"
    if err < REPROJ_ERROR_MAUVAIS:
        return f"⚠️ {err:.3f} px"
    return f"❌ {err:.3f} px"


def render_stability_comparison(stability: dict):
    """Affiche la comparaison Cx/Cy (et Fx/Fy pour contexte) entre deux recalibrations
    indépendantes sur la 1ère et la 2e moitié des vues (test de stabilité du centre image).
    """
    r1, r2 = stability["half1"], stability["half2"]
    if not (r1.get("success") and r2.get("success")):
        st.error(
            "Damier insuffisamment détecté dans au moins une des deux moitiés "
            f"(moitié 1 : {r1.get('detected_count', 0)}/{r1.get('total_count', 0)}, "
            f"moitié 2 : {r2.get('detected_count', 0)}/{r2.get('total_count', 0)}) — "
            "il faut au moins 3 vues valides par moitié."
        )
        return

    w, h = r1["image_size"]
    dcx = r2["cx_px"] - r1["cx_px"]
    dcy = r2["cy_px"] - r1["cy_px"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Cx — moitié 1", f"{r1['cx_px']:.1f} px")
    col2.metric("Cx — moitié 2", f"{r2['cx_px']:.1f} px", f"{dcx:+.1f} px")
    col3.metric("Écart Cx", f"{100 * dcx / w:+.2f} % largeur")

    col4, col5, col6 = st.columns(3)
    col4.metric("Cy — moitié 1", f"{r1['cy_px']:.1f} px")
    col5.metric("Cy — moitié 2", f"{r2['cy_px']:.1f} px", f"{dcy:+.1f} px")
    col6.metric("Écart Cy", f"{100 * dcy / h:+.2f} % hauteur")

    with st.expander("Fx / Fy pour contexte"):
        dfx = r2["fx_px"] - r1["fx_px"]
        dfy = r2["fy_px"] - r1["fy_px"]
        st.write(f"Fx : {r1['fx_px']:.1f} px → {r2['fx_px']:.1f} px (Δ {dfx:+.1f} px)")
        st.write(f"Fy : {r1['fy_px']:.1f} px → {r2['fy_px']:.1f} px (Δ {dfy:+.1f} px)")

    st.caption(
        f"Moitié 1 : {r1['detected_count']} vue(s), RMS {r1['rms']:.3f} px — "
        f"Moitié 2 : {r2['detected_count']} vue(s), RMS {r2['rms']:.3f} px. "
        "Un écart de quelques pixels entre les deux moitiés est normal ; un écart de "
        "plusieurs dizaines de pixels indique une couverture de captures insuffisante "
        "(pas assez d'angles/positions variés) ou un point principal mal contraint."
    )


def parse_table(table: dict) -> list[dict]:
    """Transforme une cameraParameterTable en liste de lignes (dict par colonne)."""
    columns = [c.strip() for c in table["header"].split(",")]
    rows = []
    for line in table["data"].splitlines():
        line = line.strip()
        if not line:
            continue
        values = [float(v.strip()) for v in line.split(",")]
        rows.append(dict(zip(columns, values)))
    return rows


@st.cache_data
def load_ulens(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lens = json.load(f)

    tables = {
        t["parameterName"]: parse_table(t)
        for t in lens["cameraParameterTables"]
    }
    return {
        "metaData": lens["metaData"],
        "sensorDimensions": lens["sensorDimensions"],
        "imageDimensions": lens["imageDimensions"],
        "tables": tables,
    }


def compute_distortion_grid(fx, fy, k1, k2, k3, p1, p2, width, height, n_lines):
    xs = np.linspace(-width / 2, width / 2, n_lines)
    ys = np.linspace(-height / 2, height / 2, n_lines)
    grid_x, grid_y = np.meshgrid(xs, ys)

    xn = grid_x / fx
    yn = grid_y / fy
    r2 = xn**2 + yn**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn

    dist_x = xd * fx
    dist_y = yd * fy
    return grid_x, grid_y, dist_x, dist_y


def compute_displacement_map(fx, fy, k1, k2, k3, p1, p2, width, height, resolution):
    xs = np.linspace(-width / 2, width / 2, resolution)
    ys = np.linspace(-height / 2, height / 2, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    xn = grid_x / fx
    yn = grid_y / fy
    r2 = xn**2 + yn**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn

    disp = np.hypot((xd - xn) * fx, (yd - yn) * fy)
    return grid_x, grid_y, disp


@st.cache_data
def build_distortion_figure(fx, fy, k1, k2, k3, p1, p2, width, height, n_lines, exaggeration, focal_mm, hfov_deg):
    grid_x, grid_y, dist_x, dist_y = compute_distortion_grid(
        fx, fy, k1, k2, k3, p1, p2, width, height, n_lines
    )
    exag_x = grid_x + (dist_x - grid_x) * exaggeration
    exag_y = grid_y + (dist_y - grid_y) * exaggeration

    disp_grid_x, disp_grid_y, disp = compute_displacement_map(
        fx, fy, k1, k2, k3, p1, p2, width, height, resolution=200
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Distorsion du Lens File — K1={k1:.3g}  K2={k2:.3g}  K3={k3:.3g}  "
        f"P1={p1:.3g}  P2={p2:.3g}  (focale {focal_mm:.1f} mm, HFOV {hfov_deg:.1f}°)"
    )

    ax1.set_title(f"Grille déformée (bleu) vs idéale (gris) — exagération ×{exaggeration:g}")
    for i in range(n_lines):
        ax1.plot(grid_x[i, :], grid_y[i, :], color="lightgray", linewidth=0.8)
        ax1.plot(grid_x[:, i], grid_y[:, i], color="lightgray", linewidth=0.8)
        ax1.plot(exag_x[i, :], exag_y[i, :], color="steelblue", linewidth=1.2)
        ax1.plot(exag_x[:, i], exag_y[:, i], color="steelblue", linewidth=1.2)
    ax1.set_xlabel("pixels")
    ax1.set_ylabel("pixels")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Amplitude du déplacement dû à la distorsion")
    mesh = ax2.pcolormesh(disp_grid_x, disp_grid_y, disp, cmap="Blues", shading="auto")
    contours = ax2.contour(disp_grid_x, disp_grid_y, disp, colors="white", linewidths=0.8)
    ax2.clabel(contours, inline=True, fontsize=8, fmt="%.0f px")
    ax2.set_xlabel("pixels")
    ax2.set_ylabel("pixels")
    ax2.set_aspect("equal")
    fig.colorbar(mesh, ax=ax2, label=f"déplacement (pixels, image {width:g} px)")
    return fig


STREAM_INFO_RE = re.compile(rb"Video:.*?(\d+)x(\d+)[^,]*,\s*([\d.]+)\s*fps")


class LatestFrame:
    """Boîte à une seule case : le thread de capture écrase en continu, l'UI lit la dernière valeur.

    Lire le pipe ffmpeg séquentiellement depuis la boucle Streamlit (rerun par rerun) fait
    prendre du retard dès que l'affichage/l'encodage PNG est plus lent que la cadence de
    capture : les frames s'accumulent dans le pipe et on regarde alors le passé. En
    découplant capture (thread dédié, boucle serrée) et affichage (échantillonnage de la
    dernière frame connue), on élimine ce retard qui grandit dans le temps : les frames
    non affichées sont simplement perdues plutôt que mises en file.
    """

    def __init__(self):
        self.frame = None
        self.seq = 0
        self.lock = threading.Lock()
        self.stopped = False


def _drain_stderr(proc: subprocess.Popen, line_queue: "queue.Queue"):
    for line in iter(proc.stderr.readline, b""):
        line_queue.put(line)
    line_queue.put(None)


def _capture_frames(proc: subprocess.Popen, width: int, height: int, holder: LatestFrame):
    frame_size = width * height * 3
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))[:, :, ::-1]
        with holder.lock:
            holder.frame = frame
            holder.seq += 1
    with holder.lock:
        holder.stopped = True


def start_decklink_capture(device_name: str) -> tuple[subprocess.Popen, "queue.Queue"]:
    """Démarre ffmpeg sur le périphérique DirectShow, sans forcer de résolution/cadence.

    Passer un -video_size/-framerate explicite au pilote WDM de la DeckLink peut faire
    bloquer indéfiniment la négociation : on laisse dshow auto-détecter le signal SDI reçu,
    puis on lit la résolution réellement négociée dans le banner stderr de ffmpeg (voir
    detect_stream_format) plutôt que de la pré-interroger via ffprobe, qui s'est montré
    sujet aux mêmes blocages sur ce pilote.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats",
        "-f", "dshow",
        "-i", f"video={device_name}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-an", "-",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    line_queue = queue.Queue()
    threading.Thread(target=_drain_stderr, args=(proc, line_queue), daemon=True).start()
    return proc, line_queue


def detect_stream_format(line_queue: "queue.Queue", timeout: float = 8.0):
    """Lit le banner stderr de ffmpeg pour en extraire largeur/hauteur/fps négociés."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = line_queue.get(timeout=max(0.1, deadline - time.time()))
        except queue.Empty:
            break
        if line is None:
            break
        match = STREAM_INFO_RE.search(line)
        if match:
            return int(match.group(1)), int(match.group(2)), match.group(3).decode()
    return None


def _calibrate_from_correspondences(
    objpoints: list, imgpoints: list, image_size: tuple[int, int],
    total_count: int, used_indices: list, fix_distortion: bool = False,
    fix_principal_point: bool = False, source: str = "live",
) -> dict:
    """Calcule K1-P2/Fx/Fy/Cx/Cy + erreurs de reprojection à partir de correspondances
    objet/image déjà connues (détectées par nous, ou importées d'une session existante).

    Si fix_distortion est vrai (caméra dont la distorsion est déjà corrigée en interne,
    ex. Nikon ZR + Auto distortion control ON, confirmé par l'onglet « Caméra déjà
    calibrée ? »), K1/K2/K3/P1/P2 sont forcés à 0 et seuls Fx, Fy, Cx, Cy sont résolus —
    cf. doc/calibration_Nikkor_28mm.md, section « Flags utiles » : la correction interne
    change le FOV effectif, donc fx/fy/cx/cy doivent quand même être mesurés.

    Si fix_principal_point est vrai, Cx/Cy sont forcés au centre géométrique de l'image
    (CALIB_FIX_PRINCIPAL_POINT, sans CALIB_USE_INTRINSIC_GUESS : reste au centre pendant
    toute l'optimisation) plutôt que résolus librement — cumulable avec fix_distortion.

    `source` ("live" ou "session") identifie la galerie (captures live vs session importée)
    à laquelle appartiennent used_indices, pour un ré-affichage fiable de l'erreur par image.
    """
    if len(objpoints) < 3:
        return {
            "success": False,
            "detected_count": len(objpoints),
            "total_count": total_count,
            "source": source,
        }

    flags = 0
    if fix_distortion:
        flags |= (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        )
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    per_view_errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        observed = imgpoints[i].reshape(-1, 2).astype(np.float64)
        projected = projected.reshape(-1, 2).astype(np.float64)
        error = cv2.norm(observed, projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))

    k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

    return {
        "success": True,
        "detected_count": len(objpoints),
        "total_count": total_count,
        "used_indices": used_indices,
        "image_size": image_size,
        "rms": float(rms),
        "per_view_errors": per_view_errors,
        "fx_px": float(camera_matrix[0, 0]),
        "fy_px": float(camera_matrix[1, 1]),
        "cx_px": float(camera_matrix[0, 2]),
        "cy_px": float(camera_matrix[1, 2]),
        "k1": float(k1), "k2": float(k2), "k3": float(k3),
        "p1": float(p1), "p2": float(p2),
        "fix_distortion": fix_distortion,
        "fix_principal_point": fix_principal_point,
        "source": source,
    }


def run_camera_calibration(
    frames: list, pattern_size: tuple[int, int],
    fix_distortion: bool = False, fix_principal_point: bool = False,
) -> dict:
    """Calibration de caméra OpenCV (damier) à partir d'une liste de frames RGB.

    Détecte les coins du damier dans chaque frame (findChessboardCorners), puis calibre.
    La taille réelle des cases du damier n'influence pas K1-P2/Fx/Fy (invariants d'échelle
    du modèle sténopé) : on utilise donc des points objet en unités arbitraires (case = 1).
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints, imgpoints, used_indices = [], [], []
    image_size = None

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used_indices.append(idx)

    return _calibrate_from_correspondences(
        objpoints, imgpoints, image_size, len(frames), used_indices,
        fix_distortion, fix_principal_point, source="live",
    )


def _planarize_object_points(points_3d) -> np.ndarray:
    """Réexprime un nuage de points 3D planaire dans son propre repère local (z=0).

    Les Row*.json exportent des objectPoints dans un repère monde arbitraire (le plan du
    damier y est incliné, pas aligné sur z=0). C'est mathématiquement toujours un plan
    (confirmé par SVD : la 3e valeur singulière est ~0), mais cv2.calibrateCamera exige des
    points de calibration explicitement dans le plan z=0 pour son initialisation planaire —
    sinon il réclame une matrice caméra initiale (« non-planar calibration rigs »). Cette
    reprojection rigide (PCA) ne change ni les distances ni les angles réels entre points,
    donc n'affecte pas K1-P2/Fx/Fy/Cx/Cy recouvrés, seulement les rvec/tvec par vue.
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    basis = vt[:2]
    local_2d = (pts - centroid) @ basis.T
    return np.column_stack([local_2d, np.zeros(len(pts))]).astype(np.float32)


def run_camera_calibration_from_session(
    rows: list, image_size: tuple[int, int],
    fix_distortion: bool = False, fix_principal_point: bool = False,
) -> dict:
    """Calibration à partir de correspondances objet/image déjà connues (session importée).

    Contrairement à run_camera_calibration, aucune détection de damier n'est refaite : les
    points 2D/3D viennent directement des Row*.json de la session (déjà appariés).
    """
    objpoints = [_planarize_object_points(r["object_points"]) for r in rows]
    imgpoints = [np.asarray(r["image_points"], dtype=np.float32).reshape(-1, 1, 2) for r in rows]
    used_indices = [r["index"] for r in rows]
    return _calibrate_from_correspondences(
        objpoints, imgpoints, image_size, len(rows), used_indices,
        fix_distortion, fix_principal_point, source="session",
    )


def build_ulens_json(name, serial, focus, zoom, result, sensor_w_mm, sensor_h_mm, fstop=0.0):
    width, height = result["image_size"]
    fx_norm = result["fx_px"] / width
    fy_norm = result["fy_px"] / height
    cx_norm = result["cx_px"] / width
    cy_norm = result["cy_px"] / height
    # Le f-stop n'affecte pas K1-P2/Fx/Fy/Cx/Cy (distorsion géométrique indépendante du
    # diaphragme) : enregistré uniquement en métadonnée de traçabilité, jamais comme axe
    # des cameraParameterTables (qui ne sont indexées que par Focus/Zoom, cf. .ulens).
    user_metadata = [{"name": "FStop", "value": f"{fstop:g}"}] if fstop > 0 else []
    return {
        "metaData": {
            "type": "LensFile",
            "version": "0.0.0",
            "lensInfo": {
                "serialNumber": serial,
                "modelName": name,
                "distortionModel": "Spherical",
            },
            "name": name,
            "nodalOffsetCoordinateSystem": "OpenCV",
            "fxFyUnits": "Normalized",
            "cxCyUnits": "Normalized",
            "userMetadata": user_metadata,
        },
        "sensorDimensions": {"width": sensor_w_mm, "height": sensor_h_mm, "units": "Millimeters"},
        "imageDimensions": {"width": width, "height": height},
        "cameraParameterTables": [
            {
                "parameterName": "FocalLengthTable",
                "header": "FocusEncoder, ZoomEncoder, Fx, Fy",
                "data": f"{focus:g}, {zoom:g}, {fx_norm:.6f}, {fy_norm:.6f}",
            },
            {
                "parameterName": "ImageCenterTable",
                "header": "FocusEncoder, ZoomEncoder, Cx, Cy",
                "data": f"{focus:g}, {zoom:g}, {cx_norm:.6f}, {cy_norm:.6f}",
            },
            {
                "parameterName": "NodalOffsetTable",
                "header": "FocusEncoder, ZoomEncoder, Qx, Qy, Qz, Qw, Tx, Ty, Tz",
                "data": f"{focus:g}, {zoom:g}, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0",
            },
            {
                "parameterName": "DistortionTable",
                "header": "FocusEncoder, ZoomEncoder, K1, K2, K3, P1, P2",
                "data": (
                    f"{focus:g}, {zoom:g}, {result['k1']:.6f}, {result['k2']:.6f}, "
                    f"{result['k3']:.6f}, {result['p1']:.6f}, {result['p2']:.6f}"
                ),
            },
        ],
        "encoderTables": [
            {"parameterName": "Focus", "header": "FocusEncoder, FocusCM", "data": ""},
            {"parameterName": "Iris", "header": "IrisEncoder, IrisFstop", "data": ""},
        ],
    }


def _read_json_maybe_utf16(path: Path) -> dict:
    """Les fichiers .ucamcalib / Row*.json de ce format sont encodés en UTF-16."""
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-16"))
    except (UnicodeError, ValueError):
        return json.loads(raw.decode("utf-8"))


def find_calibration_sessions() -> list[Path]:
    """Liste tous les SessionData.ucamcalib sous calibration_input/, triés du plus récent."""
    if not CALIBRATION_INPUT_DIR.exists():
        return []
    sessions = list(CALIBRATION_INPUT_DIR.rglob("SessionData.ucamcalib"))
    return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)


def load_calibration_session(session_path: Path) -> dict:
    """Charge une session de calibration exportée (Row*.json + Row*.png à côté du
    SessionData.ucamcalib) : chaque Row*.json fournit déjà les correspondances
    objectPoints (3D) / imagePoints (2D) détectées par l'outil d'origine, donc aucune
    redétection de damier n'est nécessaire ici.
    """
    session_meta = _read_json_maybe_utf16(session_path)
    session_dir = session_path.parent

    rows = []
    checkerboard_dims = None
    image_size = None
    for row_json_path in sorted(session_dir.glob("Row*.json")):
        row_data = _read_json_maybe_utf16(row_json_path)
        png_path = row_json_path.with_suffix(".png")
        if not png_path.exists():
            continue
        frame_bgr = cv2.imread(str(png_path))
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if image_size is None:
            image_size = (frame_rgb.shape[1], frame_rgb.shape[0])
        if checkerboard_dims is None:
            cb = row_data.get("checkerboardDimensions")
            if cb:
                checkerboard_dims = (cb["x"], cb["y"])

        object_points = [(p["x"], p["y"], p["z"]) for p in row_data["objectPoints"]["points"]]
        image_points = [(p["x"], p["y"]) for p in row_data["imagePoints"]["points"]]

        rows.append({
            "id": row_data["index"],
            "index": row_data["index"],
            "frame": frame_rgb,
            "object_points": object_points,
            "image_points": image_points,
            "source": row_json_path.name,
        })

    return {
        "session_datetime": session_meta.get("SessionDateTime"),
        "rows": rows,
        "checkerboard_dims": checkerboard_dims,
        "image_size": image_size,
    }


def draw_curvature_annotations(display_frame: np.ndarray, points_display: list) -> np.ndarray:
    """Dessine les points cliqués (numérotés) et la corde reliant le 1er et le dernier."""
    vis = display_frame.copy()
    for i, (x, y) in enumerate(points_display):
        p = (int(round(x)), int(round(y)))
        cv2.circle(vis, p, 6, (255, 0, 0), -1)
        cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if len(points_display) >= 2:
        a = tuple(int(round(v)) for v in points_display[0])
        b = tuple(int(round(v)) for v in points_display[-1])
        cv2.line(vis, a, b, (0, 180, 0), 2)
    return vis


def analyse_courbure(points: list, image_w: int, image_h: int) -> dict:
    """Port direct de l'analyse de doc/mesure_courbure.py, en coordonnées pixel natives.

    Mesure l'écart perpendiculaire maximal (flèche/sagitta) des points intermédiaires
    par rapport à la corde reliant le premier et le dernier point cliqués.
    """
    P = np.array(points, float)
    a, b = P[0], P[-1]
    d = b - a
    chord_length = float(np.linalg.norm(d))
    n = np.array([-d[1], d[0]]) / chord_length
    ecarts = (P[1:-1] - a) @ n

    i = int(np.argmax(np.abs(ecarts)))
    emax = float(ecarts[i])
    sagitta = abs(emax)

    milieu = P[len(P) // 2]
    centre = np.array([image_w / 2.0, image_h / 2.0])
    vers_centre = (centre - milieu) @ n
    sens = None
    if sagitta > 0.3:
        sens = "barillet (la ligne bombe vers l'extérieur)" if emax * vers_centre < 0 \
            else "coussinet (la ligne creuse vers le centre)"

    if sagitta < SEUIL_NEGLIGEABLE:
        verdict = "neutre"
    elif sagitta < SEUIL_A_CALIBRER:
        verdict = "zone_grise"
    else:
        verdict = "a_calibrer"

    return {
        "chord_length": chord_length,
        "chord_pct_width": 100 * chord_length / image_w,
        "ecarts": ecarts.tolist(),
        "sagitta": sagitta,
        "sagitta_pct_height": 100 * sagitta / image_h,
        "sens": sens,
        "verdict": verdict,
    }


st.set_page_config(page_title="Distorsion d'objectif (.ulens)", layout="wide")
st.title("Visualisation de la distorsion d'objectif")

tab_distortion, tab_video, tab_calib, tab_check = st.tabs(
    [
        "Distorsion d'objectif",
        "Vidéo Live (DeckLink)",
        "Calibration (OpenCV)",
        "Caméra déjà calibrée ?",
    ]
)

if "calib_captures" not in st.session_state:
    st.session_state.calib_captures = []
    st.session_state.calib_next_id = 0
    st.session_state.calib_result = None

if "imported_session" not in st.session_state:
    st.session_state.imported_session = None

if "curvature_frame" not in st.session_state:
    st.session_state.curvature_frame = None
    st.session_state.curvature_points = []
    st.session_state.curvature_last_click = None

ulens_files = sorted(DATA_DIR.glob("*.ulens"))
if not ulens_files:
    st.error(f"Aucun fichier .ulens trouvé dans {DATA_DIR}")
    st.stop()

with st.sidebar:
    st.header("Fichier lens")
    selected_file = st.selectbox(
        "Fichier .ulens", ulens_files, format_func=lambda p: p.name
    )

lens = load_ulens(str(selected_file))

focal_rows = lens["tables"]["FocalLengthTable"]
center_rows = lens["tables"]["ImageCenterTable"]
distortion_rows = lens["tables"]["DistortionTable"]

with st.sidebar:
    st.caption(
        f"**{lens['metaData']['lensInfo']['modelName']}** "
        f"(SN {lens['metaData']['lensInfo']['serialNumber']})"
    )

    row_labels = [
        f"Focus={r['FocusEncoder']:g}, Zoom={r['ZoomEncoder']:g}"
        for r in distortion_rows
    ]
    row_idx = st.selectbox(
        "Point de calibration (Focus/Zoom)",
        range(len(distortion_rows)),
        format_func=lambda i: row_labels[i],
    )

    st.header("Image de sortie")
    img_w = st.number_input(
        "Largeur (px)", min_value=1, value=int(lens["imageDimensions"]["width"])
    )
    img_h = st.number_input(
        "Hauteur (px)", min_value=1, value=int(lens["imageDimensions"]["height"])
    )

    st.header("Affichage")
    n_lines = st.slider("Densité de la grille", 4, 30, 12)
    exaggeration = st.slider("Exagération de la distorsion", 1.0, 10.0, 3.0, 0.5)

focal = focal_rows[row_idx]
dist = distortion_rows[row_idx]

fx = focal["Fx"] * img_w
fy = focal["Fy"] * img_h
k1, k2, k3 = dist["K1"], dist["K2"], dist["K3"]
p1, p2 = dist["P1"], dist["P2"]

sensor_w_mm = lens["sensorDimensions"]["width"]
focal_mm = focal["Fx"] * sensor_w_mm
hfov_deg = np.degrees(2 * np.arctan(0.5 / focal["Fx"]))

with tab_distortion:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("K1", f"{k1:.5f}")
    col2.metric("K2", f"{k2:.5f}")
    col3.metric("K3", f"{k3:.5f}")
    col4.metric("P1", f"{p1:.5f}")
    col5.metric("P2", f"{p2:.5f}")

    fig = build_distortion_figure(
        fx, fy, k1, k2, k3, p1, p2, img_w, img_h, n_lines, exaggeration, focal_mm, hfov_deg
    )
    st.pyplot(fig)

    with st.expander("Détails du fichier .ulens"):
        st.json(lens["metaData"])
        st.write("**FocalLengthTable**", focal_rows)
        st.write("**ImageCenterTable**", center_rows)
        st.write("**DistortionTable**", distortion_rows)

with tab_video:
    st.header("Aperçu vidéo — Blackmagic DeckLink")

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        st.error("ffmpeg/ffprobe introuvables dans le PATH. Installez-les pour activer l'aperçu vidéo.")
    else:
        if "capture_proc" not in st.session_state:
            st.session_state.capture_proc = None
            st.session_state.capture_running = False
            st.session_state.capture_size = None
            st.session_state.capture_fps = None
            st.session_state.capture_holder = None

        col_video, col_controls = st.columns([3, 1])

        with col_controls:
            device_name = st.text_input(
                "Périphérique DirectShow", DECKLINK_DEVICE_NAME, disabled=st.session_state.capture_running
            )
            st.caption(
                "Le format (résolution/cadence) est auto-détecté depuis le signal SDI reçu — "
                "aucun réglage manuel nécessaire."
            )

            if st.button("Démarrer", disabled=st.session_state.capture_running, type="primary"):
                proc, line_queue = start_decklink_capture(device_name)
                with st.spinner("Détection du signal SDI…"):
                    detected = detect_stream_format(line_queue)
                if detected is None:
                    proc.terminate()
                    st.error(
                        f"Aucun signal détecté sur « {device_name} » (câble débranché, "
                        "entrée SDI inactive, ou périphérique déjà utilisé par une autre application)."
                    )
                else:
                    w, h, fps = detected
                    holder = LatestFrame()
                    threading.Thread(target=_capture_frames, args=(proc, w, h, holder), daemon=True).start()
                    st.session_state.capture_proc = proc
                    st.session_state.capture_size = (w, h)
                    st.session_state.capture_fps = fps
                    st.session_state.capture_holder = holder
                    st.session_state.capture_running = True
                    st.rerun()

            if st.button("Arrêter", disabled=not st.session_state.capture_running):
                if st.session_state.capture_proc is not None:
                    st.session_state.capture_proc.terminate()
                st.session_state.capture_proc = None
                st.session_state.capture_holder = None
                st.session_state.capture_running = False
                st.rerun()

            if st.session_state.capture_running and st.session_state.capture_size is not None:
                w, h = st.session_state.capture_size
                st.metric("Signal détecté", f"{w}×{h}", f"{st.session_state.capture_fps} fps")

            st.divider()
            capture_clicked = st.button(
                "📸 Capturer cette frame",
                disabled=not st.session_state.capture_running,
                help="Ajoute l'image actuelle à la liste de captures pour la calibration OpenCV "
                "(onglet « Calibration ») ET la met à disposition dans l'onglet "
                "« Caméra déjà calibrée ? » pour le test de distorsion rapide.",
            )
            if capture_clicked and st.session_state.capture_holder is not None:
                with st.session_state.capture_holder.lock:
                    captured = st.session_state.capture_holder.frame
                if captured is not None:
                    st.session_state.calib_captures.append(
                        {"id": st.session_state.calib_next_id, "frame": captured.copy()}
                    )
                    st.session_state.calib_next_id += 1
                    st.session_state.curvature_frame = captured.copy()
                    st.session_state.curvature_points = []
                    st.session_state.curvature_last_click = None
                    st.toast(f"Frame capturée ({len(st.session_state.calib_captures)} au total)")
            st.caption(f"{len(st.session_state.calib_captures)} capture(s) pour la calibration")

        with col_video:
            frame_slot = st.empty()
            status_slot = st.empty()

            if st.session_state.capture_running and st.session_state.capture_proc is not None:
                proc = st.session_state.capture_proc
                holder = st.session_state.capture_holder
                status_slot.caption(f"Lecture en direct depuis « {device_name} »")

                stream_lost = False
                last_seq = -1
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    with holder.lock:
                        frame, seq, stopped = holder.frame, holder.seq, holder.stopped
                    if frame is not None and seq != last_seq:
                        frame_slot.image(frame, channels="RGB", use_container_width=True)
                        last_seq = seq
                    if stopped:
                        stream_lost = True
                        break
                    time.sleep(0.04)

                if stream_lost:
                    err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
                    status_slot.error(
                        "Flux vidéo interrompu (signal SDI absent, périphérique occupé, "
                        "ou format non supporté)." + (f"\n\n{err[-500:]}" if err else "")
                    )
                    st.session_state.capture_proc = None
                    st.session_state.capture_holder = None
                    st.session_state.capture_running = False
                else:
                    st.rerun()
            else:
                status_slot.info("Cliquez sur Démarrer pour afficher le flux vidéo de la carte DeckLink.")

with tab_calib:
    st.header("Calibration de la distorsion (OpenCV)")
    st.caption(
        "Dans l'onglet « Vidéo Live », démarrez le flux, présentez un damier de calibration "
        "sous plusieurs angles et cliquez sur « 📸 Capturer cette frame » à chaque prise. "
        "Revenez ensuite ici pour lancer la calibration."
    )

    col_focus, col_zoom, col_fstop, col_cols, col_rows = st.columns(5)
    with col_focus:
        calib_focus = st.number_input(
            "Focus (distance, cm)", min_value=0.0, value=100.0,
            help="Distance de mise au point réelle, en cm, au moment des captures.",
        )
    with col_zoom:
        calib_zoom = st.number_input(
            "Zoom (focale, mm)", min_value=0.0, value=28.0,
            help="Focale de l'objectif en mm au moment des captures "
            "(pour un objectif fixe, sa focale nominale).",
        )
    with col_fstop:
        calib_fstop = st.number_input(
            "Ouverture (f-stop)", min_value=0.0, value=8.0, step=0.1,
            help="Ouverture utilisée pendant la capture. N'affecte pas le calcul de "
            "K1-P2/Fx/Fy/Cx/Cy (la distorsion géométrique dépend du focus/zoom, pas du "
            "diaphragme) — enregistrée uniquement comme métadonnée de traçabilité dans "
            "le fichier .ulens, comme dans l'exemple « Nkkor 28mm 4k 16fstop.ulens ». "
            "Mettez 0 pour ne rien enregistrer.",
        )
    with col_cols:
        pattern_cols = st.number_input("Coins internes — colonnes", min_value=3, value=13)
    with col_rows:
        pattern_rows = st.number_input("Coins internes — lignes", min_value=3, value=9)

    col_sw, col_sh = st.columns(2)
    with col_sw:
        save_sensor_w = st.number_input(
            "Largeur capteur (mm)", min_value=0.1, value=float(sensor_w_mm)
        )
    with col_sh:
        save_sensor_h = st.number_input(
            "Hauteur capteur (mm)", min_value=0.1, value=float(lens["sensorDimensions"]["height"])
        )

    col_fix1, col_fix2 = st.columns(2)
    with col_fix1:
        fix_distortion = st.checkbox(
            "Distorsion déjà corrigée par la caméra (ex : Nikon ZR + Auto distortion control ON)",
            value=False,
            help="Ne calibre que Fx, Fy, Cx, Cy (K1, K2, K3, P1, P2 fixés à 0). À cocher si "
            "l'onglet « Caméra déjà calibrée ? » a montré des lignes droites : la correction "
            "interne change quand même le champ de vision effectif, donc la focale/le centre "
            "image doivent être mesurés même sans distorsion résiduelle à corriger.",
        )
    with col_fix2:
        fix_principal_point = st.checkbox(
            "Forcer le centre image à Cx=0.5, Cy=0.5",
            value=False,
            help="Fixe le point principal au centre géométrique de l'image au lieu de le "
            "laisser libre (CALIB_FIX_PRINCIPAL_POINT). Utile avec peu de vues ou une "
            "couverture incomplète, où Cx/Cy sont mal contraints (cf. test de stabilité). "
            "Cumulable avec l'option distorsion déjà corrigée.",
        )

    live_result = st.session_state.calib_result
    live_errors_by_position = {}
    if (
        live_result is not None and live_result.get("success")
        and live_result.get("source") == "live"
        and live_result.get("total_count") == len(st.session_state.calib_captures)
    ):
        live_errors_by_position = dict(zip(live_result["used_indices"], live_result["per_view_errors"]))

    st.subheader(f"Captures ({len(st.session_state.calib_captures)})")
    if st.session_state.calib_captures:
        thumb_cols = st.columns(6)
        removed_id = None
        for i, capture in enumerate(st.session_state.calib_captures):
            with thumb_cols[i % 6]:
                st.image(capture["frame"], channels="RGB", use_container_width=True)
                if live_errors_by_position:
                    err = live_errors_by_position.get(i)
                    st.caption(format_reprojection_error(err) if err is not None else "⬜ Damier non détecté")
                if st.button("🗑 Supprimer", key=f"del_capture_{capture['id']}"):
                    removed_id = capture["id"]
        if removed_id is not None:
            st.session_state.calib_captures = [
                c for c in st.session_state.calib_captures if c["id"] != removed_id
            ]
            st.rerun()

        if st.button("Tout effacer"):
            st.session_state.calib_captures = []
            st.session_state.calib_result = None
            st.rerun()
    else:
        st.info(
            "Aucune capture pour l'instant. Allez dans l'onglet « Vidéo Live » pour capturer "
            "des images d'un damier de calibration sous différents angles (10-20 images conseillées)."
        )

    if st.button(
        "Lancer la calibration OpenCV",
        type="primary",
        disabled=len(st.session_state.calib_captures) < 3,
    ):
        with st.spinner("Détection des coins du damier et calibration…"):
            st.session_state.calib_result = run_camera_calibration(
                [c["frame"] for c in st.session_state.calib_captures],
                (int(pattern_cols), int(pattern_rows)),
                fix_distortion=fix_distortion,
                fix_principal_point=fix_principal_point,
            )

    if st.button(
        "🔬 Tester la stabilité de Cx/Cy (1ère moitié vs 2e moitié)",
        disabled=len(st.session_state.calib_captures) < 6,
        help="Recalibre séparément sur la première et la seconde moitié des captures "
        "(dans l'ordre de capture) pour vérifier que le centre image (Cx, Cy) reste stable.",
        key="stability_live_btn",
    ):
        frames_all = [c["frame"] for c in st.session_state.calib_captures]
        mid = len(frames_all) // 2
        with st.spinner("Calibration sur chaque moitié…"):
            half1 = run_camera_calibration(
                frames_all[:mid], (int(pattern_cols), int(pattern_rows)),
                fix_distortion, fix_principal_point,
            )
            half2 = run_camera_calibration(
                frames_all[mid:], (int(pattern_cols), int(pattern_rows)),
                fix_distortion, fix_principal_point,
            )
        st.session_state.stability_result_live = {"half1": half1, "half2": half2}

    if st.session_state.get("stability_result_live"):
        st.subheader("Stabilité de Cx/Cy — captures live")
        render_stability_comparison(st.session_state.stability_result_live)

    st.divider()
    with st.expander("📂 Importer une session de calibration existante"):
        st.caption(
            "Charge les correspondances déjà détectées (Row*.json + Row*.png) d'une session "
            "exportée sous calibration_input/ — aucune redétection de damier nécessaire."
        )
        sessions = find_calibration_sessions()
        if not sessions:
            st.info(f"Aucune session trouvée sous {CALIBRATION_INPUT_DIR}")
        else:
            session_choice = st.selectbox(
                "Session",
                sessions,
                format_func=lambda p: str(p.parent.relative_to(CALIBRATION_INPUT_DIR)),
            )
            if st.button("📥 Importer cette session"):
                with st.spinner("Chargement de la session…"):
                    st.session_state.imported_session = load_calibration_session(session_choice)
                st.rerun()

        session = st.session_state.imported_session
        if session is not None:
            n_rows = len(session["rows"])
            cb = session["checkerboard_dims"]
            st.write(
                f"**{n_rows} capture(s) importée(s)** — session du {session['session_datetime']}"
                + (f", damier {cb[0]}×{cb[1]} points" if cb else "")
            )
            session_result = st.session_state.calib_result
            session_errors_by_id = {}
            if (
                session_result is not None and session_result.get("success")
                and session_result.get("source") == "session"
                and session_result.get("total_count") == len(session["rows"])
            ):
                session_errors_by_id = dict(
                    zip(session_result["used_indices"], session_result["per_view_errors"])
                )

            if session["rows"]:
                thumb_cols = st.columns(6)
                removed_idx = None
                for i, row in enumerate(session["rows"]):
                    with thumb_cols[i % 6]:
                        st.image(row["frame"], channels="RGB", use_container_width=True,
                                  caption=row["source"])
                        if session_errors_by_id:
                            err = session_errors_by_id.get(row["id"])
                            st.caption(format_reprojection_error(err) if err is not None else "⬜ Non utilisé")
                        if st.button("🗑 Supprimer", key=f"del_session_row_{row['id']}"):
                            removed_idx = row["id"]
                if removed_idx is not None:
                    session["rows"] = [r for r in session["rows"] if r["id"] != removed_idx]
                    st.rerun()

            col_launch, col_clear = st.columns(2)
            with col_launch:
                if st.button(
                    "Lancer la calibration sur la session importée",
                    type="primary",
                    disabled=len(session["rows"]) < 3,
                ):
                    with st.spinner("Calibration à partir des points importés…"):
                        st.session_state.calib_result = run_camera_calibration_from_session(
                            session["rows"], session["image_size"],
                            fix_distortion=fix_distortion, fix_principal_point=fix_principal_point,
                        )
            with col_clear:
                if st.button("Oublier la session importée"):
                    st.session_state.imported_session = None
                    st.rerun()

            if st.button(
                "🔬 Tester la stabilité de Cx/Cy (1ère moitié vs 2e moitié)",
                disabled=len(session["rows"]) < 6,
                help="Recalibre séparément sur la première et la seconde moitié des vues "
                "(dans l'ordre d'index) pour vérifier que le centre image (Cx, Cy) reste stable.",
                key="stability_session_btn",
            ):
                rows_all = session["rows"]
                mid = len(rows_all) // 2
                with st.spinner("Calibration sur chaque moitié…"):
                    half1 = run_camera_calibration_from_session(
                        rows_all[:mid], session["image_size"], fix_distortion, fix_principal_point
                    )
                    half2 = run_camera_calibration_from_session(
                        rows_all[mid:], session["image_size"], fix_distortion, fix_principal_point
                    )
                st.session_state.stability_result_session = {"half1": half1, "half2": half2}

            if st.session_state.get("stability_result_session"):
                st.subheader("Stabilité de Cx/Cy — session importée")
                render_stability_comparison(st.session_state.stability_result_session)

    result = st.session_state.calib_result
    if result is not None:
        if not result["success"]:
            st.error(
                f"Damier détecté dans seulement {result['detected_count']}/{result['total_count']} "
                f"image(s) — 3 minimum requis. Vérifiez le nombre de coins internes "
                f"({int(pattern_cols)}×{int(pattern_rows)}) ou ajoutez des captures plus nettes."
            )
        else:
            st.success(
                f"Calibration réussie — damier détecté dans {result['detected_count']}/"
                f"{result['total_count']} image(s)."
            )
            if result.get("fix_distortion") and result.get("fix_principal_point"):
                st.info(
                    "Mode « distorsion déjà corrigée » + « centre image fixé » : K1, K2, K3, "
                    "P1, P2 sont fixés à 0 et Cx=Cy=0.5 ci-dessous — seuls Fx, Fy ont été calculés."
                )
            elif result.get("fix_distortion"):
                st.info(
                    "Mode « distorsion déjà corrigée » : K1, K2, K3, P1, P2 sont fixés à 0 "
                    "ci-dessous — seuls Fx, Fy, Cx, Cy ont été calculés."
                )
            elif result.get("fix_principal_point"):
                st.info(
                    "Mode « centre image fixé » : Cx=Cy=0.5 ci-dessous — K1-P2, Fx, Fy ont "
                    "été calculés avec le point principal verrouillé au centre."
                )

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("K1", f"{result['k1']:.5f}")
            col2.metric("K2", f"{result['k2']:.5f}")
            col3.metric("K3", f"{result['k3']:.5f}")
            col4.metric("P1", f"{result['p1']:.5f}")
            col5.metric("P2", f"{result['p2']:.5f}")

            calib_w, calib_h = result["image_size"]
            col_fx, col_fy, col_cx, col_cy = st.columns(4)
            col_fx.metric("Fx", f"{result['fx_px']:.1f} px")
            col_fy.metric("Fy", f"{result['fy_px']:.1f} px")
            col_cx.metric("Cx", f"{result['cx_px']:.1f} px")
            col_cy.metric("Cy", f"{result['cy_px']:.1f} px")
            st.caption(
                "Normalisé (tel qu'enregistré dans le .ulens) : "
                f"Fx={result['fx_px'] / calib_w:.4f}, Fy={result['fy_px'] / calib_h:.4f}, "
                f"Cx={result['cx_px'] / calib_w:.4f}, Cy={result['cy_px'] / calib_h:.4f}"
            )

            col6, col7 = st.columns(2)
            col6.metric("Erreur de reprojection (RMS)", f"{result['rms']:.3f} px")
            col7.metric("Résolution calibrée", f"{calib_w}×{calib_h}")

            fig_err, ax_err = plt.subplots(figsize=(10, 3))
            view_labels = [str(i + 1) for i in range(len(result["per_view_errors"]))]
            ax_err.bar(view_labels, result["per_view_errors"], color="steelblue")
            ax_err.axhline(
                result["rms"], color="crimson", linestyle="--",
                label=f"RMS global = {result['rms']:.3f} px",
            )
            ax_err.set_xlabel("Image de calibration")
            ax_err.set_ylabel("Erreur de reprojection (px)")
            ax_err.set_title("Erreur de reprojection par image")
            ax_err.legend()
            st.pyplot(fig_err)

            calib_hfov_deg = np.degrees(2 * np.arctan(calib_w / (2 * result["fx_px"])))
            calib_focal_mm = (result["fx_px"] / calib_w) * save_sensor_w
            fig_calib_dist = build_distortion_figure(
                result["fx_px"], result["fy_px"],
                result["k1"], result["k2"], result["k3"], result["p1"], result["p2"],
                calib_w, calib_h, 12, 3.0, calib_focal_mm, calib_hfov_deg,
            )
            st.pyplot(fig_calib_dist)

            st.info(
                f"💡 Focale déduite de la calibration (à partir de la largeur capteur "
                f"renseignée ci-dessus) : **{calib_focal_mm:.1f} mm**. Ajustez le champ "
                f"« Zoom (focale, mm) » en haut si besoin avant d'enregistrer."
            )

            st.divider()
            st.subheader("Enregistrer en fichier .ulens")
            default_name = f"Calibration OpenCV {datetime.now():%Y%m%d-%H%M}"
            col_name, col_serial = st.columns(2)
            with col_name:
                lens_name = st.text_input("Nom du lens file", default_name)
            with col_serial:
                lens_serial = st.text_input("Numéro de série", "")

            safe_name = re.sub(r'[\\/:*?"<>|]', "_", lens_name).strip() or "lens_calibration"
            default_filename = f"{safe_name}{f' {calib_fstop:g}fstop' if calib_fstop > 0 else ''}.ulens"
            out_filename = st.text_input("Nom de fichier", default_filename)

            if st.button("💾 Enregistrer le fichier .ulens"):
                new_lens = build_ulens_json(
                    lens_name, lens_serial, calib_focus, calib_zoom, result,
                    save_sensor_w, save_sensor_h, fstop=calib_fstop,
                )
                out_path = DATA_DIR / out_filename
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(new_lens, f, indent=1)
                st.success(f"Fichier enregistré : {out_path}")

with tab_check:
    st.header("La caméra est-elle déjà calibrée ?")
    st.caption(
        "Test rapide de distorsion (inspiré de doc/mesure_courbure.py) : utile pour savoir si "
        "la correction de distorsion embarquée du Nikon ZR (Nikkor) neutralise déjà les lignes "
        "droites dans le format tourné, ou s'il faut une calibration complète."
    )

    grille_path = INPUT_DIR / "grille_3840x2160.png"
    col_grid, col_howto = st.columns([1, 2])
    with col_grid:
        if grille_path.exists():
            st.image(str(grille_path), caption="grille_3840x2160.png", use_container_width=True)
            with open(grille_path, "rb") as f:
                st.download_button(
                    "⬇ Télécharger la grille", f, file_name=grille_path.name, mime="image/png"
                )
        else:
            st.error(f"Grille introuvable : {grille_path}")
    with col_howto:
        st.markdown(
            "**Protocole :**\n"
            "1. Affichez `grille_3840x2160.png` en plein écran sur un moniteur "
            "(ou imprimez-la bien à plat) et filmez-la avec la caméra/objectif à tester, "
            "dans le format exact de production.\n"
            "2. Démarrez le flux dans l'onglet « Vidéo Live », cadrez la grille, puis "
            "capturez une frame ci-dessous.\n"
            "3. Cliquez 5 à 9 points le long d'**une ligne la plus périphérique possible** "
            "(le cadre extérieur en haut ou en bas de la grille), d'un bord à l'autre de "
            "l'image : c'est là que la distorsion résiduelle est la plus visible.\n"
            "4. Le verdict (lignes droites / zone grise / lignes courbes) s'affiche "
            "automatiquement dès 3 points cliqués."
        )

    st.divider()

    if st.session_state.capture_running and st.session_state.capture_holder is not None:
        if st.button("📸 Capturer cette frame pour le test", type="primary"):
            with st.session_state.capture_holder.lock:
                captured = st.session_state.capture_holder.frame
            if captured is not None:
                st.session_state.curvature_frame = captured.copy()
                st.session_state.curvature_points = []
                st.session_state.curvature_last_click = None
                st.rerun()
    else:
        st.info("Démarrez le flux dans l'onglet « Vidéo Live » pour pouvoir capturer une frame.")

    if st.session_state.curvature_frame is not None:
        frame = st.session_state.curvature_frame
        frame_h, frame_w = frame.shape[:2]

        col_reset1, col_reset2, col_reset3 = st.columns(3)
        with col_reset1:
            if st.button("↩ Annuler dernier point", disabled=not st.session_state.curvature_points):
                st.session_state.curvature_points.pop()
                st.session_state.curvature_last_click = None
                st.rerun()
        with col_reset2:
            if st.button("🗑 Effacer les points", disabled=not st.session_state.curvature_points):
                st.session_state.curvature_points = []
                st.session_state.curvature_last_click = None
                st.rerun()
        with col_reset3:
            if st.button("🔄 Nouvelle capture"):
                st.session_state.curvature_frame = None
                st.session_state.curvature_points = []
                st.session_state.curvature_last_click = None
                st.rerun()

        display_scale = min(1.0, 1400 / frame_w)
        disp_w, disp_h = int(frame_w * display_scale), int(frame_h * display_scale)
        display_frame = cv2.resize(frame, (disp_w, disp_h)) if display_scale < 1.0 else frame
        points_display = [(x * display_scale, y * display_scale) for x, y in st.session_state.curvature_points]
        annotated = draw_curvature_annotations(display_frame, points_display)

        st.caption(f"{len(st.session_state.curvature_points)} point(s) — cliquez sur l'image pour en ajouter")
        click = streamlit_image_coordinates(annotated, key="curvature_click")
        if click is not None:
            new_point_display = (click["x"], click["y"])
            if new_point_display != st.session_state.curvature_last_click:
                st.session_state.curvature_last_click = new_point_display
                original_point = (click["x"] / display_scale, click["y"] / display_scale)
                st.session_state.curvature_points.append(original_point)
                st.rerun()

        n_pts = len(st.session_state.curvature_points)
        if n_pts < 3:
            st.info("Ajoutez au moins 3 points (les deux extrémités + un point intermédiaire) "
                     "pour calculer la flèche.")
        else:
            result_c = analyse_courbure(st.session_state.curvature_points, frame_w, frame_h)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Flèche maximale (sagitta)", f"{result_c['sagitta']:.2f} px")
            col_b.metric("En % de la hauteur", f"{result_c['sagitta_pct_height']:.3f} %")
            col_c.metric("Longueur de la corde", f"{result_c['chord_length']:.0f} px "
                         f"({result_c['chord_pct_width']:.0f} % largeur)")

            if result_c["verdict"] == "neutre":
                st.success(
                    "✅ **Lignes droites — distorsion déjà neutralisée.** "
                    "Ne calibrer que les intrinsèques (Fx, Fy, Cx, Cy) + le nodal offset ; "
                    "pas besoin de dé-distorsion supplémentaire."
                )
            elif result_c["verdict"] == "zone_grise":
                st.warning(
                    "⚠️ **Zone grise — distorsion résiduelle faible mais mesurable.** "
                    "Une calibration complète (onglet « Calibration ») est recommandée ; "
                    "un modèle k1,k2 suffira probablement."
                )
            else:
                st.error(
                    "❌ **Lignes courbes — distorsion à calibrer entièrement.** "
                    "Utilisez l'onglet « Calibration (OpenCV) » avec un modèle Brown-Conrady "
                    "complet (k1,k2,k3,p1,p2)."
                )

            if result_c["sens"]:
                st.caption(f"Sens de la distorsion : {result_c['sens']}")

            fig_ecarts, ax_ecarts = plt.subplots(figsize=(10, 3))
            point_labels = [str(i + 2) for i in range(len(result_c["ecarts"]))]
            ax_ecarts.bar(point_labels, result_c["ecarts"], color="steelblue")
            ax_ecarts.axhline(0, color="black", linewidth=0.8)
            ax_ecarts.axhline(SEUIL_NEGLIGEABLE, color="green", linestyle="--", linewidth=1,
                               label=f"seuil négligeable = {SEUIL_NEGLIGEABLE:g} px")
            ax_ecarts.axhline(-SEUIL_NEGLIGEABLE, color="green", linestyle="--", linewidth=1)
            ax_ecarts.axhline(SEUIL_A_CALIBRER, color="crimson", linestyle="--", linewidth=1,
                               label=f"seuil à calibrer = {SEUIL_A_CALIBRER:g} px")
            ax_ecarts.axhline(-SEUIL_A_CALIBRER, color="crimson", linestyle="--", linewidth=1)
            ax_ecarts.set_xlabel("Point intermédiaire (n° sur l'image)")
            ax_ecarts.set_ylabel("Écart perpendiculaire à la corde (px)")
            ax_ecarts.set_title("Écart de chaque point par rapport à la ligne idéale")
            ax_ecarts.legend(fontsize=8)
            st.pyplot(fig_ecarts)

            st.caption(
                "Rappel : refaire cette mesure pour chaque format candidat "
                "(RAW N-RAW/R3D NE vs H.265 ADC ON vs H.265 ADC OFF)."
            )
