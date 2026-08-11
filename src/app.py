"""Visualisation de la distorsion d'objectif à partir de fichiers .ulens."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DECKLINK_DEVICE_NAME = "Decklink Video Capture"
VIDEO_PRESETS = {
    "1920x1080 @25": (1920, 1080, "25"),
    "1920x1080 @29.97": (1920, 1080, "29.97"),
    "1920x1080 @50": (1920, 1080, "50"),
    "1920x1080 @59.94": (1920, 1080, "59.9402"),
    "3840x2160 @25 (UHD)": (3840, 2160, "25"),
    "3840x2160 @29.97 (UHD)": (3840, 2160, "29.97"),
    "3840x2160 @30 (UHD)": (3840, 2160, "30"),
    "3840x2160 @24 (UHD)": (3840, 2160, "24"),
}


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


def start_decklink_capture(device_name: str, width: int, height: int, framerate: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "dshow",
        "-video_size", f"{width}x{height}",
        "-framerate", str(framerate),
        "-vcodec", "v210",
        "-i", f"video={device_name}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-an", "-",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=width * height * 3 * 2,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def read_decklink_frame(proc: subprocess.Popen, width: int, height: int):
    frame_size = width * height * 3
    buf = proc.stdout.read(frame_size)
    if len(buf) < frame_size:
        return None
    frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
    return frame[:, :, ::-1]  # BGR -> RGB


st.set_page_config(page_title="Distorsion d'objectif (.ulens)", layout="wide")
st.title("Visualisation de la distorsion d'objectif")

tab_distortion, tab_video = st.tabs(["Distorsion d'objectif", "Vidéo Live (DeckLink)"])

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

    if shutil.which("ffmpeg") is None:
        st.error("ffmpeg est introuvable dans le PATH. Installez-le pour activer l'aperçu vidéo.")
    else:
        if "capture_proc" not in st.session_state:
            st.session_state.capture_proc = None
            st.session_state.capture_running = False
            st.session_state.capture_size = None

        col_video, col_controls = st.columns([3, 1])

        with col_controls:
            device_name = st.text_input(
                "Périphérique DirectShow", DECKLINK_DEVICE_NAME, disabled=st.session_state.capture_running
            )
            preset_label = st.selectbox(
                "Format d'entrée", list(VIDEO_PRESETS.keys()), disabled=st.session_state.capture_running
            )
            width, height, framerate = VIDEO_PRESETS[preset_label]

            if st.button("Démarrer", disabled=st.session_state.capture_running, type="primary"):
                st.session_state.capture_proc = start_decklink_capture(device_name, width, height, framerate)
                st.session_state.capture_size = (width, height)
                st.session_state.capture_running = True
                st.rerun()

            if st.button("Arrêter", disabled=not st.session_state.capture_running):
                if st.session_state.capture_proc is not None:
                    st.session_state.capture_proc.terminate()
                st.session_state.capture_proc = None
                st.session_state.capture_running = False
                st.rerun()

        with col_video:
            frame_slot = st.empty()
            status_slot = st.empty()

            if st.session_state.capture_running and st.session_state.capture_proc is not None:
                proc = st.session_state.capture_proc
                w, h = st.session_state.capture_size
                status_slot.caption(f"Lecture en direct — {w}x{h} @ {framerate} fps depuis « {device_name} »")

                stream_lost = False
                for _ in range(15):
                    if proc.poll() is not None:
                        stream_lost = True
                        break
                    frame = read_decklink_frame(proc, w, h)
                    if frame is None:
                        stream_lost = True
                        break
                    frame_slot.image(frame, channels="RGB", use_container_width=True)

                if stream_lost:
                    err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
                    status_slot.error(
                        "Flux vidéo interrompu (signal SDI absent, périphérique occupé, "
                        "ou format non supporté)." + (f"\n\n{err[-500:]}" if err else "")
                    )
                    st.session_state.capture_proc = None
                    st.session_state.capture_running = False
                else:
                    st.rerun()
            else:
                status_slot.info("Cliquez sur Démarrer pour afficher le flux vidéo de la carte DeckLink.")
