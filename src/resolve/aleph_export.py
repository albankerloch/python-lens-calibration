"""Rendu depuis DaVinci Resolve d'un extrait de la timeline courante, aux contraintes Aleph 2.0.

Contraintes Runway (https://docs.dev.runwayml.com/assets/inputs/#videos) :
 - résolution conservée jusqu'à 1080p : la sortie est réduite pour tenir dans 1920×1080 ;
 - 2 à 30 s, 30 i/s maximum ;
 - upload éphémère limité à 200 Mo.

Utilisé par src/runway_app.py ; Resolve doit être lancé avec
Préférences > Système > Général > External scripting using = Local.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ALEPH_MAX_LONG_SIDE = 1920
ALEPH_MAX_SHORT_SIDE = 1080
ALEPH_MIN_SECONDS = 2.0
ALEPH_MAX_SECONDS = 30.0
ALEPH_MAX_FPS = 30.0
UPLOAD_LIMIT_MB = 200

H264_BITRATE_KBPS = 40_000  # 30 s ≈ 150 Mo, marge sous les 200 Mo

# Les uploads précédents (Compositing_Upload_V8.mp4) étaient en AV1 10 bits NVIDIA, accepté par Runway,
# mais SetCurrentRenderFormatAndCodec refuse l'AV1 dans Resolve 21.1 (mp4 comme mkv). On passe donc
# par un preset Deliver créé à la main : MP4 + « AV1 10-bit - NVIDIA », enregistré sous AV1_PRESET.
AV1_PRESET = "Aleph AV1"

# libellé -> (format, codecs par ordre de préférence, débit approx. en Mb/s à 1080p 29.97,
#             EncodingProfile, preset Deliver à charger à la place du codec)
CODEC_CHOICES = {
    f"AV1 10 bits MP4 (preset « {AV1_PRESET} »)": ("mp4", ["AV1YUV420_10_NVIDIA"], 30, None, AV1_PRESET),
    "H.265 10 bits MP4": ("mp4", ["H265_NVIDIA", "H265"], H264_BITRATE_KBPS / 1000, "Main10", None),
    "H.264 8 bits MP4": ("mp4", ["H264_NVIDIA", "H264"], H264_BITRATE_KBPS / 1000, "High", None),
    "ProRes 422 LT": ("mov", ["ProRes422LT"], 102, None, None),
    "ProRes 422": ("mov", ["ProRes422"], 147, None, None),
    "ProRes 422 HQ": ("mov", ["ProRes422HQ"], 220, None, None),
}
DEFAULT_CODEC = f"AV1 10 bits MP4 (preset « {AV1_PRESET} »)"
# « Same as Project » laisse la transfer non tagguée ; V8 était tagué bt709 (Aleph veut du SDR Rec.709).
GAMMA_TAG = "Rec.709"
BITRATE_SCALED_CODECS = {"ProRes 422 LT", "ProRes 422", "ProRes 422 HQ"}


def connect():
    """Objet Resolve de l'instance locale, ou RuntimeError si elle est injoignable."""
    modules = os.path.join(
        os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
        r"Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
    )
    if modules not in sys.path:
        sys.path.append(modules)
    try:
        import DaVinciResolveScript as dvr
    except ImportError as exc:
        raise RuntimeError(f"Module DaVinciResolveScript introuvable ({exc}).") from exc
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError(
            "Resolve injoignable : lancer DaVinci Resolve et régler "
            "Préférences > Système > Général > External scripting using = Local."
        )
    return resolve


def current_timeline(resolve):
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        raise RuntimeError("Aucune timeline ouverte dans Resolve.")
    return project, timeline


def list_timelines(resolve) -> tuple[list[str], str | None]:
    """Noms des timelines du projet courant (ordre du projet) et nom de la timeline ouverte."""
    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        raise RuntimeError("Aucun projet ouvert dans Resolve.")
    names = [project.GetTimelineByIndex(i).GetName() for i in range(1, project.GetTimelineCount() + 1)]
    current = project.GetCurrentTimeline()
    return names, current.GetName() if current else None


def get_timeline(resolve, name: str | None = None):
    """(projet, timeline) : la timeline nommée `name`, ou la timeline ouverte si `name` est None."""
    if name is None:
        return current_timeline(resolve)
    project = resolve.GetProjectManager().GetCurrentProject()
    for i in range(1, (project.GetTimelineCount() if project else 0) + 1):
        timeline = project.GetTimelineByIndex(i)
        if timeline.GetName() == name:
            return project, timeline
    raise RuntimeError(f"Timeline « {name} » introuvable dans le projet.")


def timeline_info(resolve, name: str | None = None) -> dict:
    project, timeline = get_timeline(resolve, name)
    fps = float(timeline.GetSetting("timelineFrameRate"))
    start, end = timeline.GetStartFrame(), timeline.GetEndFrame()
    marks = (timeline.GetMarkInOut() or {}).get("video") or {}
    return {
        "project": project.GetName(),
        "timeline": timeline.GetName(),
        "fps": fps,
        "width": int(timeline.GetSetting("timelineResolutionWidth")),
        "height": int(timeline.GetSetting("timelineResolutionHeight")),
        "start_frame": start,
        "frames": end - start,
        # repères In/Out relatifs au début de la timeline (Out inclus)
        "mark_in": marks.get("in"),
        "mark_out": marks.get("out"),
    }


def aleph_size(width: int, height: int) -> tuple[int, int]:
    """Plus grande taille paire, au même ratio, qui tient dans 1920×1080 (dans les deux orientations)."""
    long_side, short_side = max(width, height), min(width, height)
    scale = min(1.0, ALEPH_MAX_LONG_SIDE / long_side, ALEPH_MAX_SHORT_SIDE / short_side)
    return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2


def estimated_size_mb(choice: str, width: int, height: int, fps: float, seconds: float) -> float:
    mbps = CODEC_CHOICES[choice][2]
    if choice in BITRATE_SCALED_CODECS:
        mbps *= (width * height) / (1920 * 1080) * fps / 29.97
    return mbps * seconds / 8


def render_extract(
    resolve,
    start_offset: int,
    n_frames: int,
    target_dir: Path,
    name: str,
    choice: str = DEFAULT_CODEC,
    on_progress=None,
    poll_seconds: float = 0.5,
    timeline_name: str | None = None,
    check_duration: bool = True,
) -> Path:
    """Rend `n_frames` images de la timeline `timeline_name` (par défaut la timeline ouverte) à partir
    de `start_offset` (relatif à son début), sans audio, à la taille Aleph. Bloque jusqu'à la fin du
    rendu et renvoie le fichier produit. `on_progress(pourcentage)` est appelé pendant le rendu.
    Resolve ne rend que la timeline ouverte : elle est ouverte le temps du rendu, puis l'ancienne est rétablie.
    `check_duration=False` lève la limite 2-30 s (export complet destiné à être découpé ensuite)."""
    project, timeline = get_timeline(resolve, timeline_name)
    info = timeline_info(resolve, timeline_name)
    if info["fps"] > ALEPH_MAX_FPS + 1e-3:
        raise ValueError(
            f"Timeline à {info['fps']:g} i/s : Aleph accepte 30 i/s maximum. "
            "Créez une timeline à 25 ou 30 i/s (Timeline Settings sans « Use project settings »)."
        )
    seconds = n_frames / info["fps"]
    if check_duration and not ALEPH_MIN_SECONDS <= seconds <= ALEPH_MAX_SECONDS + 1e-6:
        raise ValueError(f"Durée {seconds:.2f} s hors des limites Aleph (2 à 30 s).")
    if start_offset < 0 or start_offset + n_frames > info["frames"]:
        raise ValueError("Plage hors de la timeline.")

    fmt, codecs, _, profile, preset = CODEC_CHOICES[choice]
    if preset:
        current = project.GetCurrentRenderFormatAndCodec() if project.LoadRenderPreset(preset) else {}
        # L'API ne sait pas non plus lire le codec AV1 : il revient vide après chargement du preset.
        if current.get("format") != fmt or current.get("codec") not in (*codecs, ""):
            raise RuntimeError(
                f"Preset Deliver « {preset} » absent ou incorrect ({current or 'introuvable'}). "
                f"Dans Resolve > Deliver : Format MP4, Codec « AV1 10-bit - NVIDIA », "
                f"puis ⋯ > Save As New Preset sous le nom « {preset} »."
            )
    else:
        available = set((project.GetRenderCodecs(fmt) or {}).values())
        codec = next((c for c in codecs if c in available), None)
        if codec is None or not project.SetCurrentRenderFormatAndCodec(fmt, codec):
            raise RuntimeError(f"Codec {choice} indisponible dans ce Resolve.")

    width, height = aleph_size(info["width"], info["height"])
    target_dir.mkdir(parents=True, exist_ok=True)
    previous = project.GetCurrentTimeline()
    if not project.SetCurrentTimeline(timeline):
        raise RuntimeError(f"Impossible d'ouvrir la timeline « {info['timeline']} » dans Resolve.")
    saved_marks = timeline.GetMarkInOut() or {}  # MarkIn/MarkOut du rendu écrasent les repères de la timeline
    try:
        return _render(project, info, start_offset, n_frames, target_dir, name, fmt, profile,
                       width, height, on_progress, poll_seconds)
    finally:
        _restore_marks(timeline, saved_marks)
        if previous is not None:
            project.SetCurrentTimeline(previous)


def render_full_timeline(
    resolve,
    timeline_name: str,
    target_dir: Path,
    name: str,
    choice: str = DEFAULT_CODEC,
    on_progress=None,
) -> Path:
    """Rend toute la timeline `timeline_name` à la taille Aleph, quelle que soit sa durée."""
    info = timeline_info(resolve, timeline_name)
    return render_extract(resolve, 0, info["frames"], target_dir, name, choice, on_progress,
                          timeline_name=timeline_name, check_duration=False)


def _restore_marks(timeline, marks: dict) -> None:
    # SetMarkInOut n'accepte que le type « all » (video/audio séparés refusés en 21.1).
    timeline.ClearMarkInOut()
    rng = marks.get("video") or marks.get("audio") or {}
    if "in" in rng and "out" in rng:
        timeline.SetMarkInOut(rng["in"], rng["out"])


def _render(project, info, start_offset, n_frames, target_dir, name, fmt, profile, width, height,
            on_progress, poll_seconds) -> Path:
    first = info["start_frame"] + start_offset
    project.SetCurrentRenderMode(1)  # Single clip : un seul fichier, même si le preset est en Individual clips
    # Resolve 21 refuse tout le dict si une seule clé est refusée : réglages appliqués un par un.
    # (SelectAllFrames=False, ExportAlpha et ReplaceExistingFilesInPlace sont refusés : MarkIn/MarkOut
    # suffisent à limiter la plage, et l'ancien fichier est supprimé avant le rendu.)
    settings = {
        "MarkIn": first,
        "MarkOut": first + n_frames - 1,
        "TargetDir": str(target_dir),
        "CustomName": name,
        "UseUniqueFilenames": False,
        "ExportVideo": True,
        "ExportAudio": False,
        "FormatWidth": width,
        "FormatHeight": height,
        "GammaTag": GAMMA_TAG,
    }
    if profile:
        settings["EncodingProfile"] = profile
    refused = [k for k, v in settings.items() if not project.SetRenderSettings({k: v})]
    if refused:
        raise RuntimeError(f"Resolve a refusé les réglages de rendu : {', '.join(refused)}.")
    if fmt == "mp4":
        project.SetRenderSettings({"VideoQuality": H264_BITRATE_KBPS})  # refusé par certains encodeurs
    (target_dir / f"{name}.{fmt}").unlink(missing_ok=True)

    job_id = project.AddRenderJob()
    if not job_id:
        raise RuntimeError("Impossible d'ajouter le job de rendu.")
    try:
        job = next((j for j in project.GetRenderJobList() if j.get("JobId") == job_id), {})
        if not project.StartRendering([job_id]):
            raise RuntimeError("Le rendu n'a pas démarré.")
        while True:
            status = project.GetRenderJobStatus(job_id) or {}
            state = status.get("JobStatus", "")
            if on_progress:
                on_progress(int(status.get("CompletionPercentage", 0)))
            if state == "Complete":
                break
            if state in {"Failed", "Cancelled", "Background Render Cancelled"}:
                raise RuntimeError(f"Rendu {state} : {status.get('Error', '')}".strip(" :"))
            time.sleep(poll_seconds)
    finally:
        if not project.IsRenderingInProgress():
            project.DeleteRenderJob(job_id)

    out = Path(job.get("TargetDir") or target_dir) / (job.get("OutputFilename") or f"{name}.{fmt}")
    if not out.is_file():
        raise RuntimeError(f"Rendu terminé mais fichier introuvable : {out}")
    return out
