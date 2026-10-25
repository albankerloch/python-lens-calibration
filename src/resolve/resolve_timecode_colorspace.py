"""Cale le Start TC des séquences d'images (EXR, DPX...) du Media Pool
sur le numéro de leur première frame : frame N -> timecode N / fps.

Pour les exports Unreal dont la numérotation est basée sur le timecode,
ex. Scene_1_41_Layer1_[00127756-00129800].exr -> 01:25:10:06 à 25 i/s.

Installation : copier ce fichier dans
  %APPDATA%/Blackmagic Design/DaVinci Resolve/Support/Fusion/Scripts/Utility/
(le nom du fichier devient le nom de l'entrée de menu).

Menu : Workspace > Scripts > Utility > <nom du fichier>
 - avec des clips sélectionnés dans le Media Pool : traite seulement ceux-là ;
 - sans sélection : traite toutes les séquences du projet dont le Start TC
   est encore à 00:00:00:00 (un timecode existant n'est pas écrasé), sauf les
   rendus Unreal, toujours recalés (Resolve leur attribue un Start TC faux à l'import).

Pour les rendus Unreal (nom en *_Layer<N>_*), règle aussi Input Color Space
sur Rec.709 (Scene). Input Gamma n'est pas modifiable par l'API de Resolve 21 :
le script signale les clips à passer en Linear à la main (Clip Attributes).
"""
import os
import re
import sys

try:
    resolve  # injecté par la console / certains contextes
except NameError:
    try:
        resolve = app.GetResolve()  # noqa: F821  (menu Workspace > Scripts)
    except NameError:
        # ligne de commande : module fourni par Resolve, hors du sys.path par défaut
        sys.path.append(os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
                                     r"Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules"))
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
        if resolve is None:
            sys.exit("Resolve injoignable : lancer DaVinci Resolve et régler "
                     "Préférences > Système > Général > External scripting using = Local.")

SEQUENCE = re.compile(r"\[(\d+)-(\d+)\]\.\w+$")
UNREAL_LAYER = re.compile(r"_Layer\d+_\[")
INPUT_COLOR_SPACE = "Rec.709"      # affiché « Rec.709 (Scene) » en RCM v2
INPUT_GAMMA = "Linear"


def tc_from_frame(frame, fps):
    fps = int(round(fps))
    ff, s = frame % fps, frame // fps
    return "%02d:%02d:%02d:%02d" % ((s // 3600) % 24, (s // 60) % 60, s % 60, ff)


def all_clips(folder):
    for clip in folder.GetClipList() or []:
        yield clip
    for sub in folder.GetSubFolderList() or []:
        yield from all_clips(sub)


def main():
    project = resolve.GetProjectManager().GetCurrentProject()
    media_pool = project.GetMediaPool()
    selected = media_pool.GetSelectedClips() or []
    clips = list(selected.values()) if isinstance(selected, dict) else list(selected)
    forced = bool(clips)
    if not clips:
        clips = list(all_clips(media_pool.GetRootFolder()))

    gamma_a_regler = []
    for clip in clips:
        props = clip.GetClipProperty() or {}
        name = props.get("File Name", "")
        if not SEQUENCE.search(name):
            continue

        unreal = bool(UNREAL_LAYER.search(name))
        if unreal:
            if not props.get("Input Color Space", "").startswith(INPUT_COLOR_SPACE):
                ok = clip.SetClipProperty("Input Color Space", INPUT_COLOR_SPACE)
                print("%s  Input Color Space -> %s  %s" % ("OK " if ok else "ÉCHEC", INPUT_COLOR_SPACE, name))
            if clip.GetClipProperty("Input Gamma") != INPUT_GAMMA:
                gamma_a_regler.append(name)

        tc = tc_from_frame(int(props["Start"]), float(props.get("FPS") or 25))
        if props.get("Start TC") == tc:
            print("déjà bon : %s  %s" % (name, tc))
            continue
        # numérotation Unreal = timecode : un Start TC différent (attribué par Resolve à l'import) est faux
        if not forced and not unreal and props.get("Start TC") not in ("00:00:00:00", "", None):
            print("ignoré (timecode déjà présent) : %s  %s" % (name, props.get("Start TC")))
            continue
        ok = clip.SetClipProperty("Start TC", tc)
        print("%s  %s -> %s  %s" % ("OK " if ok else "ÉCHEC", props.get("Start TC"), tc, name))

    if gamma_a_regler:
        print("\nÀ FAIRE à la main (non modifiable par script) : Clip Attributes > "
              "Input Gamma = %s pour :" % INPUT_GAMMA)
        for name in gamma_a_regler:
            print("  - " + name)


main()
