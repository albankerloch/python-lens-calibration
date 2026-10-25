"""Côté DaVinci Resolve du matching d'éclairage (voir src/light_match.py).

- export_stills (utilisé par la boucle) : deux stills de la timeline, sans rendu Fusion :
  la frame finale (look compris) et le matte de l'acteur (nœud LM_Matte, branché
  temporairement sur MediaOut1), plus layers.json.
- export_layers : variante qui rend l'acteur et le décor en EXR par comp.Render().
  À éviter : après un rendu Fusion lancé par script, Resolve refuse les stills un long
  moment, et CorridorKey peut échouer pendant le rendu (frame non keyée).
- apply_correction : écrit les gains, gammas et la saturation dans le nœud ColorGain
  « Person_Match », créé une seule fois entre Person_Expo et BG_Merge.Foreground ;
  le gain de l'ombre portée dans « Sink_Shadow » s'il existe ; le gain et le gamma du décor
  dans « BG_Match », créé une seule fois avant BG_Merge.Background ; et, si demandé, le gain
  de l'image entière dans « Ref_Grade », créé une seule fois entre BG_Merge et Look_Film.
  Avec `time`, les valeurs sont posées en keyframes à ce temps du comp.

Précautions (cf. incidents du 2026-10-22) :
- jamais AddTool(…, -32768, -32768) : Fusion insère alors l'outil dans le flux ;
- le graphe de connexions est relevé avant/après chaque opération, et toute
  différence non prévue lève une exception ;
- la source de l'acteur est vérifiée avant chaque export : rendu CorridorKey en EXR
  (Loader) présent pour la frame, ou plugin CorridorKey sans erreur. Une frame non keyée
  ne doit jamais être mesurée.

Utilisable depuis la console de Resolve, le MCP Resolve (variables resolve/project
injectées) ou en externe (Préférences > Général > External scripting = Local).
"""
from pathlib import Path
import json
import re
import time

try:
    resolve  # noqa: F821  (injecté par la console)
except NameError:
    try:
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
    except ImportError:
        resolve = None  # depuis le MCP : light_match_resolve.resolve = resolve après l'import

PERSON_SRC = "Person_Expo"      # sortie de l'acteur keyé, avant le merge
MATCH_NODE = "Person_Match"
SAT_NODE = "Person_Sat"         # saturation de l'acteur, juste avant MATCH_NODE
MERGE = "BG_Merge"
LOOK = "Look_Film"              # premier nœud après le merge
LOOK_NODES = ("Look_Film", "Look_Warm")   # mis en bypass le temps du still sans look
GLOBAL_NODE = "Ref_Grade"       # correction de l'image entière, entre MERGE et LOOK
BG_NODE = "BG_Match"            # correction du décor seul, juste avant MERGE.Background
BACKGROUND = "BG_Unreal"
OWN_SAVERS = ("Saver_Plate", "Saver_MatteBas", "Saver_AlphaHint")
SHADOW_NODE = "Sink_Shadow"     # ColorCorrector de l'ombre portée (Effect Mask = son matte)
SHADOW_MATTE = "LM_ShadowMatte" # matte de l'ombre portée, placé comme dans BG_Merge
OUTPUT = "MediaOut1"
OUT_SRC = "Out_CST"             # ce qui alimente normalement MediaOut1
MATTE = "LM_Matte"              # matte de l'acteur (blanc sur noir), placé comme dans BG_Merge
# « Blue » de la palette Set Color de Fusion (qui ramène toute couleur posée par script à sa palette) :
# repère les nœuds de correction dans le flow.
TILE_COLOR = {"R": 0.08235294117647059, "G": 0.3843137254901961, "B": 0.5176470588235295}


class Shot:
    """Comp Fusion du clip acteur, avec sa timeline et son clip (pour les conversions de temps)."""

    def __init__(self, timeline, item):
        self.timeline, self.item = timeline, item
        self.comp = item.GetFusionCompByIndex(1)


def get_comp(timeline_name="Take_41", track=None):
    """Clip de l'acteur : celui de `track`, ou à défaut le premier dont le comp contient MERGE."""
    project = resolve.GetProjectManager().GetCurrentProject()
    for i in range(1, project.GetTimelineCount() + 1):
        tl = project.GetTimelineByIndex(i)
        if tl.GetName() != timeline_name:
            continue
        tracks = [track] if track else range(1, tl.GetTrackCount("video") + 1)
        for k in tracks:
            for item in tl.GetItemListInTrack("video", k) or []:
                if item.GetFusionCompCount() and item.GetFusionCompByIndex(1).FindTool(MERGE):
                    return Shot(tl, item)
        raise LookupError(f"aucun clip avec un comp contenant {MERGE} dans {timeline_name}")
    raise LookupError(f"timeline {timeline_name} introuvable")


def graph(comp):
    g = {}
    for t in comp.GetToolList(False).values():
        name = t.GetAttrs()["TOOLS_Name"]
        for inp in t.GetInputList().values():
            src = inp.GetConnectedOutput()
            if src:
                g[f"{name}.{inp.GetAttrs()['INPS_ID']}"] = src.GetTool().GetAttrs()["TOOLS_Name"]
    return g


def _check_graph(before, after, allowed=()):
    diff = {k: (before.get(k), after.get(k)) for k in set(before) | set(after)
            if before.get(k) != after.get(k) and k not in allowed}
    if diff:
        raise RuntimeError(f"connexions Fusion modifiées de façon inattendue : {diff}")


def _timeline_fps(tl):
    return int(round(float(tl.GetSetting("timelineFrameRate"))))


def timecode_for(shot, comp_frame):
    """Timecode de la timeline qui montre le temps `comp_frame` du comp (inverse de comp_time)."""
    tl, item = shot.timeline, shot.item
    fps = _timeline_fps(tl)
    frame = int(comp_frame) + item.GetStart() - item.GetSourceStartFrame()
    s, f = divmod(frame, fps)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def comp_time(shot, timecode=None):
    return _comp_time(shot, timecode)


def _comp_time(shot, timecode=None):
    tl, item = shot.timeline, shot.item
    tc = timecode or tl.GetCurrentTimecode()
    fps = _timeline_fps(tl)
    h, m, s, f = (int(v) for v in tc.replace(";", ":").split(":"))
    frame = ((h * 60 + m) * 60 + s) * fps + f
    return frame - item.GetStart() + item.GetSourceStartFrame()


def _person_output(comp):
    return comp.FindTool(MATCH_NODE) or comp.FindTool(PERSON_SRC)


def wait_ready(timeout=600.0, poll=2.0):
    """Attend que Resolve réponde de nouveau (après un comp.Render(), GetCurrentPage()
    renvoie None un long moment). Retourne le temps d'attente en secondes."""
    start = time.monotonic()
    while resolve.GetCurrentPage() is None:
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Resolve indisponible depuis plus de {timeout:.0f} s")
        time.sleep(poll)
    return time.monotonic() - start


def _goto(tl, timecode, tries=3):
    """SetCurrentTimecode peut échouer sans erreur : on relit et on réessaie."""
    for _ in range(tries):
        tl.SetCurrentTimecode(timecode)
        time.sleep(1.0)
        if tl.GetCurrentTimecode() == timecode:
            return
        resolve.OpenPage("color")
        time.sleep(2.0)
    raise RuntimeError(f"impossible de placer la tête de lecture sur {timecode}"
                       f" (elle est sur {tl.GetCurrentTimecode()})")


def export_layers(shot, out_dir, timecode=None, settle=4.0):
    """Exporte dans `out_dir` :
    - person_<t>.exr : l'acteur keyé (RGBA), rendu dans Fusion, pour son alpha ;
    - background_<t>.exr : le décor Unreal (BG_Unreal), pour diagnostic ;
    - still.png : la frame de la timeline telle qu'elle sort (look compris), mesurée
      comme l'image de référence ;
    - layers.json : fichiers, temps du comp, Center du Merge, temps d'attente de Resolve.
    """
    comp, tl = shot.comp, shot.timeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    waited = wait_ready()
    resolve.OpenPage("color")
    if timecode:
        _goto(tl, timecode)
    t = _comp_time(shot, timecode)

    # la frame de la timeline d'abord : juste après comp.Render(), Resolve reste un moment
    # indisponible (GetCurrentPage() -> None) et refuse ExportCurrentFrameAsStill
    time.sleep(settle)
    project = resolve.GetProjectManager().GetCurrentProject()
    if not project.ExportCurrentFrameAsStill(str(out_dir / "still.png")):
        raise RuntimeError("ExportCurrentFrameAsStill en échec (page Color, clip sous la tête de lecture ?)")

    before = graph(comp)
    own = [comp.FindTool(n) for n in OWN_SAVERS if comp.FindTool(n)]
    own_state = [s.GetAttrs()["TOOLB_PassThrough"] for s in own]
    comp.SetActiveTool(None)
    savers = {}
    try:
        for i, (key, tool) in enumerate((("person", _person_output(comp)),
                                         ("background", comp.FindTool(BACKGROUND)))):
            s = comp.AddTool("Saver", 60 + i, 60)
            s.SetAttrs({"TOOLS_Name": f"LM_TMP_{key}"})
            s.Clip = str(out_dir / f"{key}_.exr")
            s.Input = tool.Output
            savers[key] = s
        for s in own:
            s.SetAttrs({"TOOLB_PassThrough": True})
        if not comp.Render({"Start": t, "End": t, "Wait": True}):
            raise RuntimeError("rendu Fusion en échec")
    finally:
        for s, state in zip(own, own_state):
            s.SetAttrs({"TOOLB_PassThrough": state})
        for s in savers.values():
            s.Input = None
            s.Delete()
    _check_graph(before, graph(comp))

    center = comp.FindTool(MERGE).GetInput("Center")
    files = {key: f"{key}_{int(t):04d}.exr" for key in ("person", "background")}
    for f in [*files.values(), "still.png"]:
        if not (out_dir / f).is_file():
            raise FileNotFoundError(out_dir / f)
    meta = {**files, "still": "still.png", "transfer": "gamma24", "comp_time": t,
            "resolve_wait_s": round(waited, 1),
            "timecode": tl.GetCurrentTimecode(), "merge_center": [center[1], center[2]],
            "person_node": _person_output(comp).GetAttrs()["TOOLS_Name"]}
    (out_dir / "layers.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _person_source(comp):
    """Premier Loader ou plugin OFX en remontant les entrées Input depuis PERSON_SRC :
    le rendu CorridorKey (EXR) ou le plugin lui-même."""
    tool = comp.FindTool(PERSON_SRC)
    for _ in range(20):
        src = tool.Input.GetConnectedOutput() if tool.Input else None
        if not src:
            break
        tool = src.GetTool()
        reg = tool.GetAttrs()["TOOLS_RegID"]
        if reg == "Loader" or reg.startswith("ofx."):
            return tool
    raise LookupError(f"aucune source (Loader ou plugin) en amont de {PERSON_SRC}")


def _loader_frame_file(loader, t):
    """Fichier de la séquence lu par `loader` au temps `t` du comp (None hors plage)."""
    first = Path(loader.GetAttrs()["TOOLST_Clip_Name"][1])
    g_in, g_out = loader.GetInput("GlobalIn"), loader.GetInput("GlobalOut")
    if not g_in <= t <= g_out:
        return None
    m = re.search(r"(\d+)$", first.stem)
    if not m:
        return first
    n = int(m.group(1)) + int(t - g_in) + int(loader.GetInput("ClipTimeStart") or 0)
    return first.with_name(first.stem[:m.start()] + str(n).zfill(len(m.group(1))) + first.suffix)


def wait_keyer(comp, t=None, timeout=120.0, poll=1.0):
    """Vérifie que l'acteur est keyé au temps `t` du comp (défaut : comp.CurrentTime, qui ne
    suit pas la tête de lecture de la page Color : passer `t`).

    Rendu CorridorKey en EXR (Loader) : le fichier de la frame doit exister.
    Plugin CorridorKey : attend qu'il ait fini la frame ; lève une erreur s'il a échoué
    (en erreur, il laisse passer la plate sans la keyer : la mesurer donnerait une
    correction fausse).
    """
    src = _person_source(comp)
    if src.GetAttrs()["TOOLS_RegID"] == "Loader":
        t = comp.CurrentTime if t is None else t
        f = _loader_frame_file(src, t)
        if f is None or not f.is_file():
            raise RuntimeError(f"{src.GetAttrs()['TOOLS_Name']} : pas de rendu CorridorKey pour la frame"
                               f" {t:.0f} ({f})")
        return f"loader {f.name}"
    # états vus dans corridorkey-ofx.log : "rendering 0 of 1", "gathering 1 frame",
    # "2/15 waiting for runtime", "GPU warmup", "failed, retry in 8s" (occupé) ;
    # "rendered", "cached, no re-render" (prêt) ; "error, see log" (échec)
    busy = re.compile(r"^(rendering|gathering|\d+/\d+|GPU warmup|failed, retry)")
    start = time.monotonic()
    while True:
        status = str(src.GetInput("status") or "")
        if not busy.match(status):
            break
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"CorridorKey toujours occupé après {timeout:.0f} s : {status}")
        time.sleep(poll)
    if status.startswith(("error", "failed")):
        raise RuntimeError(f"CorridorKey en erreur ({status}) : la frame n'est pas keyée")
    return status


def _ensure_matte_node(comp):
    """LM_Matte = blanc là où l'acteur est opaque, sur fond noir opaque, décalé comme
    l'acteur dans BG_Merge (Center lié par expression). Créé une seule fois."""
    node = comp.FindTool(MATTE)
    if node:
        return node
    before = graph(comp)
    comp.SetActiveTool(None)
    black = comp.AddTool("Background", 110, 70)
    black.SetAttrs({"TOOLS_Name": "LM_Black"})
    black.SetInput("TopLeftAlpha", 1.0)
    white = comp.AddTool("Background", 110, 72)
    white.SetAttrs({"TOOLS_Name": "LM_White"})
    for ch in ("TopLeftRed", "TopLeftGreen", "TopLeftBlue", "TopLeftAlpha"):
        white.SetInput(ch, 1.0)
    white.EffectMask = _person_output(comp).Output
    node = comp.AddTool("Merge", 112, 71)
    node.SetAttrs({"TOOLS_Name": MATTE})
    node.Background = black.Output
    node.Foreground = white.Output
    node.Center.SetExpression(f"{MERGE}.Center")
    expected = {"LM_White.EffectMask", f"{MATTE}.Background", f"{MATTE}.Foreground"}
    _check_graph(before, graph(comp), allowed=expected)
    return node


def _ensure_shadow_matte_node(comp):
    """SHADOW_MATTE = matte de l'ombre portée (celui de SHADOW_NODE.EffectMask) en blanc sur
    noir, décalé comme l'acteur dans BG_Merge. Créé une seule fois ; None sans SHADOW_NODE."""
    node = comp.FindTool(SHADOW_MATTE)
    if node:
        return node
    shadow = comp.FindTool(SHADOW_NODE)
    src = shadow.EffectMask.GetConnectedOutput() if shadow else None
    if not src:
        return None
    _ensure_matte_node(comp)   # crée LM_Black au besoin
    before = graph(comp)
    comp.SetActiveTool(None)
    white = comp.AddTool("Background", 110, 75)
    white.SetAttrs({"TOOLS_Name": "LM_ShadowWhite"})
    for ch in ("TopLeftRed", "TopLeftGreen", "TopLeftBlue", "TopLeftAlpha"):
        white.SetInput(ch, 1.0)
    white.EffectMask = src
    node = comp.AddTool("Merge", 112, 75)
    node.SetAttrs({"TOOLS_Name": SHADOW_MATTE})
    node.Background = comp.FindTool("LM_Black").Output
    node.Foreground = white.Output
    node.Center.SetExpression(f"{MERGE}.Center")
    expected = {"LM_ShadowWhite.EffectMask", f"{SHADOW_MATTE}.Background", f"{SHADOW_MATTE}.Foreground"}
    _check_graph(before, graph(comp), allowed=expected)
    return node


def _thumb(path):
    """Vignette en niveaux de gris d'un still (None si OpenCV indisponible, ex. Python de Resolve)."""
    try:
        import cv2
    except ImportError:
        return None
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if im is None:
        return None
    im = cv2.resize(im, (128, 72), interpolation=cv2.INTER_AREA).astype("float32")
    return im / (65535.0 if im.max() > 255 else 255.0)


def _export_still(path, settle, differs_from=None, tries=5):
    """Exporte la frame affichée. Avec `differs_from` (un still déjà exporté), vérifie que l'image
    a bien changé : après un changement dans le comp, Resolve peut encore livrer l'ancienne
    image ; on réessaie alors une seconde plus tard, plutôt que d'attendre longtemps à chaque fois."""
    project = resolve.GetProjectManager().GetCurrentProject()
    ref = _thumb(differs_from) if differs_from else None
    time.sleep(settle)
    for _ in range(tries):
        if not project.ExportCurrentFrameAsStill(str(path)):
            raise RuntimeError(f"ExportCurrentFrameAsStill en échec ({path.name})")
        if ref is None:
            return
        new = _thumb(path)
        if new is None or abs(new - ref).mean() > 1.0 / 255:
            return
        time.sleep(1.0)
    raise RuntimeError(f"{path.name} identique à {Path(differs_from).name} après {tries} essais :"
                       " Resolve n'a pas rafraîchi l'image")


def export_stills(shot, out_dir, timecode=None, settle=1.0, prelook=True):
    """Exporte dans `out_dir`, sans rendu Fusion :
    - still.png : la frame de la timeline telle qu'elle sort (look compris) ; 8 bits suffisent
      pour la teinte, la saturation et les hautes lumières ;
    - still_prelook.tif : la même sans le look (LOOK_NODES en bypass), en 16 bits, pour les noirs,
      l'ombre portée et le pivot du décor ; seulement si `prelook` ;
    - matte.png : le matte de l'acteur à sa place dans l'image (blanc = acteur) ;
    - shadow.png : le matte de l'ombre portée, si SHADOW_NODE existe ;
    - layers.json.
    MediaOut1 est rebranché sur LM_Matte (puis LM_ShadowMatte) le temps de ces stills, puis rétabli.
    """
    comp, tl = shot.comp, shot.timeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    waited = wait_ready()
    resolve.OpenPage("color")
    if timecode:
        _goto(tl, timecode)
    matte_node = _ensure_matte_node(comp)
    shadow_node = _ensure_shadow_matte_node(comp)
    out = comp.FindTool(OUTPUT)
    t = _comp_time(shot, timecode)
    status = wait_keyer(comp, t)
    still = out_dir / "still.png"
    _export_still(still, settle)

    looks = [comp.FindTool(n) for n in LOOK_NODES if comp.FindTool(n)] if prelook else []
    look_state = [t.GetAttrs()["TOOLB_PassThrough"] for t in looks]
    try:
        for tool in looks:
            tool.SetAttrs({"TOOLB_PassThrough": True})
        if looks:
            _export_still(out_dir / "still_prelook.tif", settle, differs_from=still)
    finally:
        for tool, state in zip(looks, look_state):
            tool.SetAttrs({"TOOLB_PassThrough": state})

    before = graph(comp)
    try:
        out.Input = matte_node.Output
        _export_still(out_dir / "matte.png", settle, differs_from=still)
        if shadow_node:
            out.Input = shadow_node.Output
            _export_still(out_dir / "shadow.png", settle, differs_from=out_dir / "matte.png")
    finally:
        out.Input = comp.FindTool(OUT_SRC).Output
    _check_graph(before, graph(comp))
    status_after = wait_keyer(comp, t)

    meta = {"still": "still.png", "matte": "matte.png", "transfer": "gamma24",
            "comp_time": _comp_time(shot, timecode), "timecode": tl.GetCurrentTimecode(),
            "resolve_wait_s": round(waited, 1), "keyer_status": [status, status_after],
            "person_node": _person_output(comp).GetAttrs()["TOOLS_Name"]}
    if shadow_node:
        meta["shadow"] = "shadow.png"
    if looks:
        meta["prelook"] = "still_prelook.tif"
    (out_dir / "layers.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _mark(node):
    """Colore en bleu un nœud de correction créé par ce script (aussi ceux créés avant cette règle)."""
    node.TileColor = TILE_COLOR
    return node


def _ensure_match_node(comp):
    node = comp.FindTool(MATCH_NODE)
    if node:
        return _mark(node)
    merge, person = comp.FindTool(MERGE), comp.FindTool(PERSON_SRC)
    before = graph(comp)
    comp.SetActiveTool(None)
    node = comp.AddTool("ColorGain", 80, 60)
    node.SetAttrs({"TOOLS_Name": MATCH_NODE})
    node.Input = person.Output
    merge.Foreground = node.Output
    expected = {f"{MATCH_NODE}.Input", f"{MERGE}.Foreground"}
    _check_graph(before, graph(comp), allowed=expected)
    return _mark(node)


def _ensure_sat_node(comp):
    """SAT_NODE = ColorGain qui ne fait que la saturation de l'acteur, juste avant MATCH_NODE.

    Séparée de la teinte (gains de MATCH_NODE) pour que les deux corrections ne se poursuivent
    pas : la saturation, mesurée insensible aux gains, se règle seule, et la teinte s'ajuste
    ensuite sur une saturation fixée. Créé une seule fois ; la saturation éventuellement
    réglée dans MATCH_NODE (ancienne version) y est reprise, et remise à 1 dans MATCH_NODE.
    """
    node = comp.FindTool(SAT_NODE)
    if node:
        return _mark(node)
    match = _ensure_match_node(comp)
    src = match.Input.GetConnectedOutput()
    before = graph(comp)
    comp.SetActiveTool(None)
    node = comp.AddTool("ColorGain", 79, 60)
    node.SetAttrs({"TOOLS_Name": SAT_NODE})
    node.Input = src
    match.Input = node.Output
    _check_graph(before, graph(comp), allowed={f"{SAT_NODE}.Input", f"{MATCH_NODE}.Input"})
    node.SetInput("PreDividePostMultiply", 1)
    for ch in ("Red", "Green", "Blue"):
        if _input(match, f"Saturation{ch}").GetConnectedOutput() is None:
            node.SetInput(f"Saturation{ch}", match.GetInput(f"Saturation{ch}"))
            match.SetInput(f"Saturation{ch}", 1.0)
    flow = comp.CurrentFrame.FlowView if comp.CurrentFrame else None
    if flow:  # entre la source et MATCH_NODE, pour la lisibilité
        a, b = flow.GetPosTable(src.GetTool()), flow.GetPosTable(match)
        flow.SetPos(node, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
    return _mark(node)


def _input(node, name):
    for inp in node.GetInputList().values():
        if inp.GetAttrs()["INPS_ID"] == name:
            return inp
    raise KeyError(f"{node.GetAttrs()['TOOLS_Name']} n'a pas d'entrée {name}")


def _set(node, name, value, time=None):
    """Valeur fixe, ou keyframe au temps `time` du comp (l'entrée est animée au besoin).
    Une entrée déjà animée n'accepte qu'une keyframe : une valeur fixe n'y agirait
    qu'à la frame courante."""
    animated = _input(node, name).GetConnectedOutput() is not None
    if time is None:
        if animated:
            raise RuntimeError(f"{node.GetAttrs()['TOOLS_Name']}.{name} est animé : relancer avec"
                               " --animate, ou supprimer l'animation dans Fusion")
        node.SetInput(name, value)
        return
    if not animated:
        node.AddModifier(name, "BezierSpline")
    node.SetInput(name, value, time)


def _ensure_global_node(comp):
    """GLOBAL_NODE = ColorGain sur l'image entière, entre MERGE et LOOK. Créé une seule fois."""
    node = comp.FindTool(GLOBAL_NODE)
    if node:
        return _mark(node)
    merge, look = comp.FindTool(MERGE), comp.FindTool(LOOK)
    if look.Input.GetConnectedOutput().GetTool().GetAttrs()["TOOLS_Name"] != MERGE:
        raise RuntimeError(f"{LOOK} n'est pas branché directement sur {MERGE} : placer {GLOBAL_NODE} à la main")
    before = graph(comp)
    comp.SetActiveTool(None)
    node = comp.AddTool("ColorGain", 82, 60)
    node.SetAttrs({"TOOLS_Name": GLOBAL_NODE})
    node.Input = merge.Output
    look.Input = node.Output
    _check_graph(before, graph(comp), allowed={f"{GLOBAL_NODE}.Input", f"{LOOK}.Input"})
    flow = comp.CurrentFrame.FlowView if comp.CurrentFrame else None
    if flow:  # entre les deux nœuds dans le flow, pour la lisibilité
        a, b = flow.GetPosTable(merge), flow.GetPosTable(look)
        flow.SetPos(node, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2 - 1)
    return _mark(node)


def _ensure_bg_node(comp):
    """BG_NODE = ColorGain sur le décor seul, entre ce qui alimente MERGE.Background et MERGE.
    Créé une seule fois."""
    node = comp.FindTool(BG_NODE)
    if node:
        return _mark(node)
    merge = comp.FindTool(MERGE)
    src = merge.Background.GetConnectedOutput()
    if not src:
        raise RuntimeError(f"{MERGE}.Background n'est pas branché")
    before = graph(comp)
    comp.SetActiveTool(None)
    node = comp.AddTool("ColorGain", 84, 60)
    node.SetAttrs({"TOOLS_Name": BG_NODE})
    node.Input = src
    merge.Background = node.Output
    _check_graph(before, graph(comp), allowed={f"{BG_NODE}.Input", f"{MERGE}.Background"})
    flow = comp.CurrentFrame.FlowView if comp.CurrentFrame else None
    if flow:  # entre la source du décor et le merge, pour la lisibilité
        a, b = flow.GetPosTable(src.GetTool()), flow.GetPosTable(merge)
        flow.SetPos(node, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
    return _mark(node)


def apply_correction(shot, corr, with_global=False, time=None):
    """Acteur : saturation dans SAT_NODE, puis out = gain_c * in ** k_c dans MATCH_NODE, sur
    l'image dé-prémultipliée.
    Ombre portée (SHADOW_NODE) : gain de la zone d'ombre, si `corr["shadow_gain"]` est donné.
    Décor (BG_NODE) : out = bg_gain * in ** bg_k, si la correction n'est pas l'identité.
    Image entière (GLOBAL_NODE, après le merge) : out = global_gain * in, si `with_global`.
    `time` : keyframes à ce temps du comp au lieu de valeurs fixes.

    ColorGain applique le gain puis le gamma : out = (gain * in) ** (1/gamma).
    Pour obtenir G * in**k on règle gamma = 1/k et gain_c = G_c ** (1/k), canal par canal.
    """
    k = corr["k"] if isinstance(corr["k"], (list, tuple)) else [corr["k"]] * 3
    node = _ensure_match_node(shot.comp)
    _set(node, "PreDividePostMultiply", 1)
    for ch, g, kc in zip(("Red", "Green", "Blue"), corr["gain"], k):
        _set(node, f"Gain{ch}", float(g) ** (1.0 / kc), time)
        _set(node, f"Gamma{ch}", 1.0 / kc, time)
    snode = _ensure_sat_node(shot.comp)
    for ch in ("Red", "Green", "Blue"):
        _set(snode, f"Saturation{ch}", float(corr.get("sat", 1.0)), time)
    shadow = shot.comp.FindTool(SHADOW_NODE)
    if shadow and corr.get("shadow_gain") is not None:
        _set(shadow, "PreDividePostMultiply", 1)
        _set(shadow, "MasterRGBGain", float(corr["shadow_gain"]), time)
    bg_k, bg_gain = float(corr.get("bg_k", 1.0)), float(corr.get("bg_gain", 1.0))
    if shot.comp.FindTool(BG_NODE) or bg_k != 1.0 or bg_gain != 1.0:
        bnode = _ensure_bg_node(shot.comp)
        for ch in ("Red", "Green", "Blue"):
            _set(bnode, f"Gain{ch}", bg_gain ** (1.0 / bg_k), time)
            _set(bnode, f"Gamma{ch}", 1.0 / bg_k, time)
    if with_global:
        gnode = _ensure_global_node(shot.comp)
        for ch, g in zip(("Red", "Green", "Blue"), corr["global_gain"]):
            _set(gnode, f"Gain{ch}", float(g), time)
    return node


def read_correction(shot, time=None):
    """Correction en place (au temps `time` du comp, ou à la frame courante) : k et gain par
    canal, saturation, gain de l'ombre portée, global_gain ; identité pour les nœuds absents."""
    corr = {"k": [1.0, 1.0, 1.0], "gain": [1.0, 1.0, 1.0], "global_gain": [1.0, 1.0, 1.0],
            "sat": 1.0, "shadow_gain": None, "bg_k": 1.0, "bg_gain": 1.0}
    get = (lambda n, name: n.GetInput(name)) if time is None else (lambda n, name: n.GetInput(name, time))
    chs = ("Red", "Green", "Blue")
    node = shot.comp.FindTool(MATCH_NODE)
    if node:
        k = [1.0 / get(node, f"Gamma{ch}") for ch in chs]
        corr.update(k=k, gain=[get(node, f"Gain{ch}") ** kc for ch, kc in zip(chs, k)],
                    sat=float(sum(get(node, f"Saturation{ch}") for ch in chs) / 3))
    snode = shot.comp.FindTool(SAT_NODE)
    if snode:   # saturation séparée de la teinte (MATCH_NODE garde alors 1)
        corr["sat"] = float(sum(get(snode, f"Saturation{ch}") for ch in chs) / 3)
    shadow = shot.comp.FindTool(SHADOW_NODE)
    if shadow:
        corr["shadow_gain"] = float(get(shadow, "MasterRGBGain"))
    bnode = shot.comp.FindTool(BG_NODE)
    if bnode:
        bg_k = 1.0 / get(bnode, "GammaGreen")
        corr.update(bg_k=bg_k, bg_gain=get(bnode, "GainGreen") ** bg_k)
    gnode = shot.comp.FindTool(GLOBAL_NODE)
    if gnode:
        corr["global_gain"] = [get(gnode, f"Gain{ch}") for ch in chs]
    return corr
