"""Rendu Movie Render Queue - Scene_1_41 (prise 41), couche décor Unreal.

Usage, dans l'Output Log d'Unreal (Cmd: Python) :
    py "C:/Users/alban/Desktop/Shaking Tokyo 2026 V2/Story Board/Unreal Engine/Scripts/mrq_take41.py" test
    py "C:/Users/alban/Desktop/Shaking Tokyo 2026 V2/Story Board/Unreal Engine/Scripts/mrq_take41.py" full

Sans argument (ex. Fichier > Exécuter un script Python), le mode TEST est utilisé.

TEST : 5 frames 128898-128902 -> take_41/_test_128900
DIFF : frame 128900 sans ombre ni occlusion du perso -> take_41/_test_sans_perso
       (réglages du perso restaurés à la fin du rendu)
TOUT : TEST puis DIFF enchaînés automatiquement
FULL : Movie Render Graph MRG_Take41_DVR (copie de DVRDefaultRenderGraph) -> take_41
       plage de la séquence + 12 frames de handles = 127756-129800, EXR + métadonnées unreal/*

EXR : demi-flottant (float16), compression DWAA (EXR_COMPRESSION), dans tous les modes.
DWAA est avec pertes sur R, G, B (invisibles à la qualité par défaut, 45) et sans pertes sur
A ; les fichiers sont 4 à 6 fois plus légers qu'en ZIPS et bien plus rapides à lire dans
Fusion. En mode FULL, les nœuds EXR du graphe sont réglés et le graphe enregistré.
En fin de rendu, l'en-tête de la dernière EXR écrite est contrôlé (compression, float16).

Le script crée une file d'attente séparée : la file MRQ de l'interface n'est pas modifiée.
Après le rendu, ajouter le timecode EXR (lu par DaVinci Resolve) :
    python exr_timecode.py "<dossier des EXR>"
"""
import os
import struct
import sys
import unreal

SEQUENCE = "/Game/Cinematics/Takes/2026-10-22/Scene_1_41.Scene_1_41"
LEVEL = "/Game/LatestTestMap.LatestTestMap"
EXPORT_ROOT = "C:/Users/alban/Desktop/Shaking Tokyo 2026 V2/Story Board/Unreal Engine/Export Unreal/take_41"
FILE_NAME = "Scene_1_41_Layer1_{frame_number}"
PRESET_DIR = "/Game/Cinematics/MRQ"
PRESET_NAME = "MRQ_Take41_EXR_Lineaire"
GRAPH = "/Game/Cinematics/MRQ/MRG_Take41_DVR.MRG_Take41_DVR"
EXR_COMPRESSION = "DWAA"
# codes de compression OpenEXR (octet de l'attribut « compression ») et types de pixel (chlist)
EXR_COMPRESSION_CODES = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ", 5: "PXR24",
                         6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}
EXR_PIXEL_TYPES = {0: "uint32", 1: "float16", 2: "float32"}

MODES = {
    # custom_end_frame est exclusif
    "test": dict(start=128898, end=128903, out=EXPORT_ROOT + "/_test_128900"),
    "full": dict(start=127756, end=129801, out=EXPORT_ROOT),
    # Frame 128900 sans aucune empreinte du perso (ombre, occlusion Lumen) -> comparaison avec TEST
    "diff": dict(start=128900, end=128901, out=EXPORT_ROOT + "/_test_sans_perso"),
}
PERSO_LABEL = "PERSO_Cuisine_Mangeur"
PERSO_PROPS = ("cast_shadow", "cast_hidden_shadow", "affect_indirect_lighting_while_hidden")

_keep_alive = {}
_report = []


def set_if(obj, name, value):
    """Applique une propriété et note le résultat (le nom varie selon les versions d'UE)."""
    try:
        obj.set_editor_property(name, value)
        _report.append(f"  OK   {type(obj).__name__}.{name} = {value}")
        return True
    except Exception as e:
        _report.append(f"  SKIP {type(obj).__name__}.{name} ({e})")
        return False


def exr_compression():
    fmt = getattr(unreal.EXRCompressionFormat, EXR_COMPRESSION, None)
    if fmt is None:
        known = [n for n in dir(unreal.EXRCompressionFormat) if n.isupper()]
        raise RuntimeError(f"compression EXR {EXR_COMPRESSION} absente de cette version d'Unreal ({known})")
    return fmt


def graph_nodes(graph):
    """Nœuds du Movie Render Graph : liste complète si exposée, sinon parcours depuis la sortie."""
    try:
        return list(graph.get_editor_property("all_nodes"))
    except Exception:
        pass
    seen, todo = [], [graph.get_output_node()]
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.append(node)
        for pin in node.get_input_pins():
            todo.extend(pin.get_connected_nodes())
    return seen


def configure_graph_exr(graph):
    """Règle la compression de tous les nœuds de sortie EXR du graphe, puis l'enregistre."""
    fmt = exr_compression()
    nodes = [n for n in graph_nodes(graph) if isinstance(n, unreal.MovieGraphImageSequenceOutputNode_EXR)]
    if not nodes:
        raise RuntimeError(f"aucun nœud de sortie EXR trouvé dans {GRAPH}")
    for node in nodes:
        set_if(node, "override_compression", True)
        set_if(node, "compression", fmt)
    unreal.EditorAssetLibrary.save_loaded_asset(graph)
    _report.append(f"  OK   graphe enregistré ({len(nodes)} nœud(s) EXR)")


def exr_format(path):
    """(compression, {canal: type de pixel}) lus dans l'en-tête d'une EXR scanline."""
    with open(path, "rb") as f:
        data = f.read(65536)
    if data[:4] != b"\x76\x2f\x31\x01":
        raise ValueError(f"pas un fichier EXR : {path}")
    i, comp, channels = 8, None, {}
    while data[i] != 0:
        j = data.index(b"\0", i)
        k = data.index(b"\0", j + 1)
        name, size = data[i:j], struct.unpack("<i", data[k + 1:k + 5])[0]
        value = data[k + 5:k + 5 + size]
        if name == b"compression":
            comp = EXR_COMPRESSION_CODES.get(value[0], value[0])
        elif name == b"channels":
            p = 0
            while value[p] != 0:
                q = value.index(b"\0", p)
                ptype = struct.unpack("<i", value[q + 1:q + 5])[0]
                channels[value[p:q].decode()] = EXR_PIXEL_TYPES.get(ptype, ptype)
                p = q + 17      # type (4) + pLinear (1) + réservé (3) + xSampling (4) + ySampling (4)
        i = k + 5 + size
    return comp, channels


def check_last_exr(root):
    """Contrôle la dernière EXR écrite sous `root` : compression attendue et float16."""
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".exr")]
    if not files:
        unreal.log_warning(f"[mrq_take41] aucune EXR sous {root} : contrôle impossible")
        return
    last = max(files, key=os.path.getmtime)
    comp, channels = exr_format(last)
    ok = comp == EXR_COMPRESSION and set(channels.values()) == {"float16"}
    log = unreal.log if ok else unreal.log_error
    log(f"[mrq_take41] {os.path.basename(last)} : compression {comp}, canaux {channels}"
        + ("" if ok else f" -> attendu {EXR_COMPRESSION} en float16"))


def build_job(queue, mode):
    m = MODES[mode]
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = f"Scene_1_41_Layer1_{mode}"
    job.sequence = unreal.SoftObjectPath(SEQUENCE)
    job.map = unreal.SoftObjectPath(LEVEL)
    cfg = job.get_configuration()

    # Sortie : 4032x2268, 25 i/s, numérotation timecode sur 8 chiffres
    out = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    set_if(out, "output_directory", unreal.DirectoryPath(m["out"]))
    set_if(out, "file_name_format", FILE_NAME)
    set_if(out, "output_resolution", unreal.IntPoint(4032, 2268))
    set_if(out, "use_custom_frame_rate", True)
    set_if(out, "output_frame_rate", unreal.FrameRate(25, 1))
    set_if(out, "use_custom_playback_range", True)
    set_if(out, "custom_start_frame", m["start"])
    set_if(out, "custom_end_frame", m["end"])
    set_if(out, "zero_pad_frame_numbers", 8)
    set_if(out, "frame_number_offset", 0)
    set_if(out, "handle_frame_count", 0)
    set_if(out, "override_existing_output", True)

    # Passe deferred (finale)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

    # EXR linéaire demi-flottant (seul format de la passe finale), DWAA, une seule couche
    exr = cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
    set_if(exr, "compression", exr_compression())
    set_if(exr, "multilayer", False)

    # Couleur : pas de tone curve, pas d'OCIO (primaires de travail sRGB/Rec.709)
    col = cfg.find_or_add_setting_by_class(unreal.MoviePipelineColorSetting)
    set_if(col, "disable_tone_curve", True)
    try:
        ocio = col.get_editor_property("ocio_configuration")
        ocio.set_editor_property("is_enabled", False)
        col.set_editor_property("ocio_configuration", ocio)
        _report.append("  OK   OCIO désactivé")
    except Exception as e:
        _report.append(f"  SKIP OCIO ({e})")

    # Anti-aliasing : 8 échantillons temporels, 32 frames de warm-up
    aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    set_if(aa, "spatial_sample_count", 1)
    set_if(aa, "temporal_sample_count", 8)
    set_if(aa, "override_anti_aliasing", True)
    set_if(aa, "anti_aliasing_method", unreal.AntiAliasingMethod.AAM_NONE)
    set_if(aa, "use_camera_cut_for_warm_up", False)
    set_if(aa, "engine_warm_up_count", 32)
    set_if(aa, "render_warm_up_count", 32)
    set_if(aa, "render_warm_up_frames", True)

    # Obturateur centré sur la frame ; angle 150° = MotionBlurAmount 0.4167 (PPV_Take41_ExpoPhysique)
    cam = cfg.find_or_add_setting_by_class(unreal.MoviePipelineCameraSetting)
    set_if(cam, "shutter_timing", unreal.MoviePipelineShutterTiming.FRAME_CENTER)
    set_if(cam, "overscan_percentage", 0.0)

    return job, cfg


def save_preset(cfg):
    try:
        path = f"{PRESET_DIR}/{PRESET_NAME}"
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            preset = unreal.EditorAssetLibrary.load_asset(path)
        else:
            preset = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                PRESET_NAME, PRESET_DIR, unreal.MoviePipelinePrimaryConfig, None)
        preset.copy_from(cfg)
        unreal.EditorAssetLibrary.save_loaded_asset(preset)
        _report.append(f"  OK   preset enregistré : {path}")
    except Exception as e:
        _report.append(f"  SKIP preset ({e})")


def perso_components():
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        if actor.get_actor_label() == PERSO_LABEL:
            return list(actor.get_components_by_class(unreal.StaticMeshComponent))
    raise RuntimeError(f"{PERSO_LABEL} introuvable")


def neutralize_perso():
    """Coupe ombre et occlusion du perso ; renvoie les valeurs d'origine pour les restaurer."""
    saved = []
    for comp in perso_components():
        saved.append((comp, {p: comp.get_editor_property(p) for p in PERSO_PROPS}))
        for p in PERSO_PROPS:
            comp.set_editor_property(p, False)
    return saved


def restore_perso(saved):
    for comp, values in saved:
        for p, v in values.items():
            comp.set_editor_property(p, v)
    unreal.log(f"[mrq_take41] {PERSO_LABEL} restauré : {[v for _, v in saved]}")


def on_finished(executor, success):
    unreal.log(f"[mrq_take41] Rendu terminé - succès : {success}")
    if success:
        try:
            check_last_exr(_keep_alive.get("out", EXPORT_ROOT))
        except Exception as e:
            unreal.log_warning(f"[mrq_take41] contrôle EXR impossible : {e}")
    if "perso_saved" in _keep_alive:
        restore_perso(_keep_alive.pop("perso_saved"))
    pending = _keep_alive.get("pending", [])
    if success and pending:
        # Lancement du rendu suivant au tick suivant, une fois l'exécuteur libéré
        def next_render(_dt):
            unreal.unregister_slate_post_tick_callback(_keep_alive.pop("tick"))
            render(pending.pop(0))
        _keep_alive["tick"] = unreal.register_slate_post_tick_callback(next_render)
    elif pending:
        unreal.log_error(f"[mrq_take41] rendu en échec, rendus suivants annulés : {pending}")
        pending.clear()


def build_graph_job(queue):
    """Job FULL piloté par le Movie Render Graph (mêmes réglages et métadonnées que l'export v1)."""
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = "Scene_1_41"
    job.sequence = unreal.SoftObjectPath(SEQUENCE)
    job.map = unreal.SoftObjectPath(LEVEL)
    graph = unreal.load_asset(GRAPH)
    configure_graph_exr(graph)
    job.set_graph_preset(graph)
    return job


def render(mode):
    _report.clear()
    queue = unreal.MoviePipelineQueue()
    if mode == "full":
        build_graph_job(queue)
        unreal.log(f"[mrq_take41] mode=full graphe={GRAPH} -> {EXPORT_ROOT}\n" + "\n".join(_report))
    else:
        job, cfg = build_job(queue, mode)
        save_preset(cfg)
        unreal.log(f"[mrq_take41] mode={mode} frames={MODES[mode]['start']}-{MODES[mode]['end'] - 1}"
                   f" -> {MODES[mode]['out']}\n" + "\n".join(_report))
    _keep_alive["out"] = MODES[mode]["out"]

    if mode == "diff":
        _keep_alive["perso_saved"] = neutralize_perso()

    executor = unreal.MoviePipelinePIEExecutor()
    executor.on_executor_finished_delegate.add_callable_unique(on_finished)
    _keep_alive.update(queue=queue, executor=executor)
    executor.execute(queue)


def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 and sys.argv[1] else "test")
    if mode == "tout":
        _keep_alive["pending"] = ["diff"]
        mode = "test"
    if mode not in MODES:
        unreal.log_error(f"[mrq_take41] mode inconnu '{mode}' (test | diff | tout | full)")
        return
    render(mode)


main()
