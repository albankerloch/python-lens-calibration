"""Corrections IA avec Runway Aleph 2.0, de l'export Resolve à la génération complète.

Lancement (depuis la racine du dépôt) :
    .venv/Scripts/streamlit run src/runway_app.py

Onglets : 1. export de la timeline Resolve, 2. découpe d'un extrait, 3. préparation du prompt et de
la keyframe, 4. essais sur l'extrait, 5. génération complète (à partir d'un essai réussi), 6. suivi
des tâches, 7. comparaison des résultats.

Les tâches sont créées sans attendre leur fin : plusieurs prompts/seeds peuvent tourner
en parallèle chez Runway, et l'onglet « 6. Suivi » télécharge les résultats quand ils sont prêts.
L'historique est gardé dans video_output/tests/runs.json.

L'assistant de prompt passe par la CLI Claude Code (`claude -p`), donc par l'abonnement Claude
connecté à cette CLI, et non par une clé d'API Anthropic.
"""

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from runwayml import APIError, RunwayML, TaskFailedError, TaskTimeoutError

from runway_aleph_video_to_video import (
    DEFAULT_SEED,
    LAST_TASK_ID_FILE,
    OUTPUT_FORMAT_EXTENSIONS,
    OUTPUT_FORMATS,
    PRORES_PROFILES,
    download_outputs,
)
from runway_ephemeral_upload import VIDEO_CONTENT_TYPES, upload_ephemeral_asset
from resolve import aleph_export

ROOT = Path(__file__).resolve().parent.parent
VIDEO_INPUT_DIR = ROOT / "video_input"
EXTRACTS_DIR = VIDEO_INPUT_DIR / "extraits"
TIMELINE_DIR = VIDEO_INPUT_DIR / "timeline_complete"  # exports complets depuis Resolve (étape 1)
PROMPT_DIR = ROOT / "prompt_input"
# Modèle vidéo pour lequel chaque prompt a été rédigé ; un prompt absent (ancien fichier) est pour Aleph 2.
PROMPT_MODELS_FILE = PROMPT_DIR / "models.json"
OUTPUT_DIR = ROOT / "video_output"  # résultats des générations complètes
TESTS_OUTPUT_DIR = OUTPUT_DIR / "tests"  # résultats des essais sur extraits
RUNS_FILE = TESTS_OUTPUT_DIR / "runs.json"

TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
STATUS_ICONS = {
    "PENDING": "⏳",
    "THROTTLED": "⏳",
    "RUNNING": "🔄",
    "SUCCEEDED": "✅",
    "FAILED": "❌",
    "CANCELLED": "🚫",
}
# Modèles vidéo de retouche (video_to_video) proposés pour les essais sur extraits (onglet 4).
ALEPH_MODEL = "aleph2"
OMNI_MODEL = "gemini_omni_flash"  # sans seed, keyframe ni format : la keyframe part en image de référence
SEEDANCE_MODEL = "seedance2_5"  # génération guidée par la vidéo (mode reference) ; keyframe en image de référence
KLING_MODEL = "kling_4"  # retouche vidéo (runwayml >= 5.24) ; keyframe en image de référence, sortie en 1080p
VIDEO_MODEL_NAMES = {
    ALEPH_MODEL: "Runway Aleph 2", OMNI_MODEL: "Gemini Omni Flash", SEEDANCE_MODEL: "Seedance 2.5",
    KLING_MODEL: "Kling 4.0",
}
SEEDANCE_RATIOS = ["1920:1080", "2206:946", "1664:1248", "1440:1440", "1280:720", "854:480"]  # paysage
SEEDANCE_MIN_SECONDS = 4  # durée de sortie minimale acceptée par l'API
ALEPH_CREDITS_PER_SECOND = 28  # mesuré sur les essais (84 crédits pour 3 s en mp4)
NON_MP4_SURCHARGE = 5  # crédits par seconde de sortie (prores / png_sequence)
CREDIT_USD = 0.01  # prix d'un crédit de l'API Runway
COMPARE_CELL = (960, 540)  # taille de chaque vignette dans la vidéo côte à côte

CLAUDE_MODELS = ["opus", "fable", "sonnet", "haiku"]  # alias acceptés par `claude --model`
CLAUDE_TIMEOUT = 300  # secondes

KEYFRAMES_DIR = TESTS_OUTPUT_DIR / "keyframes"
KEYFRAME_SELECTION_FILE = KEYFRAMES_DIR / "selection.json"  # keyframe retenue par vidéo (onglet 3)
NO_PROMPT = "(aucun prompt : keyframe seule)"
TYPED_PROMPT = "(prompt court saisi ici)"
TYPED_PROMPT_NAME = "prompt_court"  # nom noté dans le suivi et les fichiers de sortie (sans .txt : pas un fichier)
ALEPH_PROMPT_MAX_CHARS = 3500
# Nom donné par cut_extract : <source>_t<début en s>_d<durée en s>
EXTRACT_NAME = re.compile(r"^(?P<source>.+)_t(?P<start>\d+(?:\.\d+)?)_d(?P<duration>\d+(?:\.\d+)?)$")
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
KEYFRAME_MAX_BYTES = 5 * 1024 * 1024  # limite Runway pour une image en data URI
RATIO_TOLERANCE = 0.02  # écart de ratio toléré entre la keyframe et la vidéo

# Modèles de génération d'image Runway qui acceptent des images de référence, avec leurs ratios de
# sortie paysage (bibliothèque runwayml, text_to_image.create).
KEYFRAME_MODELS = {
    "gemini_image3_pro": [
        "1344:768", "1536:672", "1184:864", "1248:832", "1152:896", "1024:1024", "2752:1536",
        "3168:1344", "2528:1696", "2400:1792", "2304:1856", "5504:3072", "6336:2688",
    ],
    "gemini_image3.1_flash": [
        "1344:768", "1536:672", "1184:864", "1248:832", "1152:896", "1024:1024", "2752:1536",
        "3168:1344", "2528:1696", "2400:1792", "2304:1856", "5504:3072", "6336:2688",
    ],
    "gpt_image_2": [
        "1920:1088", "2048:880", "1920:1280", "1920:1440", "2560:1440", "2912:1248", "2560:1712",
        "2560:1920", "3840:2160", "3840:1648", "3504:2336",
    ],
    "gen4_image": [
        "1920:1080", "1280:720", "1360:768", "1808:768", "2112:912", "1680:720", "1440:1080",
        "1168:880", "960:720", "1024:1024", "1080:1080", "720:720",
    ],
    "seedream5_pro": ["1376:768", "1184:896", "1296:864", "2720:1530", "2304:1728", "2496:1664", "1024:1024"],
    "grok_imagine_image_2": [
        "1280:720", "1152:864", "1248:832", "1280:576", "1408:704", "2816:1584", "2368:1776",
        "2496:1664", "2912:1344", "3200:1440", "1024:1024",
    ],
}
# Noms commerciaux, pour l'affichage.
KEYFRAME_MODEL_NAMES = {
    "gemini_image3_pro": "Nano Banana Pro",
    "gemini_image3.1_flash": "Nano Banana 2",
    "gpt_image_2": "GPT Image 2",
    "gen4_image": "Runway Gen-4 Image",
    "seedream5_pro": "Seedream 5 Pro",
    "grok_imagine_image_2": "Grok Imagine Image 2",
}
# Nombre d'images possible en une seule tâche (output_count) ; les autres modèles (gen4_image) lancent
# une tâche par image.
KEYFRAME_COUNTS = {
    "gemini_image3_pro": [1, 4],
    "gemini_image3.1_flash": [1, 4],
    "gpt_image_2": [1, 2, 3, 4],
    "seedream5_pro": [1, 2, 3, 4],
    "grok_imagine_image_2": [1, 2, 3, 4],
}
# Modèles sans tag sur les images de référence : @shot et @look sont réécrits selon l'ordre des images.
KEYFRAME_UNTAGGED = {
    "seedream5_pro": ("Figure 1", "Figure 2"),
    "grok_imagine_image_2": ("the first reference image", "the second reference image"),
}
KEYFRAME_GEN_TIMEOUT = 300  # secondes
KEYFRAME_OUTPUT_WIDTH = 2048  # keyframes générées enregistrées en 2K, au ratio de la vidéo
KF_MODE_GENERATE = "Générer avec Runway à partir d'une image de référence de look"
KF_MODE_IMPORT = "Importer une image retouchée"
KEYFRAME_DEFAULT_PROMPT = (
    "Relight @shot with the lighting of @look: same light direction and softness, color temperature, "
    "contrast, shadow depth and color palette. Keep everything else in @shot exactly as it is: "
    "composition, framing, camera angle, people, faces, expressions, poses, clothing, set and objects."
)

# Prompt Aleph court pour accompagner une keyframe déjà choisie : il décrit ce que la keyframe montre
# vraiment, pour ne pas donner à Aleph une consigne différente de l'image.
KEYFRAME_SHORT_SYSTEM_PROMPT = """You write the text prompt sent to Runway Aleph 2.0 together with a keyframe.
Aleph 2.0 receives the video, one keyframe (an edited version of one frame of that video) and this single
text prompt; it propagates the keyframe edit to the whole shot and preserves everything else.
Runway's guide: with a keyframe the text prompt is optional, and its examples pair a keyframe with a short
prompt naming the same edit ("Chiaroscuro lighting", "change background to ancient city ruins"). Motion is
only worth describing when the edit needs motion that exists neither in the keyframe nor in the video
(e.g. "fire begins to spread up the trees").

Two images are provided: ./original.png is the frame of the video, ./keyframe.png is the same frame after
the edit. Read both and compare them.

Write one English sentence (at most 25 words) that:
- starts with an action verb (re-light, change, add, replace...);
- names only the difference you actually SEE between original.png and keyframe.png (light direction,
  hardness, color temperature, contrast, shadows, palette...), never something the keyframe does not show;
- does not list what to keep and uses no negation;
- adds motion only if the user's intent asks for something moving that the keyframe cannot show.

Return the requested JSON: "analysis" in French, 2 sentences (the difference you see), "prompt" in English.
"""

# Même rôle pour Gemini Omni Flash, qui reçoit la keyframe comme image de référence (sans instant).
OMNI_KEYFRAME_SHORT_SYSTEM_PROMPT = """You write the text prompt sent to Gemini Omni Flash (video editing)
together with one reference image. The model receives the video, this reference image (an edited version
of one frame of that video, but without its timestamp) and this single instruction; it applies the edit
and preserves what the instruction does not mention.
Google's guide: simple prompts work best, overly descriptive prompts lead to unintended changes; ask for
one change and end with "Keep everything else the same." Say what the reference image is for, otherwise
the model may treat it as a style, a character or something to insert.

Two images are provided: ./original.png is the frame of the video, ./keyframe.png is the same frame after
the edit (it will be the reference image). Read both and compare them.

Write the prompt in English, at most 35 words, so that it:
- starts with an action verb and names the main difference you actually SEE between original.png and
  keyframe.png, never something the keyframe does not show;
- says to match the reference image for that change (e.g. "Change the background to match the reference
  image.");
- if the difference is several unrelated changes, keeps the most visible one;
- adds an explicit "Do not ..." only for what the user's intent says must not change (e.g. the actor's
  face or hands): this model takes no negative prompt, negatives go in the instruction;
- ends with "Keep everything else the same."

Return the requested JSON: "analysis" in French, 2 sentences (the difference you see), "prompt" in English.
"""

# Même rôle pour Seedance 2.5 (mode reference), qui reçoit la keyframe comme image de référence « Image 1 ».
SEEDANCE_KEYFRAME_SHORT_SYSTEM_PROMPT = """You write the text prompt sent to Seedance 2.5 (Runway API,
"reference" mode) together with one reference image. The model receives the source clip ("Video 1") and
this reference image ("Image 1", an edited version of one frame of that clip) and GENERATES a new video:
it may redraw the actor, so the prompt must protect him explicitly.
Seedance guide: target, change, protect, restrict; map each reference image to one narrow job; preserve
the original action and timing instead of re-describing it; no vague wording.

Two images are provided: ./original.png is the frame of the video, ./keyframe.png is the same frame after
the edit (it will be Image 1). Read both and compare them.

Write the prompt in English, at most 70 words:
- "Edit Video 1:" then the change you actually SEE between original.png and keyframe.png, asking to
  match Image 1 for that change only (e.g. "use Image 1 only for the set and the lighting");
- protect: the man's identity, face, hair, clothing, pose, the objects in his hands, the original action
  and timing exactly, the same framing and camera motion;
- restrict: no added people, no extra effects, no subtitles, no cuts.

Return the requested JSON: "analysis" in French, 2 sentences (the difference you see), "prompt" in English.
"""

KEYFRAME_SHORT_SCHEMA = {
    "type": "object",
    "properties": {"analysis": {"type": "string"}, "prompt": {"type": "string"}},
    "required": ["analysis", "prompt"],
    "additionalProperties": False,
}

# Règles tirées du guide officiel (help.runwayml.com, « Aleph 2.0 Prompting Guide », juin 2026).
ALEPH_RULES = """You write text prompts for Runway Aleph 2.0, an in-context video editing model.
Aleph 2.0 changes only what the prompt asks for and preserves everything else (background, lighting,
surrounding details, the subject's performance, framing and camera motion).

Rules from Runway's official Aleph 2.0 prompting guide:
- Simplicity, iteration and precision work best. Start straightforward, add detail only when needed.
- An effective prompt has two parts: an action verb (add, remove, change, replace, re-light, re-style,
  swap...) and a description of the desired transformation.
  Examples: "Change the outfit and bag to soft yellow", "change background to ancient city ruins",
  "heavy overcast and raining", "Chiaroscuro lighting", "Swap for black leather clutch with gold buckle",
  "1920s silent era film".
- Targeted, specific language gives the best results. For relighting, describe the light source,
  direction, quality and color.
- Several edits can be combined in one generation, but each one must stay explicit and concrete.
- With a keyframe, describing motion is only useful when the edit needs motion that exists neither in
  the keyframe nor in the source video (e.g. "fire begins to spread up the trees").
- A keyframe is an edited version of one frame of the input video: Aleph propagates that edit to the
  whole clip. It is not a style reference.
The Aleph API takes a single text prompt per generation. The "prompt" and "variants" you write here are
for a generation WITHOUT keyframe, so they must carry the whole edit.

How to write the prompt:
- Write the Aleph prompt in English, as one or two short sentences (usually under 40 words).
- Begin with the action verb. Describe only what must change: do not list what to preserve, and do not
  describe the whole scene the way an image-generation prompt would.
- Do not use negations or lists of forbidden things; state the desired result instead.
- If the user's intent bundles many unrelated changes, keep the most important ones in the main prompt
  and say in the advice which changes are better done in a separate pass.

"""

# Règles tirées de la doc Google (ai.google.dev/gemini-api/docs/omni) et de la page Runway du modèle.
OMNI_RULES = """You write text prompts for Gemini Omni Flash, a video editing model called through
Runway's API: it receives the video, one text instruction and optionally reference images, and applies
the edit while preserving what the instruction does not mention.

Rules from Google's Gemini Omni Flash guide and Runway's model page:
- "Simple prompts work best for video editing. Overly descriptive prompts can lead to unintended
  changes." Keep the instruction short and focused.
- Ask for ONE change per generation. Several unrelated changes are done in separate passes.
- End the instruction with "Keep everything else the same." to keep the rest consistent.
  Example rewrites from the guide: "Make the phone invisible. Keep everything else the same.",
  "Add a cat that jumps onto his lap, he begins to pet it. Keep everything else the same."
- There is no negative prompt: anything that must not happen goes in the instruction ("Do not ...").
- Say what each reference image is for (background, style, object...), otherwise the model guesses.
- Mention "widescreen" when the framing must stay 16:9 (the model may otherwise pick 9:16), and
  "no cuts" when needed: it must stay one continuous shot.
- English only. Output is 720p on Runway, inputs up to 10 s.
The Runway API takes a single instruction per generation and no multi-turn editing: every generation
starts again from the source clip.

How to write the prompt:
- Write it in English, one to three short sentences (usually under 35 words), starting with an action
  verb (change, replace, add, remove, relight...).
- The main "prompt" works WITHOUT reference image, so it carries the whole change in words.
- The first variant is for use WITH the retained keyframe sent as reference image (an edited frame of
  this clip): it asks to match the reference image for the change, instead of describing it.
- If the user says something must stay untouched (an actor's face, hands...), add one short explicit
  "Do not change ..." sentence: here, unlike for Aleph, a negation is the recommended way.
- If the user's intent bundles many unrelated changes, keep the most important one and say in the advice
  which changes are better done in another pass.

"""

# Règles tirées du guide Seedance 2.5 (docs.seedance.tv, « Video Editing, Extension, and Final Assembly »)
# et de guides tiers concordants.
SEEDANCE_RULES = """You write text prompts for Seedance 2.5 called through Runway's API in "reference" mode:
it receives the source clip and optional reference images, and GENERATES a new video conditioned on them.
Unlike an in-place editor, it may redraw the actor, so the prompt must protect him explicitly.
In the prompt the source clip is "Video 1" and the first reference image is "Image 1" (plain labels, no @).

Rules from the Seedance 2.5 video editing guide:
- Structure the prompt in four parts: target ("Edit Video 1"), change (what is replaced, and with what),
  protect (what must stay the same), restrict (unwanted additions).
- Protect explicitly: the performer's identity, face, hair, clothing, pose, action order, speed and
  timing, the camera framing and motion. Say "preserve the original action and timing exactly"; do NOT
  re-describe the action, it causes drift.
- Map each reference image to one narrow job (e.g. "use Image 1 only for the set and the lighting"),
  otherwise the model may also take its people, pose or framing.
- If something conflicts, identity comes first, then the props in hand, wardrobe, then the set.
- Restrict: no extra effects, no subtitles, no cuts, no added people.
- Avoid vague wording ("connect naturally", "make it better"): name the change concretely.
- English. Output is 1080p at most, no audio requested.
The Runway API takes a single prompt per generation; every generation starts from the source clip.

How to write the prompt:
- Write it in English, a short focused paragraph (usually 40 to 80 words), following the four parts.
- The main "prompt" works WITHOUT reference image: it describes the new set and light in words.
- The first variant is for use WITH the retained keyframe sent as Image 1 (an edited frame of this clip):
  it asks to match Image 1 for the set and light only, and still protects the actor in words.

"""

# Règles tirées du guide officiel Kling 4.0 (kling.ai/quickstart/kling-4-model-user-guide).
KLING_RULES = """You write text prompts for Kling 4.0 in video editing mode, called through Runway's API:
it receives the source clip and optional reference images, edits the clip and keeps its framing.
Inputs are numbered in upload order: the source clip is "@Video1", the first reference image "@Image1".

Rules from Kling's official Kling 4.0 user guide:
- Start with "Edit @Video1" and state clearly what to change and how ("Edit @Video1, change ... to ...").
  Never write "Reference @Video1": it may trigger reference-based generation instead of an edit.
- Give each reference image its role (e.g. "use @Image1 for the kitchen set and the lighting"): asset
  IDs without roles make results less predictable.
- Name what must be preserved, otherwise other elements may change: the person, his actions and timing,
  the camera movement, the composition. End with "Keep everything else unchanged."
- Use descriptive, positive language and minimal negative instructions: negative prompts increase the
  likelihood of errors. The only exclusion worth writing is "no subtitles".
- English. Inputs 3 to 30 s; output keeps the input framing, in 1080p here.
The Runway API takes a single prompt per generation; every generation starts from the source clip.

How to write the prompt:
- Write it in English, one short paragraph (usually 30 to 70 words), starting with "Edit @Video1".
- The main "prompt" works WITHOUT reference image: it describes the change in words.
- The first variant is for use WITH the retained keyframe sent as @Image1 (an edited frame of this
  clip): it says what to take from @Image1 (e.g. the set and the lighting), not the whole image.
- Preserve the actor positively ("keep the man, his performance and timing exactly as in @Video1"),
  not with a list of prohibitions.

"""

# Même rôle que KEYFRAME_SHORT_SYSTEM_PROMPT pour Kling 4.0, qui reçoit la keyframe comme « @Image1 ».
KLING_KEYFRAME_SHORT_SYSTEM_PROMPT = """You write the text prompt sent to Kling 4.0 (video editing, Runway
API) together with one reference image. The model receives the source clip ("@Video1") and this reference
image ("@Image1", an edited version of one frame of that clip), edits the clip and keeps its framing.
Kling's guide: start with "Edit @Video1", give @Image1 a role, name what must be preserved and end with
"Keep everything else unchanged."; descriptive positive language, minimal negatives.

Two images are provided: ./original.png is the frame of the video, ./keyframe.png is the same frame after
the edit (it will be @Image1). Read both and compare them.

Write the prompt in English, at most 60 words:
- "Edit @Video1:" then the change you actually SEE between original.png and keyframe.png, taking only
  that from @Image1 (e.g. "replace the set and the lighting with those of @Image1");
- keep the man, his performance and timing exactly as in @Video1, and the camera movement;
- end with "Keep everything else unchanged."

Return the requested JSON: "analysis" in French, 2 sentences (the difference you see), "prompt" in English.
"""

# Contexte et second prompt (keyframe), communs aux modèles ; <MODEL> = nom du modèle vidéo.
PROMPT_CONTEXT = """Context: this is a virtual-production pipeline. The footage is usually an actor shot on a
green screen, keyed and composited over an Unreal Engine render in DaVinci Resolve; <MODEL> is used
afterwards to make that composite look photographed in one place (light wrap, matching key and rim light,
contact shadows, consistent atmosphere, grain and color), or to replace the set. If a frame is provided,
read it with the Read tool and base the prompt on what you actually see: light direction, colors,
materials, where the integration fails.
If a look reference image is provided (./reference.*), it usually comes from another film: read it,
then translate its look into words in the prompt (light source, direction, hardness, color temperature,
contrast, palette, texture). Never ask <MODEL> to copy its content (people, set, objects, framing)
and never suggest sending it to <MODEL> as a keyframe or reference image.

Second output, "keyframe_prompt": only when a look reference image is provided (otherwise return an
empty string). It is a prompt for a different model, a Runway image-editing model, that will create the
keyframe from two tagged reference images: @shot is ./frame.png (the frame to edit) and @look is the look
reference. Unlike <MODEL>, this model regenerates the whole image, so the edited frame must be told to stay
the SAME shot. Write it in English, at most 900 characters, so that it:
- starts by asking to relight / regrade @shot, then describes the target look concretely, from what you
  see in @look: light sources, direction, hardness, color temperature, contrast, shadow depth, palette,
  atmosphere and texture;
- adapts that light to the geometry of @shot (where the light can come from in this set);
- explicitly asks to keep everything else of @shot unchanged: composition, framing, camera angle,
  people, faces, expressions, poses, clothing, set and objects;
- never asks to copy content from @look (people, set, objects, framing);
- describes the same edit as the <MODEL> "prompt", so that both stay consistent.

Return the requested JSON. "analysis" and "advice" are in French and short (2 to 4 sentences each):
"analysis" says what you see and how you interpreted the intent; "advice" says what to check in the
first test and whether a keyframe is worth it. Give 2 variants that explore a different angle each
(e.g. stronger or softer, a different light description); each "angle" is a few French words.
"""

# Prompt système de l'assistant selon le modèle vidéo visé.
ASSISTANT_SYSTEM_PROMPTS = {
    ALEPH_MODEL: ALEPH_RULES + PROMPT_CONTEXT.replace("<MODEL>", "Aleph"),
    OMNI_MODEL: OMNI_RULES + PROMPT_CONTEXT.replace("<MODEL>", "Gemini Omni Flash"),
    SEEDANCE_MODEL: SEEDANCE_RULES + PROMPT_CONTEXT.replace("<MODEL>", "Seedance"),
    KLING_MODEL: KLING_RULES + PROMPT_CONTEXT.replace("<MODEL>", "Kling"),
}
KEYFRAME_SHORT_SYSTEM_PROMPTS = {
    ALEPH_MODEL: KEYFRAME_SHORT_SYSTEM_PROMPT,
    OMNI_MODEL: OMNI_KEYFRAME_SHORT_SYSTEM_PROMPT,
    SEEDANCE_MODEL: SEEDANCE_KEYFRAME_SHORT_SYSTEM_PROMPT,
    KLING_MODEL: KLING_KEYFRAME_SHORT_SYSTEM_PROMPT,
}

ALEPH_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "prompt": {"type": "string"},
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"angle": {"type": "string"}, "prompt": {"type": "string"}},
                "required": ["angle", "prompt"],
                "additionalProperties": False,
            },
        },
        "keyframe_prompt": {"type": "string"},
        "advice": {"type": "string"},
    },
    "required": ["analysis", "prompt", "variants", "keyframe_prompt", "advice"],
    "additionalProperties": False,
}

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Utilitaires fichiers / ffmpeg
# ---------------------------------------------------------------------------

def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-2000:])
    return proc


@st.cache_data(show_spinner=False)
def probe_video(path: str, mtime: float) -> dict:
    """Durée, résolution et cadence de la première piste vidéo (`mtime` sert de clé de cache)."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate",
            "-of", "json", path,
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    num, _, den = stream.get("r_frame_rate", "25/1").partition("/")
    return {
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": float(num) / float(den or 1),
    }


def video_info(path: Path) -> dict:
    return probe_video(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=256)
def extract_frame(path: str, mtime: float, t: float, width: int = 1280) -> bytes:
    """Image PNG à l'instant `t` (s), redimensionnée à `width` px de large."""
    proc = run_ffmpeg([
        "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
        "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-",
    ])
    if not proc.stdout:
        raise RuntimeError(f"Aucune image à {t:.2f} s dans {path}")
    return proc.stdout


def frame_of(path: Path, t: float, width: int = 1280) -> bytes:
    return extract_frame(str(path), path.stat().st_mtime, t, width)


def list_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    # Les vidéos illisibles (ex. rendu Resolve en cours, sans « moov atom ») sont ignorées.
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in VIDEO_CONTENT_TYPES and video_info(p)["width"]
    )


def cut_extract(source: Path, start: float, duration: float) -> Path:
    """Découpe `duration` s de `source` à partir de `start`, image seule, réduite pour tenir dans 1080p,
    en AV1 10 bits (même encodage que le preset Resolve « Aleph AV1 »)."""
    EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EXTRACTS_DIR / f"{source.stem}_t{start:.2f}_d{duration:.1f}.mp4"
    info = video_info(source)
    width, height = aleph_export.aleph_size(info["width"], info["height"])
    common_in = ["-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}", "-map", "0:v:0", "-an",
                 "-vf", f"scale={width}:{height}:flags=lanczos"]
    tags = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    try:
        run_ffmpeg([
            *common_in, "-c:v", "av1_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
            "-b:v", "30M", "-maxrate", "40M", "-bufsize", "60M", "-pix_fmt", "p010le", *tags, str(out),
        ])
    except RuntimeError:  # pas de NVENC AV1 : encodeur logiciel
        run_ffmpeg([
            *common_in, "-c:v", "libsvtav1", "-crf", "18", "-preset", "6", "-pix_fmt", "yuv420p10le",
            *tags, str(out),
        ])
    return out


def timecode(frame: int, fps: float) -> str:
    fps_i = int(round(fps))
    s, ff = divmod(frame, fps_i)
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{ff:02d}"


def resolve_export_section() -> None:
    """Export d'une timeline Resolve complète (≤ 1080p, ≤ 30 i/s) dans video_input/timeline_complete."""
    # Streamlit exécute tous les onglets à chaque interaction : sans ce verrou, chaque clic des autres
    # onglets tenterait de joindre Resolve (lent quand il est fermé).
    if not st.session_state.get("rs_connected"):
        st.caption("Les autres onglets fonctionnent sans Resolve.")
        if not st.button("Se connecter à Resolve"):
            return
        st.session_state.rs_connected = True
    if st.button("Rafraîchir depuis Resolve"):
        st.session_state.pop("rs_timeline", None)
    try:
        resolve = aleph_export.connect()
        names, opened = aleph_export.list_timelines(resolve)
    except RuntimeError as exc:
        # Déconnecté : la prochaine tentative repassera par le bouton de connexion.
        st.session_state.rs_connected = False
        st.error(str(exc))
        return
    if not names:
        st.error("Aucune timeline dans le projet Resolve ouvert.")
        return
    if st.session_state.get("rs_timeline") not in names:
        st.session_state.rs_timeline = opened if opened in names else names[0]
    c1, c2 = st.columns(2)
    timeline_name = c1.selectbox(
        "Timeline à exporter", names, key="rs_timeline",
        format_func=lambda n: f"{n}  (ouverte dans Resolve)" if n == opened else n,
    )
    choice = c2.selectbox("Codec", list(aleph_export.CODEC_CHOICES), key="rs_codec")
    try:
        info = aleph_export.timeline_info(resolve, timeline_name)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    fps = info["fps"]
    total = info["frames"] / fps
    out_w, out_h = aleph_export.aleph_size(info["width"], info["height"])
    size = aleph_export.estimated_size_mb(choice, out_w, out_h, fps, total)
    st.caption(
        f"**{info['timeline']}** ({info['project']}) · {info['width']}×{info['height']} → "
        f"rendu {out_w}×{out_h} · {fps:g} i/s · {total:.2f} s ({info['frames']} images) · "
        f"début {timecode(info['start_frame'], fps)} · ≈ {size:.0f} Mo · sans audio"
    )
    if fps > aleph_export.ALEPH_MAX_FPS + 1e-3:
        st.error(
            f"Timeline à {fps:g} i/s : Aleph accepte 30 i/s maximum. Créez une timeline à 25 ou 30 i/s "
            "(Timeline Settings sans « Use project settings »)."
        )
        return
    if timeline_name != opened:
        st.caption("Resolve ne rend que la timeline ouverte : elle sera ouverte le temps du rendu, "
                   "puis la timeline actuelle sera rétablie.")

    safe_timeline = re.sub(r"[^\w-]+", "_", info["timeline"])
    target = TIMELINE_DIR / f"{safe_timeline}.mp4"
    if target.is_file():
        st.caption(f"`{rel(target)}` existe déjà et sera remplacé.")
    if st.button("Exporter la timeline complète", type="primary"):
        bar = st.progress(0, text="Rendu dans Resolve...")
        try:
            out = aleph_export.render_full_timeline(
                resolve, timeline_name, TIMELINE_DIR, safe_timeline, choice,
                on_progress=lambda p: bar.progress(min(p, 100), text=f"Rendu dans Resolve... {p} %"),
            )
        except (RuntimeError, ValueError) as exc:
            bar.empty()
            st.error(str(exc))
            return
        bar.progress(100, text="Rendu terminé")
        i = video_info(out)
        st.success(
            f"Timeline exportée : {rel(out)} — {i['width']}×{i['height']}, {i['fps']:g} i/s, "
            f"{i['duration']:.2f} s, {out.stat().st_size / 1e6:.1f} Mo. "
            "Découpez-en un extrait dans l'onglet « 2. Extrait »."
        )

    exports = list_videos(TIMELINE_DIR)
    if exports:
        st.subheader("Timelines exportées")
        for p in exports:
            i = video_info(p)
            st.write(f"- `{rel(p)}` — {i['duration']:.2f} s, {i['width']}×{i['height']}, {i['fps']:g} i/s")


def image_size(data: bytes) -> tuple[int, int] | None:
    """(largeur, hauteur) d'une image encodée, ou None si elle est illisible."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    return None if img is None else (img.shape[1], img.shape[0])


def image_data_uri(path: Path) -> str:
    """Data URI de l'image, réencodée en JPEG si elle dépasse la limite Runway de 5 Mo."""
    return bytes_data_uri(path.read_bytes(), path.suffix)


def export_full_frame(clip: Path, t: float) -> Path:
    """Image pleine résolution de `clip` à l'instant `t`, à retoucher dans Resolve pour en faire une keyframe."""
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out = KEYFRAMES_DIR / f"{clip.stem}_t{t:.2f}_original.png"
    run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(clip), "-frames:v", "1", str(out)])
    return out


def build_comparison(videos: list[Path]) -> Path:
    """Assemble les vidéos côte à côte, chacune ramenée à la taille de vignette (16:9)."""
    key = hashlib.sha1("|".join(f"{v}:{v.stat().st_mtime}" for v in videos).encode()).hexdigest()[:10]
    out = TESTS_OUTPUT_DIR / f"compare_{key}.mp4"
    if out.is_file():
        return out
    w, h = COMPARE_CELL
    inputs, filters = [], []
    for i, v in enumerate(videos):
        inputs += ["-i", str(v)]
        filters.append(f"[{i}:v]scale={w}:{h},setsar=1[v{i}]")
    stack = "".join(f"[v{i}]" for i in range(len(videos)))
    filters.append(f"{stack}hstack=inputs={len(videos)}:shortest=1[out]")
    TESTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out),
    ])
    return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def list_prompts() -> list[str]:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in PROMPT_DIR.glob("*.txt"))


def next_prompt_version(name: str) -> Path:
    """realistic.txt -> realistic_v2.txt, realistic_v2.txt -> realistic_v3.txt, etc."""
    base = re.sub(r"_v\d+$", "", Path(name).stem)
    n = 2
    while (PROMPT_DIR / f"{base}_v{n}.txt").exists():
        n += 1
    return PROMPT_DIR / f"{base}_v{n}.txt"


def prompt_widget_key(name: str) -> str:
    return f"prompt_text::{name}"


def load_prompt_models() -> dict:
    if not PROMPT_MODELS_FILE.is_file():
        return {}
    return json.loads(PROMPT_MODELS_FILE.read_text(encoding="utf-8"))


def prompt_model(name: str) -> str:
    """Modèle vidéo du prompt enregistré `name` (Aleph 2 s'il n'est pas noté)."""
    return load_prompt_models().get(name, ALEPH_MODEL)


def set_prompt_model(name: str, model: str) -> None:
    models = load_prompt_models()
    if model == ALEPH_MODEL:
        models.pop(name, None)  # valeur par défaut : rien à noter
    else:
        models[name] = model
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_MODELS_FILE.write_text(json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8")


def select_prompt_model(prompt_key: str, model_keys: tuple[str, ...]) -> None:
    """Callback : à l'ouverture d'un prompt enregistré, remet son modèle vidéo (Aleph 2 par défaut)
    dans les sélecteurs `model_keys`."""
    name = st.session_state.get(prompt_key)
    if name in (TYPED_PROMPT, NO_PROMPT, None):
        return
    for key in model_keys:
        st.session_state[key] = prompt_model(name)


# ---------------------------------------------------------------------------
# Assistant de prompt (Claude Code CLI)
# ---------------------------------------------------------------------------

def run_claude(request: list[str], files: dict[str, bytes], system: str, schema: dict, model: str) -> dict:
    """Appelle `claude -p` (donc l'abonnement de la CLI) et renvoie sa réponse JSON validée par `schema`.

    Claude tourne dans un dossier temporaire qui ne contient que `files`, sans MCP ni réglages du
    projet, avec pour seul outil Read (pour lire les images)."""
    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("CLI `claude` introuvable dans le PATH (installez Claude Code et connectez-vous).")
    with tempfile.TemporaryDirectory(prefix="aleph_prompt_") as workdir:
        for name, data in files.items():
            Path(workdir, name).write_bytes(data)
        # Sans ANTHROPIC_API_KEY, la CLI utilise le compte connecté (abonnement) et non une clé d'API.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = subprocess.run(
            [
                claude, "-p", "\n\n".join(request),
                "--output-format", "json",
                "--model", model,
                "--system-prompt", system,
                "--json-schema", json.dumps(schema),
                "--tools", "Read", "--allowedTools", "Read",
                "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
            ],
            cwd=workdir, env=env, capture_output=True, text=True, encoding="utf-8",
            timeout=CLAUDE_TIMEOUT,
        )
    try:
        reply = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError((proc.stderr or proc.stdout or "réponse vide")[-2000:]) from None
    if reply.get("is_error") or not reply.get("structured_output"):
        raise RuntimeError(str(reply.get("result") or reply.get("subtype") or reply)[-2000:])
    return reply["structured_output"]


def ask_claude_for_aleph_prompt(
    intent: str,
    current_prompt: str,
    frame_png: bytes | None,
    model: str,
    reference: tuple[bytes, str] | None = None,
    video_model: str = ALEPH_MODEL,
) -> dict:
    """Un seul appel pour le prompt du modèle vidéo `video_model` (règles propres à chaque modèle) et, si
    `reference` est donnée, le prompt de retouche de la keyframe. `frame_png` est l'image du plan à
    l'instant de la keyframe (@shot) ; `reference` = (contenu, extension) de l'image de référence de look (@look)."""
    request = [f"User intent (may be in French):\n{intent.strip()}"]
    if current_prompt.strip():
        request.append(f"Current prompt, to improve or replace:\n{current_prompt.strip()}")
    files = {}
    if frame_png:
        files["frame.png"] = frame_png
        request.append("A frame of the clip (the keyframe instant, @shot) is in ./frame.png: read it first.")
    else:
        request.append("No frame is provided.")
    if reference:
        ref_name = f"reference{reference[1]}"
        files[ref_name] = reference[0]
        request.append(
            f"A look reference image (@look) is in ./{ref_name}: read it, describe its look in words in the "
            "video prompt, and write the keyframe_prompt."
        )
    else:
        request.append("No look reference image: return an empty keyframe_prompt.")
    return run_claude(request, files, ASSISTANT_SYSTEM_PROMPTS[video_model], ALEPH_PROMPT_SCHEMA, model)


def ask_claude_for_keyframe_short_prompt(
    original_png: bytes, keyframe_png: bytes, intent: str, model: str, video_model: str = ALEPH_MODEL,
) -> dict:
    """Prompt court, pour `video_model`, qui décrit la différence visible entre l'image d'origine et la
    keyframe retenue."""
    request = [
        f"User intent (may be in French, may be empty):\n{intent.strip() or '(none)'}",
        "The original frame is ./original.png and the edited keyframe is ./keyframe.png: read both first.",
    ]
    return run_claude(
        request, {"original.png": original_png, "keyframe.png": keyframe_png},
        KEYFRAME_SHORT_SYSTEM_PROMPTS[video_model], KEYFRAME_SHORT_SCHEMA, model,
    )


# ---------------------------------------------------------------------------
# Keyframes générées (génération d'image Runway)
# ---------------------------------------------------------------------------

def bytes_data_uri(data: bytes, suffix: str) -> str:
    """Data URI d'une image encodée, réencodée en JPEG si elle dépasse la limite Runway de 5 Mo."""
    suffix = suffix.lower()
    mime = f"image/{'jpeg' if suffix in ('.jpg', '.jpeg') else suffix[1:]}"
    if len(data) > KEYFRAME_MAX_BYTES * 3 // 4:  # le base64 grossit de 4/3
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def closest_ratio(model: str, width: int, height: int) -> tuple[int, int]:
    """Ratio de sortie du modèle le plus proche de width/height, puis la plus petite taille au moins aussi
    large que la vidéo (sinon la plus proche)."""
    target = np.log(width / height)
    sizes = [tuple(map(int, r.split(":"))) for r in KEYFRAME_MODELS[model]]
    return min(sizes, key=lambda s: (
        round(abs(np.log(s[0] / s[1]) - target), 3), s[0] < width, abs(s[0] - width),
    ))


def look_reference(uploaded, container) -> tuple[bytes, str]:
    """Image de référence de look (contenu, extension), affichée dans `container`."""
    container.image(uploaded.getvalue(), width=320)
    return uploaded.getvalue(), Path(uploaded.name).suffix.lower()


def keyframe_output_size(width: int, height: int) -> tuple[int, int]:
    """Taille 2K des keyframes générées au ratio width/height (2048×1152 pour du 16:9), hauteur paire."""
    return KEYFRAME_OUTPUT_WIDTH, round(KEYFRAME_OUTPUT_WIDTH * height / width / 2) * 2


def generate_keyframes(
    client: RunwayML, clip: Path, t: float, reference: tuple[bytes, str], prompt: str, model: str, count: int,
) -> tuple[list[Path], int]:
    """Génère `count` keyframes : l'image de `clip` à `t` (@shot) reprise avec le look de `reference` (@look).
    Renvoie les images, enregistrées en 2K au ratio de la vidéo, et les crédits consommés.

    Les vidéos sont en 16:9 : l'image est simplement redimensionnée au ratio du modèle le plus proche
    (ex. 2752:1536 pour gemini_image3_pro, 0,8 % d'écart), puis remise en 2K. gen4_image plafonne à
    1920:1080 : ses images sont légèrement agrandies."""
    info = video_info(clip)
    final_w, final_h = keyframe_output_size(info["width"], info["height"])
    shot = cv2.imdecode(np.frombuffer(frame_of(clip, t, info["width"]), np.uint8), cv2.IMREAD_COLOR)
    out_w, out_h = closest_ratio(model, final_w, final_h)
    shot = cv2.resize(shot, (out_w, out_h), interpolation=cv2.INTER_AREA)
    references = [
        {"uri": bytes_data_uri(cv2.imencode(".png", shot)[1].tobytes(), ".png"), "tag": "shot"},
        {"uri": bytes_data_uri(*reference), "tag": "look"},
    ]
    if model in KEYFRAME_UNTAGGED:
        shot_name, look_name = KEYFRAME_UNTAGGED[model]
        prompt = prompt.replace("@shot", shot_name).replace("@look", look_name)
        references = [{"uri": r["uri"]} for r in references]
    base = {"model": model, "prompt_text": prompt, "ratio": f"{out_w}:{out_h}", "reference_images": references}
    if model == "grok_imagine_image_2":
        base["edit"] = True  # retouche des images de référence plutôt que simple inspiration
    if model in KEYFRAME_COUNTS:
        jobs = [{**base, "output_count": count}]  # une seule tâche pour toutes les images
    else:
        jobs = [{**base, "seed": DEFAULT_SEED + i} for i in range(count)]

    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    saved, credits = [], 0
    tasks = [client.text_to_image.create(**job) for job in jobs]  # lancées en parallèle chez Runway
    for task in tasks:
        details = task.wait_for_task_output(timeout=KEYFRAME_GEN_TIMEOUT)
        credits += details.cost.credits
        for i, url in enumerate(details.output):
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            out = KEYFRAMES_DIR / f"{clip.stem}_t{t:.2f}_{model}_{details.id[:8]}_{i}.png"
            shrink = img.shape[1] > final_w
            cv2.imwrite(str(out), cv2.resize(
                img, (final_w, final_h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4,
            ))
            saved.append(out)
    return saved, credits


def use_suggested_prompt(text_key: str, text: str, video_model: str | None = None) -> None:
    """Callback : met la proposition dans la zone de prompt (autorisé seulement dans un callback) et
    présélectionne dans l'onglet 4 le modèle vidéo pour lequel elle a été rédigée (noté aussi pour
    l'enregistrement de la zone)."""
    st.session_state[text_key] = text
    if video_model:
        st.session_state.gx_model = st.session_state[f"{text_key}_model"] = video_model


def use_for_keyframe(text: str, mode_key: str) -> None:
    """Callback : met le prompt de retouche proposé par Claude dans le champ de la section Keyframe,
    en passant celle-ci en mode génération (le champ n'existe que dans ce mode)."""
    st.session_state[mode_key] = KF_MODE_GENERATE
    st.session_state.kf_gen_prompt = text


def prompt_name_field(name_key: str, suggestion: str, label: str) -> None:
    """Champ du nom de fichier pour « Enregistrer » : la suggestion automatique est remise à jour quand
    elle devient prise ; un nom tapé à la main est conservé, et signalé s'il existe déjà."""
    auto_key = f"{name_key}_auto"
    current = st.session_state.get(name_key, "").strip()
    taken = (PROMPT_DIR / f"{current.removesuffix('.txt')}.txt").exists()
    if not current or (taken and current == st.session_state.get(auto_key)):
        st.session_state[name_key] = st.session_state[auto_key] = suggestion
        taken = False
    st.text_input(label, key=name_key, help="Enregistré dans prompt_input/ ; un fichier existant n'est jamais écrasé.")
    if taken:
        st.warning(f"prompt_input/{current} existe déjà : choisissez un autre nom.")


# ---------------------------------------------------------------------------
# Historique des tâches
# ---------------------------------------------------------------------------

def load_runs() -> list[dict]:
    if not RUNS_FILE.is_file():
        return []
    return json.loads(RUNS_FILE.read_text(encoding="utf-8"))


def save_runs(runs: list[dict]) -> None:
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


def save_last_task_id(task_id: str) -> None:
    path = ROOT / LAST_TASK_ID_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task_id, encoding="utf-8")


@st.cache_resource
def get_client() -> RunwayML:
    return RunwayML()


def output_base_path(run: dict) -> Path:
    clip_stem = Path(run["clip"]).stem if run.get("clip") else "import"
    prompt_name = run.get("prompt_name")
    prompt_stem = Path(prompt_name).stem if prompt_name and prompt_name != NO_PROMPT else "keyframe"
    ext = OUTPUT_FORMAT_EXTENSIONS[run.get("output_format", "mp4")]
    directory = OUTPUT_DIR if run.get("kind") == "complete" else TESTS_OUTPUT_DIR
    model = run.get("model", ALEPH_MODEL)
    middle = f"__s{run.get('seed')}" if model == ALEPH_MODEL else f"__{model}"
    return directory / f"{clip_stem}__{prompt_stem}{middle}__{run['id'][:8]}{ext}"


def run_model_label(run: dict) -> str:
    """Modèle et seed d'un run, pour les listes (les anciens runs, sans « model », sont des runs Aleph)."""
    model = run.get("model", ALEPH_MODEL)
    if model == ALEPH_MODEL:
        return f"seed {run.get('seed')}"
    return VIDEO_MODEL_NAMES.get(model, model)


def refresh_run(client: RunwayML, run: dict) -> bool:
    """Met à jour `run` depuis Runway et télécharge le résultat s'il vient d'aboutir.
    Renvoie True si le run vient de passer dans un état terminal."""
    if run["status"] in TERMINAL_STATUSES:
        return False
    details = client.tasks.retrieve(run["id"])
    run["status"] = details.status
    run["progress"] = getattr(details, "progress", None)
    estimated = getattr(details, "estimated_cost", None)
    if estimated is not None:
        run["estimated_credits"] = estimated.credits
    if details.status not in TERMINAL_STATUSES:
        return False
    run["credits"] = details.cost.credits
    if details.status == "FAILED":
        run["failure"] = details.failure
    elif details.status == "SUCCEEDED" and details.output:
        saved = download_outputs(details.output, output_base_path(run), run.get("output_format", "mp4"))
        run["outputs"] = [rel(p) for p in saved]
    return True


# ---------------------------------------------------------------------------
# Keyframes retenues, prompts et lancement (partagés par les onglets 3 à 5)
# ---------------------------------------------------------------------------

def extract_origin(clip_rel: str) -> tuple[str, float] | None:
    """(nom de la vidéo source, début en s) d'un extrait nommé par cut_extract, sinon None."""
    m = EXTRACT_NAME.match(Path(clip_rel).stem)
    return (m["source"], float(m["start"])) if m else None


def load_keyframe_selection() -> dict:
    """Keyframe retenue par vidéo : {chemin relatif de la vidéo: {"path": ..., "t": ...}}."""
    if not KEYFRAME_SELECTION_FILE.is_file():
        return {}
    return json.loads(KEYFRAME_SELECTION_FILE.read_text(encoding="utf-8"))


def retained_keyframe(clip: Path) -> tuple[Path, float] | None:
    entry = load_keyframe_selection().get(rel(clip))
    if entry and (ROOT / entry["path"]).is_file():
        return ROOT / entry["path"], entry["t"]
    return None


def set_retained_keyframe(clip: Path, image: Path | None, t: float | None = None) -> None:
    """Retient `image` à l'instant `t` comme keyframe de `clip` (None : retire la keyframe)."""
    selection = load_keyframe_selection()
    if image is None:
        selection.pop(rel(clip), None)
    else:
        selection[rel(clip)] = {"path": rel(image), "t": t}
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    KEYFRAME_SELECTION_FILE.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")


def keyframe_uploader(container, clip: Path, t: float, key: str) -> None:
    """Import d'une image retouchée, retenue comme keyframe de `clip` à l'instant `t` (onglets 3 et 4)."""
    kf_file = container.file_uploader("Image retouchée", type=IMAGE_TYPES, key=f"{key}::{clip.name}")
    if not kf_file:
        return
    kf_bytes = kf_file.getvalue()
    kf_size = image_size(kf_bytes)
    if kf_size is None:
        container.error("Image illisible.")
        return
    container.image(kf_bytes, caption=f"Keyframe retouchée · {kf_size[0]}×{kf_size[1]}")
    info = video_info(clip)
    clip_ratio = info["width"] / info["height"]
    kf_ratio = kf_size[0] / kf_size[1]
    if abs(kf_ratio - clip_ratio) / clip_ratio > RATIO_TOLERANCE:
        container.warning(
            f"Le ratio de l'image ({kf_ratio:.2f}) diffère de celui de la vidéo "
            f"({clip_ratio:.2f}) : ce n'est probablement pas une image de cette vidéo. Aleph "
            "va essayer de faire ressembler le plan à cette image, contenu compris."
        )
    if container.button("Retenir comme keyframe", key=f"{key}_retain::{clip.name}"):
        KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
        kf_path = KEYFRAMES_DIR / f"{clip.stem}_t{t:.2f}_{Path(kf_file.name).name}"
        kf_path.write_bytes(kf_bytes)
        set_retained_keyframe(clip, kf_path, t)
        st.rerun()  # la keyframe retenue s'affiche partout, y compris au-dessus de ce bloc


def save_prompt(text_key: str, name: str, new_version: bool) -> None:
    """Callback : enregistre le texte de la zone `text_key`, en nouvelle version ou en écrasant `name`.
    Le modèle vidéo noté est celui de la dernière proposition de Claude mise dans la zone, sinon celui
    du fichier d'origine."""
    path = next_prompt_version(name) if new_version else PROMPT_DIR / name
    path.write_text(st.session_state[text_key].strip() + "\n", encoding="utf-8")
    model = st.session_state.pop(f"{text_key}_model", None) or prompt_model(name)
    set_prompt_model(path.name, model)
    if new_version:
        # La zone de l'ancien fichier redevient celle du disque ; on bascule sur la nouvelle version.
        del st.session_state[text_key]
        st.session_state.prompt_file = path.name
    # Les onglets 4 et 5 proposent le prompt qui vient d'être enregistré, avec son modèle.
    st.session_state.gx_prompt = st.session_state.full_prompt = path.name
    st.session_state.gx_model = st.session_state.claude_target = model
    st.session_state.flash.append(("success", f"Prompt enregistré : prompt_input/{path.name}"))


def save_suggested_prompt(text: str, name_key: str, video_model: str | None = None) -> None:
    """Callback : enregistre une proposition de Claude dans un nouveau fichier de prompt_input
    (sans jamais écraser un fichier existant), puis la sélectionne dans les onglets 3, 4 et 5, avec le
    modèle vidéo pour lequel elle a été rédigée dans l'onglet 4."""
    stem = re.sub(r"[^\w.-]+", "_", st.session_state[name_key].strip()).removesuffix(".txt").strip("._")
    if not stem:
        st.session_state.flash.append(("error", "Donnez un nom au fichier de prompt."))
        return
    path = PROMPT_DIR / f"{stem}.txt"
    if path.exists():
        st.session_state.flash.append(("error", f"prompt_input/{path.name} existe déjà : choisissez un autre nom."))
        return
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    st.session_state.prompt_file = st.session_state.gx_prompt = st.session_state.full_prompt = path.name
    model = video_model or ALEPH_MODEL
    set_prompt_model(path.name, model)
    st.session_state.gx_model = st.session_state.claude_target = model
    st.session_state[name_key] = st.session_state[f"{name_key}_auto"] = next_prompt_version(path.name).name
    st.session_state.flash.append(("success", f"Proposition de Claude enregistrée : prompt_input/{path.name}"))


def prompt_selector(key: str, default: str | None = None, model_key: str | None = None) -> tuple[str, str]:
    """Choix d'un prompt enregistré (affiché en lecture seule) ou d'un prompt court saisi sur place.
    Avec `model_key`, choisir un prompt enregistré y remet son modèle vidéo (Aleph 2 par défaut).
    Renvoie (nom : fichier, TYPED_PROMPT_NAME ou NO_PROMPT ; texte)."""
    options = list_prompts() + [TYPED_PROMPT, NO_PROMPT]
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    name = st.selectbox(
        "Prompt", options, key=key,
        help="Prompts rédigés dans l'onglet 3, ou un prompt court saisi directement ici.",
        on_change=select_prompt_model if model_key else None,
        args=(key, (model_key,)) if model_key else None,
    )
    if name == NO_PROMPT:
        st.caption("Sans prompt, Aleph applique seulement la retouche de la keyframe.")
        return name, ""
    if name == TYPED_PROMPT:
        text = st.text_area(
            "Prompt court", key=f"{key}_typed", height=100, max_chars=ALEPH_PROMPT_MAX_CHARS,
            help="Envoyé tel quel à Aleph, sans fichier dans prompt_input ; il reste noté dans le suivi (onglet 6).",
        ).strip()
        return TYPED_PROMPT_NAME, text
    text = (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    st.code(text or "(fichier vide)", language=None, wrap_lines=True)
    return name, text


def output_settings(key: str, default_format: str) -> tuple[str, str | None, str]:
    """Format, profil ProRes et ratio de sortie."""
    c1, c2, c3 = st.columns(3)
    output_format = c1.selectbox(
        "Format de sortie", OUTPUT_FORMATS, index=OUTPUT_FORMATS.index(default_format), key=f"{key}_format",
    )
    prores_profile = None
    if output_format == "prores":
        prores_profile = c2.selectbox(
            "Profil ProRes", PRORES_PROFILES, index=PRORES_PROFILES.index("422 HQ"), key=f"{key}_prores",
        )
    ratio = c3.text_input("Ratio (optionnel)", key=f"{key}_ratio", help="ex. 1280:720 — vide = valeur par défaut")
    return output_format, prores_profile, ratio.strip()


def billing_note(duration: float, n_tasks: int, output_format: str) -> None:
    billed = duration * n_tasks
    msg = f"{n_tasks} tâche(s) × {duration:.1f} s = **{billed:.1f} s de vidéo facturées**"
    credits = ALEPH_CREDITS_PER_SECOND * billed
    if output_format != "mp4":
        msg += f", dont un supplément {output_format} de **{NON_MP4_SURCHARGE * billed:.0f} crédits**"
        credits += NON_MP4_SURCHARGE * billed
    msg += f" · coût estimé : **{credits:.0f} crédits ≈ {credits * CREDIT_USD:.2f} $**"
    st.markdown(msg)
    st.caption(
        f"Estimation : {ALEPH_CREDITS_PER_SECOND} crédits/s (tarif mesuré sur les essais mp4)"
        + (f" + {NON_MP4_SURCHARGE} crédits/s pour {output_format}" if output_format != "mp4" else "")
        + f", à {CREDIT_USD} $ le crédit. Le coût réel s'affiche dans l'onglet « 6. Suivi »."
    )


def seedance_ratio(width: int, height: int) -> str:
    """Taille de sortie Seedance 2.5 au ratio le plus proche de la vidéo, la plus grande à ratio égal."""
    target = np.log(width / height)
    sizes = [tuple(map(int, r.split(":"))) for r in SEEDANCE_RATIOS]
    w, h = min(sizes, key=lambda s: (round(abs(np.log(s[0] / s[1]) - target), 3), -s[0]))
    return f"{w}:{h}"


def seedance_duration(seconds: float) -> int:
    """Durée de sortie Seedance : celle de l'extrait arrondie à la seconde, au moins SEEDANCE_MIN_SECONDS."""
    return max(SEEDANCE_MIN_SECONDS, round(seconds))


def launch_aleph(
    clip: Path, prompt_name: str, prompt_text: str, seeds: list[int], output_format: str,
    prores_profile: str | None, ratio: str, keyframe: tuple[Path, float] | None, kind: str,
    model: str = ALEPH_MODEL,
) -> None:
    """Envoie `clip` et crée une tâche par seed ; `kind` = "extract" ou "complete".
    Avec OMNI_MODEL, KLING_MODEL ou SEEDANCE_MODEL : une seule tâche (`seeds` = [None]), la keyframe envoyée comme
    image de référence ; Seedance rend une vidéo de la durée de l'extrait arrondie à la seconde."""
    client = get_client()
    runs = load_runs()
    try:
        with st.spinner(f"Envoi de {clip.name}..."):
            video_uri = upload_ephemeral_asset(clip)
        for seed in seeds:
            if model == OMNI_MODEL:
                kwargs = {"model": model, "video_uri": video_uri, "prompt_text": prompt_text}
                if keyframe:
                    kwargs["references"] = [{"uri": image_data_uri(keyframe[0])}]
            elif model == KLING_MODEL:
                kwargs = {"model": model, "video_uri": video_uri, "prompt_text": prompt_text, "ratio": "auto_1080p"}
                if keyframe:
                    kwargs["references"] = [{"uri": image_data_uri(keyframe[0])}]
            elif model == SEEDANCE_MODEL:
                info = video_info(clip)
                kwargs = {
                    "model": model, "prompt_video": video_uri, "prompt_text": prompt_text, "mode": "reference",
                    "duration": seedance_duration(info["duration"]), "audio": False,
                    "ratio": seedance_ratio(info["width"], info["height"]),
                }
                if keyframe:
                    kwargs["references"] = [{"uri": image_data_uri(keyframe[0])}]
            else:
                kwargs = {"model": model, "video_uri": video_uri, "output_format": output_format, "seed": seed}
                if prompt_text:
                    kwargs["prompt_text"] = prompt_text
                if keyframe:
                    kwargs["keyframes"] = [{"seconds": keyframe[1], "uri": image_data_uri(keyframe[0])}]
                if ratio:
                    kwargs["ratio"] = ratio
                if prores_profile:
                    kwargs["prores_profile"] = prores_profile
            task = client.video_to_video.create(**kwargs)
            save_last_task_id(task.id)
            runs.append({
                "id": task.id,
                "created": datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                "model": model,
                "clip": rel(clip),
                "clip_duration": video_info(clip)["duration"],
                "prompt_name": prompt_name,
                "prompt_text": prompt_text,
                "keyframe": rel(keyframe[0]) if keyframe else None,
                "keyframe_seconds": keyframe[1] if keyframe else None,
                "seed": seed,
                "output_format": output_format,
                "prores_profile": prores_profile,
                "ratio": ratio or None,
                "status": "PENDING",
            })
            st.session_state.flash.append((
                "success", f"Tâche {task.id} lancée ({prompt_name}, {run_model_label(runs[-1])}).",
            ))
    except (APIError, FileNotFoundError, ValueError) as exc:
        st.session_state.flash.append(("error", f"Échec du lancement : {exc}"))
    finally:
        save_runs(runs)
    st.session_state.flash.append(("info", "Suivez l'avancement dans l'onglet « 6. Suivi »."))
    st.rerun()


def apply_trial(trial_key: str, timeline: Path) -> None:
    """Callback : reprend le prompt, la seed et la keyframe de l'essai choisi pour la génération complète."""
    run = st.session_state[trial_key]
    if run is None:
        return
    prompt_name = run.get("prompt_name")
    if prompt_name == TYPED_PROMPT_NAME:
        st.session_state.full_prompt = TYPED_PROMPT
        st.session_state.full_prompt_typed = run.get("prompt_text") or ""
    else:
        st.session_state.full_prompt = prompt_name if (PROMPT_DIR / str(prompt_name)).is_file() else NO_PROMPT
    st.session_state.full_seed = int(run.get("seed") or DEFAULT_SEED)
    st.session_state.full_trial_keyframe = None
    st.session_state.pop(f"gf_kf::{timeline.name}", None)  # le choix de keyframe repart sur celle de l'essai
    origin = extract_origin(run["clip"])
    if run.get("keyframe") and origin and origin[0] == timeline.stem:
        st.session_state.full_trial_keyframe = {
            "path": run["keyframe"], "t": round(origin[1] + run["keyframe_seconds"], 3),
        }


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Corrections IA", layout="wide")
st.title("Corrections IA — Runway Aleph 2.0")

if "flash" not in st.session_state:
    st.session_state.flash = []

with st.sidebar:
    st.header("Configuration")
    if os.environ.get("RUNWAYML_API_SECRET"):
        st.success("Clé RUNWAYML_API_SECRET chargée")
    else:
        st.error("RUNWAYML_API_SECRET absente (fichier .env à la racine du dépôt)")
    runs_all = load_runs()
    spent = sum(r.get("credits") or 0 for r in runs_all)
    st.metric("Crédits consommés (historique)", spent)
    st.caption(
        "Aleph facture à la seconde de sortie. Les formats ProRes et PNG ajoutent "
        f"{NON_MP4_SURCHARGE} crédits/s : testez en mp4, gardez ProRes pour le rendu final."
    )

for kind, msg in st.session_state.flash:
    getattr(st, kind)(msg)
st.session_state.flash = []

(
    tab_export, tab_extract, tab_prepare, tab_gen_extract, tab_gen_full, tab_runs, tab_compare,
) = st.tabs([
    "1. Export timeline",
    "2. Extrait",
    "3. Générer Prompt et Keyframe",
    "4. Génération extrait",
    "5. Génération complète",
    "6. Suivi",
    "7. Comparer",
])

# --- 1. Export timeline -----------------------------------------------------
with tab_export:
    resolve_export_section()

# --- 2. Extrait -------------------------------------------------------------
with tab_extract:
    # Les exports Resolve d'abord ; les vidéos posées à la main dans video_input restent disponibles.
    sources = list_videos(TIMELINE_DIR) + list_videos(VIDEO_INPUT_DIR)
    if not sources:
        st.info("Aucune vidéo : exportez d'abord une timeline dans l'onglet « 1. Export timeline ».")
    else:
        source = st.selectbox(
            "Vidéo source", sources, key="extract_source",
            format_func=lambda p: f"{p.name}  ({'timeline_complete' if p.parent == TIMELINE_DIR else 'video_input'})",
        )
        info = video_info(source)
        out_w, out_h = aleph_export.aleph_size(info["width"], info["height"])
        st.caption(
            f"{info['width']}×{info['height']} · {info['fps']:g} i/s · {info['duration']:.2f} s"
            + (f" · extrait réduit en {out_w}×{out_h}" if (out_w, out_h) != (info["width"], info["height"]) else "")
        )
        if info["fps"] > aleph_export.ALEPH_MAX_FPS + 1e-3:
            st.warning(f"Source à {info['fps']:g} i/s : Aleph accepte 30 i/s maximum.")
        min_dur = aleph_export.ALEPH_MIN_SECONDS
        c1, c2 = st.columns(2)
        frame_step = 1.0 / info["fps"]
        start = c1.number_input(
            "Début (s)", min_value=0.0, max_value=max(info["duration"] - min_dur, 0.0),
            value=0.0, step=frame_step, format="%.2f",
        )
        # Arrondi vers le bas : un arrondi au plus proche peut dépasser la fin de la vidéo (4,16 s -> 4,2 s).
        max_dur = np.floor(min(aleph_export.ALEPH_MAX_SECONDS, info["duration"] - start) * 10) / 10
        too_short = max_dur < min_dur
        if too_short:
            st.error("Moins de 2 s disponibles : Aleph exige 2 à 30 s.")
        duration = c2.slider(
            "Durée (s)", min_value=min(min_dur, max_dur), max_value=max_dur,
            value=min(5.0, max_dur), step=0.5,
        )
        st.caption("Choisissez le passage le plus exigeant : visage proche, mouvement, contact avec le sol.")

        cols = st.columns(3)
        for col, (label, t) in zip(cols, [
            ("Début", start), ("Milieu", start + duration / 2), ("Fin", start + duration - frame_step),
        ]):
            col.image(frame_of(source, t, 640), caption=f"{label} · {t:.2f} s")

        with st.expander("Lire la vidéo source"):
            st.video(str(source), start_time=int(start))

        if st.button("Créer l'extrait", type="primary", disabled=too_short):
            with st.spinner("Découpe en cours..."):
                try:
                    out = cut_extract(source, start, duration)
                    st.success(
                        f"Extrait créé : {rel(out)} — préparez son prompt et sa keyframe dans l'onglet "
                        "« 3. Générer Prompt et Keyframe »."
                    )
                except RuntimeError as exc:
                    st.error(f"ffmpeg a échoué :\n\n{exc}")

    extracts = list_videos(EXTRACTS_DIR)
    if extracts:
        st.subheader("Extraits existants")
        for p in extracts:
            i = video_info(p)
            st.write(f"- `{rel(p)}` — {i['duration']:.2f} s, {i['width']}×{i['height']}")
        # Un seul lecteur : chaque st.video charge sa vidéo en mémoire, même dans un expander replié.
        newest = max(extracts, key=lambda p: p.stat().st_mtime)
        played = st.selectbox(
            "Extrait à lire", extracts, index=extracts.index(newest), format_func=lambda p: p.name,
            key="extract_play",
        )
        origin = extract_origin(rel(played))
        if origin:
            st.caption(f"Tiré de {origin[0]}, à partir de {origin[1]:.2f} s.")
        st.video(str(played), loop=True)


def clip_label(p: Path) -> str:
    kind = {EXTRACTS_DIR: "extrait", TIMELINE_DIR: "timeline complète"}.get(p.parent, "video_input")
    return f"{p.name}  ({kind}, {video_info(p)['duration']:.1f} s)"


# --- 3. Générer Prompt et Keyframe -----------------------------------------
with tab_prepare:
    work_clips = list_videos(EXTRACTS_DIR) + list_videos(TIMELINE_DIR) + list_videos(VIDEO_INPUT_DIR)
    if not work_clips:
        st.info("Créez d'abord un extrait dans l'onglet « 2. Extrait ».")
    else:
        clip = st.selectbox(
            "Vidéo de travail", work_clips, format_func=clip_label, key="work_clip",
            help="Son image sert de référence à Claude et de base à la keyframe.",
        )
        clip_info = video_info(clip)
        clip_duration = clip_info["duration"]

        # Une seule image de référence pour tout l'onglet : l'assistant Claude et la génération de keyframe.
        look_ref = st.file_uploader(
            "Image de référence de look (optionnel, ex. une image d'un autre film)",
            type=IMAGE_TYPES, key="look_ref",
            help="Utilisée par l'assistant Claude, qui décrit sa lumière avec des mots dans le prompt Aleph, "
            "et par la génération de keyframe (@look). Elle n'est jamais envoyée à Aleph.",
        )
        reference = look_reference(look_ref, st) if look_ref else None

        st.subheader("Prompt Aleph")
        prompts = list_prompts()
        if not prompts:
            (PROMPT_DIR / "prompt.txt").write_text("", encoding="utf-8")
            prompts = list_prompts()
        if st.session_state.get("prompt_file") not in prompts:
            st.session_state.prompt_file = prompts[0]
        prompt_name = st.selectbox(
            "Fichier de prompt", prompts, key="prompt_file",
            on_change=select_prompt_model, args=("prompt_file", ("claude_target", "gx_model")),
        )
        if "claude_target" not in st.session_state:  # premier affichage : le modèle du prompt ouvert
            st.session_state.claude_target = prompt_model(prompt_name)
        saved_text = (PROMPT_DIR / prompt_name).read_text(encoding="utf-8").strip()
        text_key = prompt_widget_key(prompt_name)
        if text_key not in st.session_state:
            st.session_state[text_key] = saved_text

        # Instant de la keyframe (réglé plus bas, section Keyframe) : son image sert à l'assistant (@shot).
        kf_t_key, kf_mode_key = f"kf_t::{clip.name}", f"kf_mode::{clip.name}"
        kf_instant = st.session_state.get(kf_t_key, round(clip_duration / 2, 2))

        with st.expander("✨ Assistant Claude : rédiger le prompt vidéo et le prompt de retouche", expanded=False):
            st.caption(
                "Décrivez en français ce que vous voulez obtenir. En un seul appel, Claude rédige le prompt "
                "du modèle vidéo choisi (selon son guide officiel : Runway pour Aleph, Google pour Gemini "
                "Omni Flash) et, si une image de référence est importée, le prompt de retouche de la keyframe. "
                "L'appel passe par la CLI Claude Code, donc par votre abonnement."
            )
            target = st.radio(
                "Modèle vidéo visé", list(VIDEO_MODEL_NAMES), format_func=VIDEO_MODEL_NAMES.get,
                key="claude_target", horizontal=True,
                help="Aleph : prompt court et ciblé, sans négation ni liste de ce qu'il faut garder. Gemini Omni "
                "Flash : un seul changement, « Keep everything else the same. », négations explicites "
                "autorisées. Seedance 2.5 : cible (« Edit Video 1 »), changement, ce qu'il faut protéger, "
                "restrictions ; la keyframe est « Image 1 ». Kling 4.0 : « Edit @Video1 », rôle de @Image1, ce "
                "qu'il faut garder puis « Keep everything else unchanged. », peu de négations. Le modèle "
                "choisi est aussi présélectionné dans l'onglet 4.",
            )
            intent = st.text_area(
                "Ce que vous voulez obtenir", key="claude_intent", height=110,
                placeholder="ex. intégrer l'acteur dans la grotte : contre-jour froid à gauche, ombre de contact "
                "au sol, même grain que le décor",
            )
            a1, a2, a3 = st.columns([2, 2, 1])
            send_frame = a1.checkbox(
                f"Montrer à Claude l'image de l'instant de la keyframe ({kf_instant:.2f} s)", value=True,
            )
            from_current = a2.checkbox("Partir du prompt actuel", value=bool(st.session_state[text_key].strip()))
            claude_model = a3.selectbox("Modèle", CLAUDE_MODELS, key="claude_model")
            use_reference = st.checkbox(
                "Donner l'image de référence à Claude (et rédiger le prompt de retouche)" if reference
                else "Donner l'image de référence à Claude (aucune importée en haut de l'onglet)",
                value=True, disabled=not reference, key="claude_use_ref",
            )
            if st.button("Proposer les prompts", disabled=not intent.strip()):
                frame = frame_of(clip, kf_instant) if send_frame else None
                with st.spinner("Claude rédige les prompts..."):
                    try:
                        st.session_state.claude_result = {
                            **ask_claude_for_aleph_prompt(
                                intent, st.session_state[text_key] if from_current else "", frame, claude_model,
                                reference if use_reference else None, target,
                            ),
                            "video_model": target,
                        }
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        st.session_state.claude_result = None
                        st.error(f"Échec de l'appel à Claude : {exc}")

            result = st.session_state.get("claude_result")
            if result:
                result_model = result.get("video_model", ALEPH_MODEL)
                st.markdown(f"**Analyse.** {result['analysis']}")
                if result_model != ALEPH_MODEL:
                    st.markdown(f"##### Prompt {VIDEO_MODEL_NAMES[result_model]}")
                    st.caption(
                        "Proposition principale : sans image de référence. Première variante : avec la keyframe "
                        "retenue envoyée comme image de référence (case cochée dans l'onglet 4)."
                    )
                else:
                    st.markdown("##### Prompt Aleph (génération sans keyframe)")
                name_key = "claude_save_name"
                prompt_name_field(name_key, next_prompt_version(prompt_name).name, "Nom du fichier pour « Enregistrer »")
                proposals = [("Proposition principale", result["prompt"])] + [
                    (f"Variante : {v['angle']}", v["prompt"]) for v in result["variants"]
                ]
                for i, (label, text) in enumerate(proposals):
                    st.markdown(f"**{label}**")
                    p1, p2, p3 = st.columns([5, 1, 1])
                    p1.code(text, language=None, wrap_lines=True)
                    p2.button(
                        "Utiliser", key=f"use_claude_{i}",
                        on_click=use_suggested_prompt, args=(text_key, text, result_model),
                    )
                    p3.button(
                        "Enregistrer", key=f"save_claude_{i}", type="primary",
                        on_click=save_suggested_prompt, args=(text, name_key, result_model),
                    )
                st.caption(
                    "« Utiliser » met la proposition dans la zone ci-dessous pour la retoucher avant de l'enregistrer ; "
                    "« Enregistrer » la sauvegarde telle quelle sous le nom indiqué."
                )
                if result.get("keyframe_prompt"):
                    st.markdown("##### Prompt de retouche (modèle d'image, pour créer la keyframe)")
                    q1, q2 = st.columns([5, 2])
                    q1.code(result["keyframe_prompt"], language=None, wrap_lines=True)
                    q2.button(
                        "Utiliser pour la keyframe", key="use_claude_kf", type="primary",
                        on_click=use_for_keyframe, args=(result["keyframe_prompt"], kf_mode_key),
                    )
                    st.caption(
                        "Ce prompt n'est pas envoyé au modèle vidéo : il sert au modèle d'image de la section "
                        "Keyframe. Une fois la keyframe retenue, « Prompt court pour cette keyframe » rédige le "
                        "prompt vidéo qui l'accompagne."
                    )
                st.markdown(f"**Conseils.** {result['advice']}")

        prompt_text = st.text_area("Prompt (modifiable)", key=text_key, height=220).strip()
        modified = prompt_text != saved_text
        s1, s2, s3 = st.columns([2, 2, 3])
        s1.button(
            f"Enregistrer sous {next_prompt_version(prompt_name).name}", type="primary", disabled=not modified,
            on_click=save_prompt, args=(text_key, prompt_name, True),
        )
        s2.button(
            f"Écraser {prompt_name}", disabled=not modified,
            on_click=save_prompt, args=(text_key, prompt_name, False),
        )
        if modified:
            s3.warning("Modifications non enregistrées : les onglets 4 et 5 utilisent les prompts enregistrés.")

        st.subheader("Keyframe")
        st.caption(
            "C'est le mode de travail principal d'Aleph 2.0 : il applique à tout le plan la retouche faite "
            "sur une image de la vidéo. La keyframe doit donc être une image de ce plan, retouchée."
        )
        kf_t = st.number_input(
            "Instant de la keyframe dans la vidéo (s)", min_value=0.0,
            max_value=max(clip_duration - 1.0 / clip_info["fps"], 0.0), value=round(clip_duration / 2, 2),
            step=1.0 / clip_info["fps"], format="%.2f", key=kf_t_key,
        )
        st.image(frame_of(clip, kf_t, 960), caption=f"Image d'origine à {kf_t:.2f} s")
        gen_key = f"kf_gen::{clip.name}"
        kf_mode = st.radio(
            "Source de la keyframe", [KF_MODE_GENERATE, KF_MODE_IMPORT], key=kf_mode_key, horizontal=True,
        )

        if kf_mode == KF_MODE_GENERATE:
            st.caption(
                "L'image ci-dessus (@shot) est reprise par un modèle d'image Runway avec la lumière de "
                "l'image de référence (@look). Vous choisissez ensuite la meilleure proposition. "
                "Chaque image générée consomme des crédits Runway."
            )
            g1, g2, g3 = st.columns([3, 2, 1])
            if reference:
                g1.caption(f"@look : l'image de référence importée en haut de l'onglet ({look_ref.name}).")
            else:
                g1.warning("Importez l'image de référence de look en haut de l'onglet (@look).")
            kf_model = g2.selectbox(
                "Modèle d'image", list(KEYFRAME_MODELS), key="kf_model",
                format_func=lambda m: f"{KEYFRAME_MODEL_NAMES[m]}  ({m})",
            )
            kf_count = g3.selectbox(
                "Propositions", KEYFRAME_COUNTS.get(kf_model, [1, 2, 3, 4]), index=1,
                key=f"kf_count::{kf_model}",
            )
            final_w, final_h = keyframe_output_size(clip_info["width"], clip_info["height"])
            out_w, out_h = closest_ratio(kf_model, final_w, final_h)
            g2.caption(f"Génération en {out_w}×{out_h}, enregistrée en {final_w}×{final_h} (2K).")

            if "kf_gen_prompt" not in st.session_state:
                st.session_state.kf_gen_prompt = KEYFRAME_DEFAULT_PROMPT
            kf_prompt = st.text_area(
                "Prompt de retouche (citez @shot et @look)", key="kf_gen_prompt", height=130, max_chars=1000,
                help="Rédigé par l'assistant Claude ci-dessus (« Utiliser pour la keyframe »), ou à la main.",
            ).strip()

            if st.button(
                "Générer les keyframes",
                disabled=not (reference and kf_prompt and os.environ.get("RUNWAYML_API_SECRET")),
            ):
                with st.spinner(f"Génération de {kf_count} image(s) avec {KEYFRAME_MODEL_NAMES[kf_model]}..."):
                    try:
                        paths, spent_kf = generate_keyframes(
                            get_client(), clip, kf_t, reference, kf_prompt, kf_model, kf_count,
                        )
                        st.session_state[gen_key] = {
                            "paths": [str(p) for p in paths], "t": kf_t, "credits": spent_kf, "model": kf_model,
                        }
                    except TaskFailedError as exc:
                        st.error(f"Génération refusée ou échouée : {exc.task_details.failure}")
                    except TaskTimeoutError:
                        st.error(f"Pas de résultat après {KEYFRAME_GEN_TIMEOUT} s.")
                    except (APIError, requests.RequestException) as exc:
                        st.error(f"Échec de la génération : {exc}")

            gen = st.session_state.get(gen_key)
            if gen:
                st.caption(
                    f"{len(gen['paths'])} proposition(s) pour {gen['t']:.2f} s, "
                    f"{KEYFRAME_MODEL_NAMES.get(gen['model'], gen['model'])}, "
                    f"{gen['credits']} crédits."
                )
                cols = st.columns(2)
                for i, p in enumerate(gen["paths"]):
                    col = cols[i % 2]
                    col.image(frame_of(Path(p), 0.0, 960), caption=Path(p).name)
                    col.button(
                        "Retenir comme keyframe", key=f"kf_choose::{p}",
                        on_click=set_retained_keyframe, args=(clip, Path(p), gen["t"]),
                    )

        else:
            k1, k2 = st.columns(2)
            k1.caption(
                "Exportez l'image d'origine, retouchez-la dans Resolve (étalonnage, ombre de contact…), "
                "puis réimportez-la. Une image d'un autre film ne convient pas comme keyframe : utilisez "
                "plutôt la génération à partir d'une image de référence."
            )
            if k1.button("Exporter cette image en pleine résolution"):
                try:
                    k1.success(f"Exportée : `{rel(export_full_frame(clip, kf_t))}`")
                except RuntimeError as exc:
                    k1.error(f"ffmpeg a échoué :\n\n{exc}")
            keyframe_uploader(k2, clip, kf_t, "kf_file")

        retained = retained_keyframe(clip)
        if retained:
            r1, r2 = st.columns([4, 1])
            r1.success(f"Keyframe retenue pour {clip.name} : `{rel(retained[0])}` à {retained[1]:.2f} s")
            r2.button("Retirer", key=f"kf_drop::{clip.name}", on_click=set_retained_keyframe, args=(clip, None))
            o1, o2 = st.columns(2)
            o1.image(frame_of(clip, retained[1], 640), caption=f"Image d'origine à {retained[1]:.2f} s")
            o2.image(frame_of(retained[0], 0.0, 640), caption="Keyframe retenue")

            # Prompt court rédigé d'après la keyframe réellement obtenue (un seul prompt_text par tâche), pour
            # le modèle visé dans l'assistant.
            short_key = f"kf_short::{clip.name}"
            short_target = st.session_state.get("claude_target", ALEPH_MODEL)
            if st.button(
                f"Prompt court pour cette keyframe ({VIDEO_MODEL_NAMES[short_target]})",
                help="Claude compare l'image d'origine et la keyframe retenue, et décrit seulement leur "
                "différence visible, pour que le modèle vidéo ne reçoive pas une consigne différente de "
                "l'image. Le modèle visé se choisit dans l'assistant Claude ci-dessus.",
            ):
                with st.spinner("Claude compare l'image d'origine et la keyframe..."):
                    try:
                        short = ask_claude_for_keyframe_short_prompt(
                            frame_of(clip, retained[1]), frame_of(retained[0], 0.0),
                            st.session_state.get("claude_intent", ""),
                            st.session_state.get("claude_model", CLAUDE_MODELS[0]), short_target,
                        )
                        st.session_state[short_key] = {
                            **short, "keyframe": rel(retained[0]), "video_model": short_target,
                        }
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        st.error(f"Échec de l'appel à Claude : {exc}")
            short = st.session_state.get(short_key)
            if short and short["keyframe"] == rel(retained[0]):
                short_model = short.get("video_model", ALEPH_MODEL)
                st.markdown(
                    f"**Différence vue par Claude** (prompt pour {VIDEO_MODEL_NAMES[short_model]}). "
                    f"{short['analysis']}"
                )
                short_name_key = f"kf_short_name::{clip.name}"
                base = re.sub(r"_v\d+$", "", Path(prompt_name).stem)
                suggestion = f"{base}_keyframe.txt"
                if (PROMPT_DIR / suggestion).exists():
                    suggestion = next_prompt_version(suggestion).name
                prompt_name_field(short_name_key, suggestion, "Nom du fichier du prompt court")
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.code(short["prompt"], language=None, wrap_lines=True)
                c2.button(
                    "Utiliser", key=f"use_short::{clip.name}",
                    on_click=use_suggested_prompt, args=(text_key, short["prompt"], short_model),
                )
                c3.button(
                    "Enregistrer", key=f"save_short::{clip.name}", type="primary",
                    on_click=save_suggested_prompt, args=(short["prompt"], short_name_key, short_model),
                )
                st.caption(
                    "À envoyer avec cette keyframe dans l'onglet 4. Comparez aussi avec la keyframe seule, "
                    "sans prompt : le guide Runway indique qu'elle suffit le plus souvent."
                    if short_model == ALEPH_MODEL else
                    f"À envoyer dans l'onglet 4 avec {VIDEO_MODEL_NAMES[short_model]}, case « Envoyer la keyframe "
                    "retenue » cochée : la keyframe part comme image de référence."
                )
        else:
            st.caption("Aucune keyframe retenue pour cette vidéo.")

# --- 4. Génération extrait --------------------------------------------------
with tab_gen_extract:
    extract_clips = list_videos(EXTRACTS_DIR)
    if not extract_clips:
        st.info("Créez d'abord un extrait dans l'onglet « 2. Extrait ».")
    else:
        st.caption("Essais courts pour choisir le prompt, la keyframe et la seed avant la génération complète.")
        clip = st.selectbox("Extrait", extract_clips, format_func=clip_label, key="gx_clip")
        clip_duration = video_info(clip)["duration"]
        if "gx_model" not in st.session_state:  # premier affichage : le modèle du prompt proposé
            st.session_state.gx_model = prompt_model(
                st.session_state.get("gx_prompt") or st.session_state.get("prompt_file") or "",
            )
        gx_model = st.radio(
            "Modèle vidéo", list(VIDEO_MODEL_NAMES), format_func=VIDEO_MODEL_NAMES.get, key="gx_model",
            horizontal=True,
        )
        omni = gx_model != ALEPH_MODEL  # Gemini Omni Flash ou Seedance : prompt obligatoire, ni seed ni format
        if gx_model == OMNI_MODEL:
            st.caption(
                "Gemini Omni Flash retouche la vidéo d'après une consigne (prompt obligatoire). La keyframe "
                "retenue lui est envoyée comme image de référence, sans instant ; pas de seed ni de format."
            )
        elif gx_model == KLING_MODEL:
            st.caption(
                "Kling 4.0 retouche la vidéo d'après une consigne (prompt obligatoire), en gardant son cadrage ; "
                "sortie en 1080p. Dans le prompt, l'extrait est « @Video1 » et la keyframe retenue, envoyée "
                "comme image de référence, est « @Image1 ». Pas de seed ni de format."
            )
        elif gx_model == SEEDANCE_MODEL:
            clip_meta = video_info(clip)
            st.caption(
                "Seedance 2.5 génère une nouvelle vidéo guidée par l'extrait (mode reference) : il peut "
                "réinventer l'acteur plus qu'Aleph. Dans le prompt, l'extrait est « Video 1 » et la keyframe "
                "retenue, envoyée comme image de référence, est « Image 1 ». Sortie en "
                f"{seedance_ratio(clip_meta['width'], clip_meta['height'])} sans audio, durée "
                f"{seedance_duration(clip_duration)} s ; pas de seed."
            )
            if clip_duration < SEEDANCE_MIN_SECONDS - 0.5:
                st.warning(
                    f"Extrait de {clip_duration:.1f} s : Seedance rend au moins {SEEDANCE_MIN_SECONDS} s, il "
                    "inventera donc la fin. Pour comparer avec Aleph sur le même plan, coupez un extrait "
                    f"d'au moins {SEEDANCE_MIN_SECONDS} s dans l'onglet « 2. Extrait »."
                )
        prompt_name, prompt_text = prompt_selector(
            "gx_prompt", default=st.session_state.get("prompt_file"), model_key="gx_model",
        )

        keyframe = None
        retained = retained_keyframe(clip)
        if retained:
            k1, k2 = st.columns([1, 3])
            k1.image(frame_of(retained[0], 0.0, 320))
            if k2.checkbox(
                f"Envoyer la keyframe retenue ({retained[0].name}, {retained[1]:.2f} s)", value=True,
                key=f"gx_kf::{clip.name}",
            ):
                keyframe = retained
        else:
            st.caption("Pas de keyframe retenue pour cet extrait (onglet 3).")
        with st.expander("Importer une keyframe" + (" (remplace celle retenue)" if retained else "")):
            info = video_info(clip)
            gx_t = st.number_input(
                "Instant de la keyframe dans l'extrait (s)", min_value=0.0,
                max_value=max(clip_duration - 1.0 / info["fps"], 0.0), value=round(clip_duration / 2, 2),
                step=1.0 / info["fps"], format="%.2f", key=f"gx_kf_t::{clip.name}",
            )
            i1, i2 = st.columns(2)
            i1.image(frame_of(clip, gx_t, 640), caption=f"Image d'origine à {gx_t:.2f} s")
            keyframe_uploader(i2, clip, gx_t, "gx_kf_file")

        if omni:
            seeds, output_format, prores_profile, ratio = [None], "mp4", None, ""
            st.caption("Une tâche par lancement. Tarif non mesuré : le coût réel s'affiche dans l'onglet « 6. Suivi ».")
            if not prompt_text:
                st.warning(f"{VIDEO_MODEL_NAMES[gx_model]} demande un prompt.")
        else:
            seeds_raw = st.text_input(
                "Seed(s)", value=str(DEFAULT_SEED), key="gx_seeds",
                help="Plusieurs seeds séparées par des virgules = une tâche par seed (ex. 42, 7, 1234).",
            )
            try:
                seeds = [int(s) for s in re.split(r"[,\s;]+", seeds_raw.strip()) if s]
            except ValueError:
                seeds = []
                st.error("Seeds invalides : entrez des entiers séparés par des virgules.")
            output_format, prores_profile, ratio = output_settings("gx", "mp4")
            billing_note(clip_duration, len(seeds), output_format)

        if st.button(
            "Lancer les essais chez Runway", type="primary",
            disabled=not (
                seeds and (prompt_text if omni else (prompt_text or keyframe))
                and os.environ.get("RUNWAYML_API_SECRET")
            ),
        ):
            launch_aleph(
                clip, prompt_name, prompt_text, seeds, output_format, prores_profile, ratio, keyframe, "extract",
                model=gx_model,
            )

# --- 5. Génération complète -------------------------------------------------
with tab_gen_full:
    full_clips = list_videos(TIMELINE_DIR) + list_videos(VIDEO_INPUT_DIR)
    if not full_clips:
        st.info("Exportez d'abord une timeline dans l'onglet « 1. Export timeline ».")
    else:
        timeline = st.selectbox("Timeline", full_clips, format_func=clip_label, key="gf_clip")
        timeline_duration = video_info(timeline)["duration"]
        if timeline_duration > aleph_export.ALEPH_MAX_SECONDS + 1e-6:
            st.error(f"{timeline_duration:.1f} s : Aleph accepte 30 s maximum par génération.")

        trials = [
            r for r in load_runs()
            if r.get("kind", "extract") == "extract" and r["status"] == "SUCCEEDED" and r.get("clip")
        ]
        same_source = [r for r in trials if (extract_origin(r["clip"]) or ("",))[0] == timeline.stem]
        trial_key = f"gf_trial::{timeline.name}"
        st.selectbox(
            "Partir d'un essai réussi (onglet 4)", [None] + list(reversed(same_source or trials)),
            format_func=lambda r: "—" if r is None else (
                f"{Path(r['clip']).name} · {r['prompt_name']} · {run_model_label(r)}"
                + (" · keyframe" if r.get("keyframe") else "") + f" · {r['id'][:8]}"
            ),
            key=trial_key, on_change=apply_trial, args=(trial_key, timeline),
            help="Reprend son prompt, sa seed et sa keyframe, replacée à l'instant correspondant de la timeline.",
        )

        prompt_name, prompt_text = prompt_selector("full_prompt")
        if "full_seed" not in st.session_state:
            st.session_state.full_seed = DEFAULT_SEED
        seed = st.number_input("Seed", min_value=0, step=1, key="full_seed")

        keyframe = None
        options = {"Aucune": None}
        own = retained_keyframe(timeline)
        if own:
            options[f"Celle retenue pour cette timeline (onglet 3), à {own[1]:.2f} s"] = own
        trial_kf = st.session_state.get("full_trial_keyframe")
        if trial_kf and (ROOT / trial_kf["path"]).is_file():
            options[f"Celle de l'essai, à {trial_kf['t']:.2f} s dans la timeline"] = (
                ROOT / trial_kf["path"], trial_kf["t"],
            )
        kf_key = f"gf_kf::{timeline.name}"
        if st.session_state.get(kf_key) not in options:  # nouvel essai choisi : sa keyframe par défaut
            st.session_state[kf_key] = list(options)[-1]
        kf_choice = st.radio("Keyframe", list(options), key=kf_key, horizontal=True)
        keyframe = options[kf_choice]
        if keyframe:
            k1, k2 = st.columns(2)
            k1.image(frame_of(timeline, keyframe[1], 640), caption=f"Timeline à {keyframe[1]:.2f} s")
            k2.image(frame_of(keyframe[0], 0.0, 640), caption=f"Keyframe {keyframe[0].name}")
            if keyframe[1] > timeline_duration:
                st.error("Cet instant dépasse la fin de la timeline.")
        if trials and not same_source:
            st.caption(
                f"Aucun essai fait sur un extrait de {timeline.name} : leur keyframe ne peut pas être replacée."
            )

        output_format, prores_profile, ratio = output_settings("gf", "prores")
        billing_note(timeline_duration, 1, output_format)
        st.caption(f"Résultat enregistré dans {rel(OUTPUT_DIR)}/.")

        if st.button(
            "Lancer la génération complète", type="primary",
            disabled=not (
                (prompt_text or keyframe) and os.environ.get("RUNWAYML_API_SECRET")
                and timeline_duration <= aleph_export.ALEPH_MAX_SECONDS + 1e-6
                and not (keyframe and keyframe[1] > timeline_duration)
            ),
        ):
            launch_aleph(
                timeline, prompt_name, prompt_text, [int(seed)], output_format, prores_profile, ratio,
                keyframe, "complete",
            )

# --- 6. Suivi ---------------------------------------------------------------
with tab_runs:
    c1, c2 = st.columns([1, 3])
    auto = c1.toggle("Rafraîchissement auto (15 s)", value=True)

    def render_runs():
        runs = load_runs()
        active = [r for r in runs if r["status"] not in TERMINAL_STATUSES]
        finished_now = False
        if active and os.environ.get("RUNWAYML_API_SECRET"):
            client = get_client()
            for r in active:
                try:
                    finished_now |= refresh_run(client, r)
                except APIError as exc:
                    st.warning(f"{r['id']} : {exc}")
            save_runs(runs)
        if not runs:
            st.info("Aucune tâche pour l'instant.")
            return
        st.caption(f"{len(active)} tâche(s) en cours · mis à jour à {datetime.now():%H:%M:%S}")
        for r in reversed(runs):
            icon = STATUS_ICONS.get(r["status"], "•")
            progress = f" {r['progress']:.0%}" if r.get("progress") is not None and r["status"] == "RUNNING" else ""
            credits = r.get("credits")
            cost = f"{credits} crédits" if credits is not None else (
                f"≈ {r['estimated_credits']:.0f} crédits" if r.get("estimated_credits") is not None else ""
            )
            title = (
                f"{icon} {r['status']}{progress} · {r.get('prompt_name') or '?'} · {run_model_label(r)} · "
                f"{Path(r['clip']).name if r.get('clip') else 'tâche importée'} · {r.get('output_format')} · {cost}"
            )
            with st.expander(title):
                st.write(f"Tâche `{r['id']}` créée le {r.get('created', '?')}")
                if r.get("failure"):
                    st.error(r["failure"])
                if r.get("keyframe"):
                    st.write(f"Keyframe : `{r['keyframe']}` à {r['keyframe_seconds']:.2f} s")
                for o in r.get("outputs", []):
                    st.write(f"Résultat : `{o}`")
                if r.get("prompt_text"):
                    st.code(r["prompt_text"], language=None, wrap_lines=True)
                if r["status"] not in TERMINAL_STATUSES and st.button("Annuler la tâche", key=f"cancel_{r['id']}"):
                    get_client().tasks.delete(r["id"])
                    r["status"] = "CANCELLED"
                    save_runs(runs)
                    st.rerun()
        if finished_now:
            st.rerun()  # rafraîchit aussi l'onglet « 7. Comparer »

    st.fragment(run_every=15 if auto else None)(render_runs)()

    with st.expander("Importer une tâche existante (ex. lancée avec le script en ligne de commande)"):
        task_id = st.text_input("ID de tâche", placeholder="laisser vide = video_output/.last_task_id")
        if st.button("Importer"):
            last = ROOT / LAST_TASK_ID_FILE
            task_id = task_id.strip() or (last.read_text(encoding="utf-8").strip() if last.is_file() else "")
            runs = load_runs()
            if not task_id:
                st.error("Aucun ID fourni et pas de .last_task_id.")
            elif any(r["id"] == task_id for r in runs):
                st.warning("Cette tâche est déjà dans l'historique.")
            else:
                runs.append({
                    "id": task_id, "created": datetime.now().isoformat(timespec="seconds"),
                    "clip": None, "prompt_name": None, "prompt_text": None, "seed": None,
                    "output_format": "mp4", "status": "PENDING",
                })
                save_runs(runs)
                st.rerun()

# --- 7. Comparer ------------------------------------------------------------
with tab_compare:
    done = [
        r for r in load_runs()
        if r["status"] == "SUCCEEDED" and r.get("clip") and r.get("outputs")
        and (ROOT / r["outputs"][0]).suffix.lower() in VIDEO_CONTENT_TYPES
        and (ROOT / r["outputs"][0]).is_file()
    ]
    if not done:
        st.info("Aucun résultat vidéo terminé à comparer.")
    else:
        clips_done = sorted({r["clip"] for r in done})
        clip_rel = st.selectbox("Extrait", clips_done, key="cmp_clip")
        candidates = [r for r in done if r["clip"] == clip_rel]

        def run_label(r: dict) -> str:
            kf = " · keyframe" if r.get("keyframe") else ""
            return f"{r['prompt_name']}{kf} · {run_model_label(r)} · {r['id'][:8]}"

        chosen = st.multiselect(
            "Résultats à comparer", candidates, default=candidates[-3:], format_func=run_label,
        )
        source_path = ROOT / clip_rel
        items = [("Source", source_path)] + [(run_label(r), ROOT / r["outputs"][0]) for r in chosen]

        dur = video_info(source_path)["duration"]
        t = st.slider("Instant (s)", 0.0, max(dur - 0.05, 0.0), value=min(dur / 2, max(dur - 0.05, 0.0)), step=0.04)
        per_row = 2 if len(items) <= 4 else 3
        for i in range(0, len(items), per_row):
            cols = st.columns(per_row)
            for col, (label, path) in zip(cols, items[i:i + per_row]):
                if not path.is_file():
                    col.warning(f"{label} : fichier introuvable")
                    continue
                try:
                    col.image(frame_of(path, t), caption=label)
                except RuntimeError:
                    col.warning(f"{label} : pas d'image à {t:.2f} s")

        for r in chosen:
            with st.expander(f"Prompt — {run_label(r)}"):
                st.code(r["prompt_text"], language=None, wrap_lines=True)

        st.subheader("Lecture synchronisée")
        if len(items) > 1 and st.button("Générer la vidéo côte à côte"):
            with st.spinner("Assemblage avec ffmpeg..."):
                try:
                    st.session_state.cmp_video = (
                        build_comparison([p for _, p in items]), [label for label, _ in items]
                    )
                except RuntimeError as exc:
                    st.error(f"ffmpeg a échoué :\n\n{exc}")
        cmp_video, cmp_labels = st.session_state.get("cmp_video") or (None, [])
        if cmp_video and Path(cmp_video).is_file():
            st.caption("De gauche à droite : " + " | ".join(cmp_labels))
            st.video(str(cmp_video), loop=True)
            st.caption(f"Fichier : {rel(cmp_video)}")
