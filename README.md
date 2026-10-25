# python-lens-calibration

## Installation

### 1) Prérequis

- Python 3
- pip
- droit d'exécution de scripts (Windows)

```bash
python3 --version
pip --version
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2) Environnement virtuel

```bash
python3 -m venv .venv
```

- macOS / Linux

```bash
source .venv/bin/activate
```

- Windows

```powershell
.venv\Scripts\activate
```

### 3) Dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Export Unreal Engine → DaVinci Resolve (production virtuelle)

Chaîne utilisée pour la prise 41 (`Scene_1_41`) : décor Unreal rendu en EXR linéaire,
incrusté dans DaVinci Resolve avec la plaque Nikon ZR.

| Script | Où il s'exécute | Rôle |
|---|---|---|
| `src/unreal/mrq_take41.py` | Unreal Editor (Output Log, mode Python) | rendus de test et rendu complet de la séquence |
| `src/resolve/resolve_timecode_colorspace.py` | DaVinci Resolve (menu Workspace > Scripts) | Start TC + espace couleur d'entrée des séquences EXR |
| `src/exr_timecode.py` | Python (ligne de commande) | variante : écrit le timecode directement dans les fichiers EXR |
| `src/light_match.py` | Python (ligne de commande) | matching d'éclairage acteur keyé / décor / image de référence |
| `src/resolve/light_match_resolve.py` | DaVinci Resolve (console, MCP ou scripting externe) | export des calques et écriture de la correction pour `light_match.py` |

#### 1) Rendu Unreal — `src/unreal/mrq_take41.py`

Les chemins (séquence, niveau, dossier d'export, graphe) sont des constantes en tête de fichier.

```
py "<chemin>/src/unreal/mrq_take41.py" test   # 5 frames 128898-128902 -> take_41/_test_128900
py "<chemin>/src/unreal/mrq_take41.py" diff   # frame 128900 sans ombre/occlusion du perso -> _test_sans_perso
py "<chemin>/src/unreal/mrq_take41.py" tout   # test puis diff enchaînés
py "<chemin>/src/unreal/mrq_take41.py" full   # rendu complet -> take_41
```

- `test` / `diff` : config Movie Render Queue créée par le script (EXR demi-flottant ZIP,
  4032×2268, 25 i/s, sans tone curve ni OCIO, 8 échantillons temporels, 32 frames de warm-up,
  obturateur centré). La comparaison `test` / `diff` isole l'empreinte du personnage
  (ombre, occlusion, reflets). Les réglages du personnage sont restaurés après `diff`.
- `full` : Movie Render Graph `/Game/Cinematics/MRQ/MRG_Take41_DVR`, copie de
  `DVRDefaultRenderGraph` (EXR ZIPS, métadonnées `unreal/*`, plage de la séquence
  + 12 frames de handles = 127756-129800, nom `Scene_1_41_Layer1_{frame_number}`).
- Les rendus tournent en arrière-plan : lancer une commande à la fois (ou `tout`).

#### 2) Timecode et espace couleur dans Resolve — `src/resolve/resolve_timecode_colorspace.py`

Installation : copier le fichier dans
`%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility\`,
puis Workspace > Scripts > Utility > `resolve_timecode_colorspace`.
On peut aussi le lancer en ligne de commande, Resolve étant ouvert
(Préférences > Système > Général > External scripting using = Local) :
`python src/resolve/resolve_timecode_colorspace.py`.

- Start TC = numéro de la première frame ÷ fps (numérotation Unreal basée sur le timecode) :
  `Scene_1_41_Layer1_[00127756-00129800].exr` → **01:25:10:06** à 25 i/s.
- **Avant de lancer le script, sélectionner le ou les clips EXR dans le Media Pool de DaVinci Resolve**
  (page Media ou Edit). Seuls les clips sélectionnés sont traités, et leur Start TC est réécrit même s'il
  existe déjà.
  Sans sélection, le script traite toutes les séquences du projet encore à 00:00:00:00
  (un timecode existant n'est pas écrasé), sauf les rendus Unreal (`*_Layer<N>_*`) : ils sont
  toujours recalés, car Resolve leur attribue à l'import un Start TC faux
  (ex. 01:24:27:22 au lieu de 01:25:10:06).
- Le Start TC ne déplace pas les clips déjà posés dans la timeline : les y remettre ensuite
  (ou les re-synchroniser au timecode).
- Rendus Unreal (`*_Layer<N>_*`) : Input Color Space = Rec.709 (Scene).
- **Input Gamma = Linear reste à régler à la main** (Clip Attributes, plusieurs clips
  sélectionnés à la fois) : l'API de Resolve 21 ne permet pas de le modifier.
  Le script liste les clips concernés dans la console (Workspace > Console).

#### 3) (Optionnel) Timecode écrit dans les EXR — `src/exr_timecode.py`

Ajoute les attributs OpenEXR `timeCode` et `framesPerSecond` à tous les EXR d'un dossier,
sans toucher aux pixels (seuls l'en-tête et la table des offsets sont réécrits).
Utile si les fichiers doivent porter leur timecode hors de Resolve ; plus lent
(chaque fichier est réécrit, ~1 fichier/s pour des EXR 4K de 22 Mo).

```bash
python src/exr_timecode.py "<dossier des EXR>" --fps 25
python src/exr_timecode.py "<dossier des EXR>" --dry-run   # simulation
```

Pris en charge : EXR scanline mono-partie (pas de tuiles, multipart ni deep).

#### 4) Matching d'éclairage — `src/light_match.py` + `src/resolve/light_match_resolve.py`

Rapproche l'éclairage de l'acteur keyé (plaque Nikon ZR, comp Fusion du clip acteur de `Take_41`)
de celui d'une image de référence, en tenant compte du décor Unreal qui l'entoure.

**Principe.** La référence est une image étalonnée : on compare des *rapports*, mesurés de la
même façon des deux côtés, sur l'image de sortie (Rec.709 gamma 2.4, look compris) décodée en linéaire.
L'anneau de décor autour de l'acteur est coupé en deux à sa verticale : côté key (celui d'où la
lumière frappe l'acteur, d'après `key_side`) et côté fill.

| Mesure | Définition | Cible |
|---|---|---|
| `key_stops` | log2(p95 luminance acteur) − log2(médiane du décor côté key) | référence (bornée, voir priorité) |
| `hi_stops` | log2(p95 luminance acteur) − log2(p95 du décor autour, hors pixels écrêtés) | référence (corrigé sur le décor, critère `decor`) |
| `contrast_stops` | log2(p90 / p20) de la luminance de l'acteur (ratio key/fill) | référence (priorité `reference` seulement) |
| `warmth_r`, `warmth_b` | log2(R/G), log2(B/G) des hautes lumières de l'acteur moins la teinte moyenne du décor | référence (priorité `reference` seulement) |
| `light_r`, `light_b` | idem, moins la teinte des hautes lumières du décor côté key (couleur de la lumière) | référence, bornée à ±`--margin` |
| `black_stops` | log2(ombres de l'acteur, 5 % les plus sombres) − log2(ombres du décor côté fill), mesuré sans le look | 0 |
| `sat_stops` | log2(saturation des tons moyens de l'acteur / celle du décor autour), chaque zone neutralisée par la couleur de ses hautes lumières | référence, bornée à ±`--margin` |
| `shadow_stops` | log2(médiane de l'acteur dans l'ombre portée `Sink_Shadow`) − log2(médiane du décor dans la même ombre) | 0 |
| `bg_stops`, `bg_r`, `bg_b` | exposition et teinte du décor lui-même | référence (image entière) |
| `key_side` | côté des hautes lumières dans l'acteur (−1 gauche, +1 droite) | non corrigée : à reprendre à l'éclairage |
| `key_fill_stops` | écart key/fill de l'acteur moins celui du décor | non corrigée (un grade uniforme ne le change pas) |

La partie de l'acteur couverte par l'ombre portée (matte de `Sink_Shadow`) est retirée des autres
mesures : sinon ses 5 % les plus sombres seraient l'ombre ajoutée, et les noirs de tout l'acteur
seraient relevés. Elle est mesurée à part (`shadow_stops`) et règle le gain de `Sink_Shadow`.
Les mesures de référence d'une version antérieure sont refusées : relancer `reference`.

**Hautes lumières du décor** (`decor`, défaut en priorité `integration`, à la place de `exposure`).
`key_stops` compare l'acteur à la *médiane* du décor : elle dépend de ce que l'anneau contient de
sombre (étagère, meuble) et, bornée par le contraste du décor, demandait d'assombrir l'acteur de 4 stops
sur `Take_41`. `hi_stops` compare plutôt hautes lumières à hautes lumières. Sur `Take_41` (frame 884),
l'acteur et la médiane du décor sont au niveau de la référence ; ce sont les surfaces éclairées du décor
CGI qui sont environ 3 stops trop sombres (`hi_stops` +5,0 contre +2,1). Le critère `decor` corrige donc
le décor, dans le ColorGain `BG_Match` (créé une fois juste avant `BG_Merge.Background`) :
`out = G × in^k`, gamma pivotant sur la médiane du décor (mesurée sans le look), qui monte ses hautes
lumières sans changer son niveau moyen. `exposure` et `decor` sont exclusifs.
Vérifié sur la frame 884 : le still sans look décodé en gamma 2.4 correspond au rendu Unreal linéaire à
0,03 stop près (pente log-log 0,999), le pivot est donc mesuré dans le domaine où `BG_Match` agit.

Les images sont réduites à 1008 px de large avant la mesure (référence comprise) : 34 s → 1,6 s par
frame en 4032 de large, pour des rapports identiques à 0,02 stop près. Seule `sat_stops` bouge
(~0,2) : la réduction moyenne le grain du look, qui gonflait la saturation mesurée en pleine
résolution. Relancer `reference` après toute mise à jour de la mesure.

Trois précautions de mesure :

- **Pixels écrêtés** (un canal au-delà de 0,97 en code value) : exclus des hautes lumières qui servent
  à la teinte (`light`, `warmth`). Un blanc écrêté par le look ne change plus de rapport R/G quand on le
  corrige, et la boucle stagnait. La part écrêtée de l'acteur est signalée au-delà de 5 %.
- **Saturation neutralisée** : chaque zone est divisée par la couleur de ses hautes lumières avant la
  mesure. Réchauffer ou refroidir l'acteur ne change donc pas `sat_stops` (avant, la boucle désaturait
  pour compenser la teinte).
- **Noirs sans le look** : `black_stops` et `shadow_stops` sont mesurés sur `still_prelook.tif`, car le
  look écrase les ombres du décor à 0 et les corrections agissent avant lui. Sous 2^-11 (luminance
  linéaire), les ombres du décor sont déclarées écrasées et ces mesures ignorées.

**Priorité** (`--priority`) :

- `integration` (défaut) : l'acteur est d'abord **calé sur le décor CGI** : mêmes noirs (niveau et
  teinte, `blacks`), même couleur de lumière (`light`). L'exposition suit la référence, mais sans
  s'écarter de plus de `--margin` stops (0,25 par défaut) de ce que montrent les hautes lumières du décor ;
  de même pour la saturation (`saturation`), et l'ombre portée vise celle du décor (`shadow`).
  C'est un calage deux points par canal : le gain cale les hautes lumières, le gamma les ombres ; le
  contraste de l'acteur en découle (affiché, non visé). La fidélité à la référence (exposition et teinte
  du décor) est ensuite visée sur **l'image entière**, après le merge : un gain commun ne change pas le
  raccord acteur/décor.
- `harmony` : **sans référence** (`--reference` facultatif), l'acteur est seulement harmonisé avec le
  décor qui l'entoure, contre l'effet de collage. Couleur de la lumière, saturation, noirs et ombre
  portée visent ceux du décor (écart nul). L'exposition de l'acteur (critère `match`) cale ses hautes
  lumières à `--hi-target` stops au-dessus de celles du décor (1,0 par défaut ; un T-shirt blanc peut être
  plus clair que du carrelage). Sur `Take_41`, la cible 0 assombrissait l'acteur de 3,6 stops jusqu'au
  niveau d'un décor CGI sous-exposé ; 1,5 a été retenu. La dominante jaune venait surtout de `BG_Warm`
  (bleu du décor à 0,65), transmise à l'acteur par `light` : `BG_Warm` a été remis neutre.
- `reference` : ancien comportement, les rapports acteur/décor de la référence sont imposés à l'acteur
  seul (gamma commun réglé par le contraste).

Corrections :

- acteur : `out = gain_c × in^k_c` par canal, dans le ColorGain `Person_Match` (créé une fois entre
  `Person_Expo` et `BG_Merge.Foreground`) ;
- image entière : `out = gain_RGB × in`, dans le ColorGain `Ref_Grade` (créé une fois entre `BG_Merge`
  et `Look_Film`), seulement si `--global` n'est pas `none`.

Les deux sont **avant** `Look_Film` : la boucle compense la non-linéarité du look (pas amorti,
convergence à ±0,1 stop d'exposition, ±0,15 stop de contraste et de noirs, ±0,03 de teinte).
Aucune ne touche `Plate_Expo` ni les entrées de `CK2_Key` : pas de nouveau rendu CorridorKey.

**1. Mesurer la référence** (une fois). L'acteur est donné par un masque PNG (blanc = acteur),
ou par une boîte `x0,y0,x1,y1` affinée par GrabCut (peu fiable sur des images sombres : préférer le masque).
Les bandes noires (letterbox) sont exclues automatiquement.

```bash
python src/light_match.py reference --image ref.png --subject-mask ref_mask.png -o ref_metrics.json
```

**2. Exporter le comp depuis Resolve.** `export_stills` exporte dans un dossier, **sans rendu Fusion** :
`still.png` (la frame de la timeline, look compris), `still_prelook.tif` (la même avec
`Look_Film` et `Look_Warm` en bypass, 16 bits, seulement si `blacks`, `shadow` ou `decor` sont corrigés), `matte.png` (le matte de l'acteur à sa place
dans l'image) et `layers.json`. Pour le matte, `MediaOut1` est branché quelques secondes sur le
nœud `LM_Matte` (créé au premier appel avec `LM_Black` et `LM_White` : blanc là où `Person_Match`
est opaque, Center lié à `BG_Merge.Center`), puis rebranché sur `Out_CST`. Si `Sink_Shadow` existe,
un troisième still `shadow.png` est exporté de la même façon (nœud `LM_ShadowMatte`). Avant chaque
export, le script vérifie la source de l'acteur, trouvée en remontant depuis `Person_Expo` : pour un
rendu CorridorKey en EXR (Loader), le fichier de la frame doit exister ; pour le plugin, il attend
qu'il ait fini la frame et s'arrête s'il est en erreur. Depuis le MCP Resolve
ou la console (un appel du MCP est limité à 60 s : la boucle se lance plutôt en externe, voir plus bas) :

```python
import sys; sys.path.insert(0, r"<repo>\src\resolve")
import light_match_resolve as lmr
lmr.resolve = resolve                      # objet injecté par la console / le MCP
shot = lmr.get_comp("Take_41")             # clip dont le comp contient BG_Merge
lmr.export_stills(shot, r"<dossier>\iter_01", "01:00:26:14")
```

**3. Calculer la correction**, puis l'appliquer, et recommencer à l'étape 2 jusqu'à `(convergé)` :

```bash
python src/light_match.py solve --layers <dossier>/iter_01 --reference ref_metrics.json \
    [--previous <dossier>/iter_00/correction.json] [--priority integration] [--correct exposure,light] \
    -o <dossier>/iter_01/correction.json
```

```python
import json
corr = json.load(open(r"<dossier>\iter_01\correction.json"))
lmr.apply_correction(shot, corr, with_global=bool(corr["corrected_global"]))
```

`--correct` limite la correction de l'acteur à certains critères (`exposure`, `contrast`, `warmth`,
`light`, `blacks`, séparés par des virgules ; défaut `exposure,light,blacks`, ou
`exposure,contrast,warmth` en priorité `reference` ; `warmth` et `light` sont exclusifs, de même que
`contrast` et `blacks`). `--global`
fait de même pour l'image entière (`exposure`, `warmth` ou `none`). Utile quand un écart vient de
l'éclairage et non du grade : par exemple sans `contrast` quand la key n'arrive pas du même côté que
dans la référence. La convergence n'est jugée que sur les critères corrigés (marqués `*` dans le rapport).
Le gamma cumulé est borné à [0,4 ; 2,5] et le gain à ±6 stops ; le rapport signale quand une borne est
atteinte (cible hors d'atteinte en étalonnage, par exemple un acteur bien plus contrasté que le décor).

Avant toute mesure, le masque de l'acteur est contrôlé : s'il remplit plus de 75 % de son rectangle
englobant (une silhouette en remplit environ 63 %), la mesure est refusée. C'est le signe que le key a
échoué et que toute la zone de garbage est passée opaque.

**Boucle automatique.** Elle enchaîne export, mesure et correction jusqu'à convergence. Elle passe par
l'API externe de Resolve (Préférences > Général > External scripting = Local) :

```bash
export RESOLVE_SCRIPT_API="C:/ProgramData/Blackmagic Design/DaVinci Resolve/Support/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="C:/Program Files/Blackmagic Design/DaVinci Resolve/fusionscript.dll"
export PYTHONPATH="$RESOLVE_SCRIPT_API/Modules"
python src/light_match.py loop --reference ref_metrics.json --work <dossier> \
    --frame 01:00:13:07 --max-iter 6

# plusieurs frames (temps du comp ou timecodes) : correction unique calée sur leur moyenne
python src/light_match.py loop --reference ref_metrics.json --work <dossier>     --frames 394,640,884,1130,1373 [--animate]
```

Avec `--frames`, le rapport signale les mesures qui varient de plus de deux fois leur tolérance
d'une frame à l'autre. `--animate` règle alors une correction par frame, posée en keyframes sur
`Person_Match`, `Sink_Shadow` et `Ref_Grade` (interpolées entre les frames). Une fois une entrée
animée, la boucle sans `--animate` refuse de la régler (supprimer l'animation dans Fusion).
`--exclude-bottom 0.08` retire le bas du cadre du décor (ancien raccord de sol `Bas_Plate`).

Ne pas utiliser Resolve pendant la boucle : elle déplace la tête de lecture, change de page et
rebranche brièvement `MediaOut1`.

Durée : environ 6 s d'export et 2 s de mesure par frame (contre ~57 s avant). Chaque still est exporté
après une pause d'1 s ; quand le comp vient de changer (look en bypass, `MediaOut1` rebranché), le
script vérifie que l'image a bien changé et réessaie sinon (contrôle actif seulement avec OpenCV, donc
dans la boucle externe, pas depuis le Python de Resolve). La boucle s'arrête d'elle-même quand plus
aucun critère hors tolérance ne réduit son écart d'au moins la moitié de sa tolérance (stagnation).
Pour aller vite : faire converger sur une frame (`--frame`), puis vérifier sur 3 (`--frames 394,884,1373`).

Testé le 2026-10-22 sur `Take_41`, frame 01:00:13:07 : convergence en 3 itérations
(exposition −0,05 → −0,01 stop, bleu relatif +0,06 → +0,015), sans attente de Resolve, en partant du
réglage fait sur 01:00:26:14. Non testé : l'ordre gain puis gamma du ColorGain quand k ≠ 1
(correction de contraste).

Précautions :

- Le graphe de connexions Fusion est relevé avant et après chaque opération : toute connexion
  modifiée de façon inattendue lève une exception. Ne jamais créer d'outil avec
  `AddTool(…, -32768, -32768)` (Fusion l'insère alors dans le flux).
- Éviter `export_layers` (rendu Fusion par `comp.Render()`, conservé pour référence) : après un rendu
  Fusion lancé par script, Resolve refuse `ExportCurrentFrameAsStill` un long moment, même une fois
  `GetCurrentPage()` revenu ; et CorridorKey a échoué pendant un de ces rendus (`key 'output' not found`),
  laissant passer la plate non keyée.
- `SetCurrentTimecode` peut échouer sans erreur : le script relit la tête de lecture et réessaie.
- Dans ce projet, Fusion échange ses images avec la timeline en **linéaire** (gamut DaVinci WG),
  pas en DaVinci Intermediate : `Plate_CST` doit avoir une entrée DWG / Linear et `Out_CST` une
  sortie DWG / Linear. Avec Intermediate, la plate était assombrie d'environ 3 stops (d'où un Gain
  de 10 pour compenser) et le décor Unreal ressortait 6 à 10 fois trop clair.
- Take_41 a sa propre gestion couleur (SDR 100, Input/Output DRT None, DaVinci WG / Intermediate).
  Il reste un écart constant d'environ −0,7 stop entre le décor dans le comp et le même EXR posé
  directement sur la timeline. Mesurer toujours la sortie du comp.

**Réglages du comp de `Take_41` issus de ce matching** (frame de référence 01:00:26:14) :

| Nœud | Rôle | Réglage |
|---|---|---|
| `BG_Warm` (ColorGain, `BG_Unreal` → `BG_Merge`) | réchauffe le décor seul, en linéaire | gain bleu 0,65 |
| `Look_Film` | look | `ColorContrast` 1,0 (1,25 écrasait les ombres), `ColorExposure` +1, `ColorSourceWB` 10 000 K |
| `Look_Warm` (ColorGain, `Look_Film` → `Out_CST`) | teinte d'ensemble | gain rouge 0,92, bleu 0,91 |
| `Person_Match` (ColorGain, `Person_Expo` → `BG_Merge`) | exposition et teinte de l'acteur | gain R/G/B 2,72 / 2,70 / 2,58 (laissé par la boucle) |

Ce que l'étalonnage ne corrige pas : le contraste key/fill de l'acteur (4,6 stops contre 8,9 dans la
référence) et la direction de la key (de face au lieu de la droite). Il faut une lumière directionnelle,
au tournage ou par relighting.

#### 5) Retouche IA avec Runway Aleph 2.0 — `src/runway_app.py`

Après le compositing, Runway Aleph 2.0 (`aleph2`, service cloud) peut affiner l'intégration à partir
d'un prompt texte (exemple : `prompt_input/realistic.txt`). La clé d'API se met dans
`RUNWAYML_API_SECRET`, dans un fichier `.env` à la racine du dépôt.

Aleph facture à la seconde de vidéo produite, et les formats ProRes / PNG coûtent 5 crédits/s de plus.
Testez donc un prompt sur quelques secondes avant de lancer la vidéo complète :

```bash
.venv/Scripts/streamlit run src/runway_app.py
```

1. **Export timeline** : rendu depuis Resolve de la timeline complète (≤ 1080p, ≤ 30 i/s, sans audio)
   dans `video_input/timeline_complete/`.
2. **Extrait** : découpe de 2 à 30 s de la timeline dans `video_input/extraits/`, nommée
   `<timeline>_t<début>_d<durée>.mp4`.
3. **Générer Prompt et Keyframe** : préparation, sans rien envoyer à Aleph. Le prompt se modifie puis
   s'enregistre explicitement (nouvelle version `_v2`, `_v3`… ou écrasement). La keyframe retenue est
   gardée pour chaque vidéo dans `video_output/tests/keyframes/selection.json`.
   L'**assistant Claude** part d'une intention écrite en français et de l'image du plan à l'instant de la
   keyframe. En un seul appel, il rédige le prompt Aleph pour une génération sans keyframe (court : verbe
   d'action + transformation, selon le [guide Aleph 2.0](https://help.runwayml.com/hc/en-us/articles/52150503729171-Aleph-2-0-Prompting-Guide)),
   avec deux variantes, et, si une image de référence est importée, le prompt de retouche de la keyframe
   (« Utiliser pour la keyframe »). Une fois la keyframe retenue, **« Prompt Aleph court pour cette
   keyframe »** compare l'image d'origine et la keyframe, et décrit seulement leur différence visible : c'est
   le prompt à envoyer avec la keyframe (l'API Aleph n'a qu'un seul champ texte par tâche). Il passe par la CLI Claude Code (`claude -p`),
   donc par l'abonnement connecté à la CLI, sans clé d'API Anthropic (modèles `opus`, `fable`, `sonnet`,
   `haiku`). Une seule **image de référence de look** (par exemple une image d'un autre film), importée en
   haut de l'onglet, sert à l'assistant, qui en décrit la lumière avec des mots, et à la génération de
   keyframe (`@look`). Elle n'est jamais envoyée à Aleph. Chaque proposition peut être reprise dans l'éditeur (« Utiliser »)
   ou enregistrée telle quelle dans `prompt_input/` (« Enregistrer », sans jamais écraser un fichier existant).
   Une **keyframe** est une image de la vidéo envoyée, retouchée : Aleph applique la retouche à tout le
   plan. C'est son mode de travail principal. L'app la produit de deux façons :
   - **générée** : la génération d'image de Runway (`gemini_image3_pro` ou `gen4_image`) reprend l'image
     du plan (`@shot`) avec la lumière d'une image de référence de look (`@look`, par exemple une image
     d'un autre film). Le prompt de retouche vient de l'assistant Claude ou s'écrit à la main. Les vidéos étant
     en 16:9, l'image est générée en 1920×1080 (`gen4_image`) ou 2752×1536 (`gemini_image3_pro`), puis remise
     à la taille de la vidéo. Vous choisissez la meilleure proposition ; les images sont gardées dans
     `video_output/tests/keyframes/` ;
   - **importée** : l'image d'origine exportée en pleine résolution, retouchée dans Resolve, puis réimportée.
     L'app signale une image dont le ratio ne correspond pas à la vidéo.
4. **Génération extrait** : essais sur un extrait avec un prompt enregistré, la keyframe retenue et une ou
   plusieurs seeds (une tâche par seed). Format mp4 par défaut, résultats dans `video_output/tests/`.
5. **Génération complète** : la timeline entière (30 s maximum), en ProRes 422 HQ par défaut, résultat
   dans `video_output/`. « Partir d'un essai réussi » reprend le prompt, la seed et la keyframe d'un essai
   de l'onglet 4 ; la keyframe est replacée dans la timeline grâce au début inscrit dans le nom de l'extrait.
6. **Suivi** : avancement et coût des tâches, téléchargement automatique, annulation, import d'une tâche
   lancée en ligne de commande. Historique : `video_output/tests/runs.json`.
7. **Comparer** : images source et résultats au même instant, et vidéo côte à côte générée avec ffmpeg.

Un extrait court ne montre pas le scintillement ni la dérive du visage sur un plan long : vérifiez-les
sur le premier rendu complet. En ligne de commande, `src/runway_aleph_video_to_video.py` fait la même
génération sur un fichier entier (`--check-task last` pour récupérer une tâche).
