# Nodal point (entrance pupil) — Nikon ZR + objectif 20 mm : quel outil ?

Trois options : **REtracker « High precision calibration »**, **Lens File d'Unreal Engine**, **script Python**.

## TL;DR

| | REtracker HPC | Unreal Engine (Lens File → Nodal Offset) | Script Python |
|---|---|---|---|
| Ce qui est calculé | Offset tracker → caméra, dans le logiciel REtracker | Offset **6 DoF** tracker → entrance pupil, dans le repère où UE rend l'image | Position de l'entrance pupil **le long de l'axe optique** uniquement |
| Connaît la pose du tracker ? | Oui | Oui (via Live Link) | **Non** |
| Même modèle de distorsion que le rendu ? | Non (modèle propre au logiciel) | **Oui** (Lens File) | Oui si on réutilise la calib de ce repo |
| Vérification visuelle (superposition CG/réel) | Selon le logiciel | **Oui**, intégrée | Non |
| Rôle conseillé | Si l'offset doit servir à plusieurs logiciels | **Référence pour l'ICVFX** | **Contrôle croisé** |

**Recommandation : Unreal Engine comme référence, Python comme vérification indépendante.**

- Pour l'ICVFX, la seule chose qui compte est la transformation **tracker → entrance pupil** dans le repère où l'inner frustum est rendu. Le Lens File d'UE la calcule à partir des mêmes données Live Link et du même modèle de distorsion que le rendu, puis permet de la valider en superposant un damier CG au damier réel pendant un pan.
- Un script Python seul **ne peut pas** donner le nodal offset complet : il ne connaît ni la position ni l'orientation du REtracker sur la cage. En revanche, il mesure très bien la distance capteur → entrance pupil le long de l'axe optique, indépendamment de tout logiciel. C'est un excellent garde-fou : la composante « avant/arrière » de l'offset UE doit lui correspondre (à la distance tracker ↔ plan capteur près, mesurable au réglet).
- REtracker HPC est pertinent si l'offset doit être intégré directement dans le flux de tracking (FreeD/Live Link) pour être utilisé par plusieurs logiciels. C'est en revanche une boîte noire, avec son propre modèle d'objectif.

## Règle absolue : un seul endroit pour l'offset

Si REtracker applique déjà l'offset dans la pose qu'il envoie, l'offset du Lens File UE doit être **nul**, et inversement. Un offset appliqué deux fois produit exactement le glissement (« swimming ») qu'on cherche à éliminer. Choisir l'un des deux et le noter dans la fiche de tournage.

## Points spécifiques au 20 mm

- Sur un grand-angle rétrofocus, l'entrance pupil se trouve en général **nettement en avant** du plan capteur, souvent dans le fût de l'objectif. Elle n'est pas au niveau de la monture : ne pas la deviner.
- Plus le champ est large et plus l'avant-plan réel est proche, plus une erreur d'offset de quelques millimètres devient visible. Un 20 mm est **plus exigeant** qu'un 28 mm sur ce point.
- La position de l'entrance pupil **bouge avec la mise au point**. Calibrer à la distance de mise au point du tournage, ou à plusieurs distances (le Lens File interpole sur le focus).
- Mêmes prérequis que pour la distorsion (cf. [calibration_Nikkor_28mm.md](calibration_Nikkor_28mm.md)) : format de production figé, VR/IBIS OFF, breathing compensation OFF, état ADC explicite.

## Workflow conseillé

1. **Distorsion et intrinsèques** : calibrées avec l'app de ce repo ou dans le Lens File (cf. doc 28 mm). Le nodal offset dépend de fx/fy : il doit être calculé **après**.
2. **Mesure physique (Python)** : déterminer la distance capteur → entrance pupil avec [nodal_point_parallax.py](nodal_point_parallax.py) (protocole ci-dessous).
3. **Nodal offset dans UE** : onglet Nodal Offset du Lens File (algorithme Checkerboard ou Points), environ 5 échantillons à des positions et des angles variés.
4. **Contrôle croisé** : comparer la composante avant/arrière de l'offset UE à la mesure de l'étape 2, en tenant compte de la position du REtracker par rapport au plan capteur (repère ⦵ sur le boîtier). Un écart de plus de quelques millimètres signale un problème (mauvais échantillons, axes du tracker mal alignés, offset appliqué deux fois).
5. **Validation finale** : pan/tilt rapide devant un objet réel au premier plan aligné avec le CG. Aucun glissement = OK.

## Protocole Python : méthode de parallaxe sur glissière nodale

Matériel : rotule panoramique ou tête vidéo + **glissière nodale graduée**, une cible proche (une tige verticale à environ 0,5–1 m) et une cible lointaine (une arête verticale à plus de 10 m), alignées au centre de l'image.

1. Poser la caméra sur la glissière. Noter la graduation `slide_mm`.
2. Panoramiquer de **+θ** (≈ 10–15°), noter `x_near − x_far` en pixels sur l'image (positions horizontales des deux cibles).
3. Panoramiquer de **−θ**, noter de nouveau `x_near − x_far`.
4. Recommencer à 3–5 positions de glissière qui encadrent le point attendu (par exemple tous les 10 mm).
5. Lancer le script. Il ajuste une droite parallaxe = f(position) et donne la graduation où la parallaxe s'annule. À cette position, **l'entrance pupil est sur l'axe de rotation**. Il reste à mesurer au réglet la distance axe de rotation ↔ repère plan capteur ⦵.

Avec une seule position de glissière, le script peut aussi estimer directement la distance pupille ↔ axe à partir de `fx` (calib de ce repo) et des distances aux deux cibles. Cette méthode est moins robuste, mais pratique pour un premier réglage.

```
python doc/nodal_point_parallax.py mesures.csv
python doc/nodal_point_parallax.py mesures.csv --fx 2950 --z-near 0.8 --z-far 15
```
