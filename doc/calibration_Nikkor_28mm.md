# Calibration d'objectif pour un pipeline ICVFX / nDisplay (Nikon ZR + Nikkor 28 mm, Unreal Engine 5.8)

## TL;DR
- **Oui, il faut calibrer** — même si le Nikon ZR corrige déjà la distorsion : sur le ZR la correction géométrique (« Auto distortion control ») est grisée/non appliquée en RAW (N-RAW, R3D NE, ProRes RAW) mais activable en H.265/ProRes 422 ; dans tous les cas la focale/FOV effectif du 28 mm ne vaut plus 28 mm nominal une fois corrigé (crop/étirement), donc les intrinsèques fx, fy, cx, cy doivent être mesurés sur l'image **telle qu'elle sort en format de production**.
- **Pour la mire, privilégier une ChArUco** rigide (dibond/mousse recommandé par Epic, ou composite alu ~6 mm), 15-40 vues couvrant tout le cadre (surtout bords et coins), modèle de distorsion Brown-Conrady 5 coefficients (k1,k2,p1,p2,k3), viser une RMS de reprojection < 0,5 px (idéalement 0,2-0,3), et se méfier d'une RMS trop basse (surajustement).
- **Le plus important pour l'ICVFX n'est pas seulement la distorsion mais le nodal offset** (entrance pupil) entre le REtracker Bliss et le centre optique : sans lui, la parallaxe entre plateau réel et inner frustum « glisse » lors des pans/tilts. Le plugin Camera Calibration d'UE 5.8 (Lens File) gère distorsion + intrinsèques + nodal offset dans un même workflow et est recommandé plutôt qu'OpenCV externe pour ce cas.

## Key Findings

1. **Le ZR ne délivre PAS une image « pré-corrigée » en RAW.** La règle Nikon (page NPS « Z 9 TIPS – Picture Quality ») indique verbatim : *« Auto distortion control is not applied to RAW footage… Note that [ON] may be selected automatically with some lenses, in which case this item will be grayed out and unavailable »* ; de même *« Vignette control is not applied to RAW footage »* et *« Diffraction compensation is not available during RAW recording »*. La capture de menu du ZR (firmware 1.01, mode R3D NE) montre l'option Auto distortion control **grisée**. En H.265/ProRes 422 elle est en revanche activable/désactivable. → Il faut calibrer dans le format exact de tournage.
2. **Le 28 mm f/2.8 Z a -2,39 % de distorsion en barillet non corrigée** (mesure Photography Life : *« The Nikon Z 28mm f/2.8 has a moderate amount of barrel distortion, -2.39% by our tests »*, à comparer aux Z 24mm f/1.8 S à -1,34 %, Z 35mm f/1.8 S à -1,21 % et Z 40mm f/2 à -0,62 %), avec un profil de correction obligatoire (non désactivable au niveau logiciel type Lightroom). C'est modéré : un modèle 5 coefficients suffit largement, pas de fisheye.
3. **La correction change le FOV** : corriger un barillet recadre/étire l'image, donc la focale effective ≠ 28 mm nominal. D'où la nécessité de mesurer fx/fy même si la distorsion résiduelle est nulle.
4. **ChArUco > damier** pour ce cas. OpenCV le recommande explicitement : *« it is highly recommended using the ChArUco corners approach since the provided corners are much more accurate »*, et *« The ArUco part is used to interpolate the position of the chessboard corners, so that it has the versatility of marker boards, since it allows occlusions or partial views… When high precision is necessary, such as in camera calibration, Charuco boards are a better option »*. Détection sub-pixel, coins identifiés de façon unique, couverture des bords/coins (crucial pour la distorsion radiale).
5. **UE 5.8 dispose d'un Lens File** qui stocke distorsion, intrinsèques (Fx,Fy,Cx,Cy normalisés) et nodal offset. La doc « Camera Lens Calibration Overview » précise : *« The plugin supports a spherical-distortion model, based on the Brown[-Conrady] »*, avec algorithmes de nodal offset Points, Aruco Markers, Checkerboard et Optical Axis ; les données sont stockées par focus/zoom et interpolables. Le plugin s'appuie sur OpenCV en interne.

## Details

### 1. Réussir la calibration OpenCV du Nikkor 28 mm

**Choix de la mire.** Trois familles existent : damier (checkerboard), ChArUco, et grilles de cercles asymétriques.
- *Damier* : coins détectés avec grande précision sub-pixel, mais l'ensemble du damier doit être visible dans chaque image, ce qui empêche de couvrir les extrêmes bords/coins du cadre — or c'est là que la distorsion radiale est la plus forte et la mieux contrainte. Ambiguïté de rotation à 180° si lignes/colonnes ne sont pas paire/impaire.
- *ChArUco* (damier + marqueurs ArUco dans les cases blanches) : combine la précision sub-pixel des coins d'échiquier avec l'identification unique des marqueurs ArUco. Comme le résume la doc OpenCV, la partie ArUco *« is used to interpolate the position of the chessboard corners, so that it has the versatility of marker boards, since it allows occlusions or partial views »*. On peut donc placer la mire de façon à ce que la caméra n'en voie qu'une partie et récolter des données jusqu'aux coins extrêmes de l'image. C'est pourquoi ChArUco est aujourd'hui recommandé. Inconvénient : nécessite une résolution suffisante pour lire l'encodage des marqueurs.
- *Cercles asymétriques* : centroïdes robustes au flou, mais moins répandus et sensibles à la distorsion perspective sur les bords.
**Recommandation : ChArUco**, dictionnaire type DICT_4X4 ou 5X5, damier d'environ 10×7 cases.

**Fabrication/impression.** La planéité est critique : toute ondulation introduit une erreur systématique interprétée comme de la distorsion. La doc UE 5.8 (« Camera Lens Calibration Quick Start ») recommande d'imprimer *« on di-bond or foam board »* et note que *« larger checkerboards will be more useful for wider angle lenses to improve coverage »* (pertinent pour votre 28 mm). Les fournisseurs métrologiques (calib.io) utilisent un composite aluminium/LDPE (type Dibond) d'environ 6 mm, plat, stable thermiquement, avec impression UV directe matte pour éviter les reflets, et une précision inter-éléments de l'ordre de quelques microns. En DIY : imprimer chez un imprimeur pro, contrecoller sur dibond/alu/verre, prévoir une bordure blanche, finition mate. **Toujours re-mesurer les carrés réels après impression** (les imprimantes mettent à l'échelle) et renseigner la vraie taille dans le code. Taille de mire : suffisamment grande pour remplir une bonne partie du cadre à distance de tournage.

**Nombre et diversité des vues.** Viser **15 à 40 images** (typiquement 20-30). Couvrir tout le cadre : centre, quatre bords, quatre coins. Varier les inclinaisons (tilt/pitch/yaw ~20-45°) et les distances. Éviter les poses trop frontales et toutes semblables (dégénérescence, ambiguïté focale/distance). L'objectif n'est pas de « voir la mire nette » mais de contraindre le modèle sur tout le domaine image.

**Réglages de prise de vue.**
- *Mise au point fixe et verrouillée* : la calibration n'est valable que pour une distance de mise au point donnée (le focus breathing modifie la focale/FOV). Verrouiller le focus à la distance de tournage.
- *Diaphragme* : fermer pour la profondeur de champ (f/5.6-f/8), mais éviter la diffraction au-delà de ~f/11 sur plein format.
- *Éclairage uniforme sans reflets*, shutter rapide, ISO bas, pas de flou de bougé.
- **Désactiver toute stabilisation (IBIS/VR)** : elle déplace le capteur/point principal et change la distorsion image par image, ce qui ruine la constance des intrinsèques.
- **Désactiver la compensation de breathing** sur le ZR pendant la calibration (elle applique un crop variable ; elle est de toute façon indisponible en RAW et en FHD 200/240p, et requiert Auto distortion control = ON).

**Format d'enregistrement.** Calibrer dans le **format exact de production** (même résolution, même crop, même ratio). Attention aux différences photo/vidéo (crop factor, oversampling) : les intrinsèques changent. Sur le ZR, chaque mode a son image area : 6K/4K ≤60p en pleine largeur FX (35,9 × 20,2 mm en vidéo 16:9), 4K 120p en crop DX (*« this is cropped to a DX format when shooting 4K 100p or 120p, thereby applying a 1.5x crop to the focal length »*, Digital Camera World), FHD 240p à ~95 % du champ FX. **Calibrer dans le mode précis qui sera tourné.**

**Modèle de distorsion.** OpenCV propose de 4 à 14 paramètres. Pour un 28 mm rectilinéaire, le **modèle standard Brown-Conrady à 5 coefficients (k1,k2,p1,p2,k3)** suffit. N'activer CALIB_RATIONAL_MODEL (k4,k5,k6), thin prism ou tilted que si des erreurs résiduelles structurées le justifient. **Ne pas utiliser le modèle fisheye** (28 mm n'est pas ultra grand-angle). Trop de coefficients → surajustement : le modèle « colle » au bruit et extrapole mal hors des points observés.

**Flags utiles.**
- CALIB_FIX_ASPECT_RATIO : impose fx=fy (raisonnable pour pixels carrés).
- CALIB_ZERO_TANGENT_DIST : force p1=p2=0 (objectif bien assemblé).
- CALIB_FIX_PRINCIPAL_POINT : verrouille cx,cy au centre (utile si peu de données, mais laisser libre si couverture complète).
- CALIB_USE_INTRINSIC_GUESS : initialise avec une estimation (focale en pixels ≈ largeur_px × 28 / 35,9).
- CALIB_FIX_K3 : souvent judicieux pour un objectif à faible distorsion.

**Interprétation de la RMS.** Viser **< 0,5 px**, idéalement **0,2-0,3 px** (à titre de repère, OptiTrack indique viser ~0,1 px de RMS de reprojection avec leur système). Mais une RMS très basse n'est PAS un gage de qualité : avec peu d'images ou trop de coefficients, elle chute par surajustement sans que le modèle soit fidèle. Analyser l'**erreur par image** (per-view) pour éliminer les mauvaises poses (flou, détection erronée), et valider en croisé (calibrer sur un sous-ensemble, tester sur le reste). Vérifier aussi que les erreurs de reprojection sont réparties aléatoirement (pas de motif directionnel systématique, signe de biais du modèle).

**Conversion OpenCV → Unreal.** OpenCV donne fx,fy,cx,cy en **pixels**. Relations :
- Focale en mm : F_mm = fx × (largeur_capteur_mm / largeur_px). Ex. : capteur 35,9 mm, image 6048 px, fx ≈ 4720 → F ≈ 28,0 mm.
- FOV horizontal : HFOV = 2·atan(largeur_capteur / (2·F_mm)), ou directement 2·atan(largeur_px/(2·fx)).
- Dans UE, on renseigne le **Filmback** (35,9 × 20,2 mm) et la **Focal Length** de la CineCamera ; le FOV en découle. Le Lens File stocke Fx/Fy **normalisés** (divisés par la taille capteur en mm ou la résolution) : après normalisation Fy/Fx ≈ ratio d'aspect, et le centre image ≈ (0,5, 0,5).

**Deux approches comparées.**
- *OpenCV externe* : contrôle total, scriptable, reproductible ; mais il faut ensuite importer les coefficients dans le Lens File (via ST Map ou saisie/Live Link Lens) et gérer soi-même le nodal offset.
- *Plugin Camera Calibration d'UE 5.8* : gère dans un seul asset (Lens File) la distorsion (Checkerboard/Points, qui appellent OpenCV en interne), les intrinsèques, l'ajustement du centre image, et surtout le **nodal offset** avec plusieurs algorithmes (Points, Aruco Markers, Checkerboard, Optical Axis). Il applique automatiquement l'effet de distorsion en post-process au rendu CG et à l'inner frustum, et les données sont interpolables sur focus/zoom. **Recommandé pour l'ICVFX** car intégré au pipeline nDisplay/Live Link. Seuls les modèles sphériques (Brown-Conrady) sont supportés — parfait pour un 28 mm.

**Nodal offset / entrance pupil.** C'est aussi important que la distorsion. Le tracker (REtracker Bliss) rapporte une pose (position/rotation) qui n'est pas au centre optique de l'objectif — généralement référencée à l'intersection de l'axe optique et du plan capteur. Le nodal offset (plus correctement « entrance pupil offset ») est la transformation entre ce point tracké et l'**entrance pupil** (point autour duquel l'objectif ne produit pas de parallaxe). Sans calibration, foreground réel et background CG se décalent (« swimming »/glissement) lors des pans/tilts — exactement ce que l'ICVFX doit éviter. Dans UE : onglet Nodal Offset du Lens File, algorithme Checkerboard ou Points (LED tracké), viser ~5 échantillons et vérifier l'alignement en superposant un damier CG au damier réel (curseur Transparency). Note optique : la position de l'entrance pupil varie avec le focus/zoom et peut se situer devant ou derrière le plan capteur ; elle est traitée comme un « intrinsèque » spatial.

**Exemple de code OpenCV (Python, API moderne 4.8+ ChArUco) :**
```python
import cv2, numpy as np, glob

# --- Paramètres de la mire (à adapter à VOTRE impression re-mesurée) ---
SQUARES_X, SQUARES_Y = 10, 7          # nb de cases
SQUARE_LEN = 0.030                    # côté d'une case en mètres (mesuré !)
MARKER_LEN = 0.022                    # côté marqueur ArUco en mètres (mesuré !)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                               SQUARE_LEN, MARKER_LEN, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

all_corners, all_ids = [], []
image_size = None
for f in glob.glob("calib/*.png"):    # frames extraites au format EXACT de prod
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
    if ch_corners is not None and len(ch_corners) > 8:   # assez de coins
        all_corners.append(ch_corners)
        all_ids.append(ch_ids)

# --- Assemblage des correspondances 3D/2D ---
obj_pts, img_pts = [], []
for c, i in zip(all_corners, all_ids):
    op, ip = board.matchImagePoints(c, i)
    obj_pts.append(op); img_pts.append(ip)

# --- Modèle 5 coefficients + focale initiale ---
fx0 = image_size[0] * 28.0 / 35.9      # estimation focale pixels (28mm / capteur)
K0 = np.array([[fx0, 0, image_size[0]/2],
               [0, fx0, image_size[1]/2],
               [0, 0, 1]], float)
flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
         cv2.CALIB_FIX_ASPECT_RATIO   |
         cv2.CALIB_ZERO_TANGENT_DIST  |
         cv2.CALIB_FIX_K3)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts, img_pts, image_size, K0, None, flags=flags)

print("RMS reprojection (px):", rms)   # viser < 0.5, idéal 0.2-0.3
print("K=\n", K, "\ndist=", dist.ravel())

# --- Erreur par vue pour éliminer les mauvaises poses ---
for j, (op, ip) in enumerate(zip(obj_pts, img_pts)):
    proj, _ = cv2.projectPoints(op, rvecs[j], tvecs[j], K, dist)
    err = cv2.norm(ip, proj, cv2.NORM_L2) / len(proj)
    print(f"vue {j}: {err:.3f} px")

# --- Conversion vers Unreal ---
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
F_mm = fx * 35.9 / image_size[0]
HFOV = np.degrees(2*np.arctan(image_size[0]/(2*fx)))
print(f"Focale effective ~{F_mm:.2f} mm, HFOV ~{HFOV:.2f}°")
```
*(Note : la syntaxe ChArUco a changé avec OpenCV ≥ 4.8 — utiliser `cv2.aruco.CharucoBoard((cols, rows), ...)` et `CharucoDetector`, et non l'ancien `CharucoBoard_create`.)*

### 2. La correction automatique du ZR est-elle un problème ?

**Ce que fait la correction in-camera Nikon.** Les hybrides Nikon Z appliquent des profils intégrés de correction de distorsion géométrique (et vignettage/aberration chromatique). Pour la plupart des objectifs Z natifs, l'« Auto distortion control » est **forcé/obligatoire** (option grisée) et le profil est écrit dans le fichier/métadonnées ; certains logiciels (Lightroom) l'appliquent de façon non désactivable. Pour le 28 mm f/2.8, la correction est décrite comme rendue *« permanently 'ON' for this lens, eliminating the option to switch it off »* (Phillip Reeve).

**Cas du Nikon ZR.**
- **RAW (N-RAW, R3D NE, ProRes RAW)** : la correction géométrique n'est **PAS** appliquée. Sur le ZR l'option Auto distortion control est **grisée** en R3D NE (confirmé firmware 1.01), cohérent avec la règle Nikon *« Auto distortion control is not applied to RAW footage »*. De même Diffraction compensation et Vignette control ne s'appliquent pas au RAW.
- **Compressé (H.265, ProRes 422 HQ, H.264)** : Auto distortion control est **activable/désactivable** par l'utilisateur (peut être auto-sélectionné et grisé pour certains objectifs).
- **Breathing compensation** : applique un léger crop, indisponible en RAW et en FHD 200/240p, et nécessite Auto distortion control = ON.
- À noter : même en RAW, seul DaVinci Resolve reconnaît/applique le profil de distorsion embarqué en N-RAW ; Premiere ne l'applique pas ; sur R3D NE même Resolve ne le rend pas actuellement.

**Conséquences pour la production virtuelle.**
- **Il faut calibrer l'image telle qu'elle sort dans le format de production.** Si vous tournez en R3D NE (RAW), l'image est distordue (~-2,4 % barillet du 28 mm) → coefficients k1,k2 non négligeables à mesurer et compenser sur l'inner frustum. Si vous tournez en H.265 avec Auto distortion control ON, l'image est quasi rectilinéaire → coefficients résiduels faibles, ce qui est **plutôt un avantage** (moins de distorsion à compenser sur l'inner frustum).
- **Mais la correction change le FOV effectif.** Corriger un barillet recadre/étire l'image : la focale et le FOV réels ne correspondent plus au 28 mm nominal. **Il faut donc mesurer fx,fy même si la distorsion est nulle**, sinon l'inner frustum ne correspondra pas à la perspective réelle (objets CG trop grands/petits, décalage aux bords).
- **Risque de correction variable/non déterministe** : la correction peut dépendre du mode, de la distance de mise au point (breathing), etc. → **figer tous les réglages** (format, focus, breathing off, ADC on/off explicite) entre calibration et tournage.
- **Ne jamais appliquer une double dé-distorsion** : si l'image est déjà corrigée in-camera ou en NLE, ne pas re-corriger dans UE (et inversement). Choisir un seul point d'application.
- **Cas où désactiver la correction interne** (tourner en RAW non corrigé) : intéressant si vous voulez maîtriser tout le pipeline (compositing, matchmove 3D, ST Map maison), avoir une source « propre » et déterministe, éviter les artefacts d'étirement de pixels. Inconvénient : il faut alors calibrer/appliquer soi-même la distorsion, et le champ vu est légèrement plus large (barillet non recadré).

**Vérification pratique.** Filmer une grille de lignes droites / une mire, dans chaque format candidat, et mesurer la courbure résiduelle. Comparer avec/sans correction quand c'est possible (H.265 ADC on vs off). Si les lignes sont droites → distorsion déjà neutralisée (ne calibrer que les intrinsèques + nodal offset). Si elles sont courbes → calibrer la distorsion complète.

## Recommendations

**Étape 0 — Figer la config caméra.** Choisir LE format de tournage (ex. R3D NE 6K 60p FX). Verrouiller focus à la distance de scène, IBIS/VR OFF, breathing compensation OFF, ADC dans un état explicite et noté. Ne plus rien changer ensuite.

**Étape 1 — Test rapide de distorsion.** Filmer une grille de lignes droites dans ce format. Lignes droites → distorsion neutralisée. Courbes → distorsion à calibrer.

**Étape 2 — Calibration des intrinsèques + distorsion.** Fabriquer/acheter une ChArUco rigide (dibond/mousse, mire re-mesurée), capturer 20-30 frames couvrant tout le cadre et les coins, angles 20-45°, distances variées. Lancer OpenCV (code ci-dessus) OU le Lens Distortion (Checkerboard) du Lens File UE. Viser RMS < 0,5 px, éliminer les vues > ~1 px. Vérifier que la focale effective mesurée est cohérente (peut différer de 28 mm nominal après correction/crop).

**Étape 3 — Nodal offset.** Dans le Lens File UE, calibrer l'offset entre REtracker Bliss et l'entrance pupil (algorithme Checkerboard ou Points/LED). Vérifier par superposition CG/réel pendant un pan : pas de glissement = OK.

**Étape 4 — Intégration nDisplay.** Renseigner Filmback 35,9 × 20,2 mm et la focale mesurée dans la CineCamera pilotant l'inner frustum. Vérifier que le FOV horizontal de l'inner frustum correspond au FOV mesuré. Éventuellement Field of View Multiplier ~1,1-1,2 et soft edge pour absorber la latence de tracking.

**Étape 5 — Validation finale.** Sur l'écran de rétroprojection, filmer un objet réel à la frontière de l'inner frustum et vérifier l'alignement/parallaxe en mouvement caméra (DJI RS4 Pro). Recalibrer si glissement.

**Seuils qui changent la décision :**
- RMS > ~1 px ou erreurs structurées → reprendre les captures (plus de couverture des bords, meilleure planéité/mise au point).
- Distorsion résiduelle > ~1 px de déplacement aux coins → activer un modèle plus riche (rational) OU appliquer la distorsion mesurée.
- Si vous changez de format, de focus, ou réactivez breathing/IBIS → **tout recalibrer**.

## Caveats
- Le string exact « Auto distortion control is not applied to RAW footage » a été trouvé sur la page NPS du Nikon Z9 (architecture RAW partagée), pas verbatim sur une URL ZR ; le comportement ZR est confirmé par la capture de menu grisé en R3D NE (firmware 1.01, blog Moose Peterson). À revérifier sur votre firmware.
- Les valeurs de distorsion du 28 mm (-2,39 %) proviennent de tests Photography Life (image non corrigée) ; votre exemplaire peut varier légèrement. Toujours mesurer sur VOTRE combinaison boîtier+objectif+format.
- Les dimensions capteur diffèrent légèrement selon les sources : Nikon donne le format vidéo FX à ~35,9 × 20,2 mm (16:9), tandis que la fiche photo pleine résolution est ~35,9 × 23,9 mm. Utiliser la valeur du **mode vidéo réellement tourné** pour le Filmback.
- Le terme « nodal point » est optiquement un abus de langage pour « entrance pupil » ; UE et la plupart des outils l'appellent nodal offset.
- L'écran de rétroprojection (projecteur AWOL Aetherion Max 4K) implique aussi une calibration colorimétrique/géométrique du projecteur distincte de la calibration objectif traitée ici.
- Le REtracker Bliss via Live Link doit être synchronisé en timecode/genlock avec le flux vidéo ; un décalage temporel produit un glissement qui n'est pas corrigible par la calibration objectif.
- Le Lens File UE stocke distorsion/intrinsèques/nodal offset par point de focus et de zoom ; sur une focale fixe verrouillée, un seul point de focus suffit, mais toute variation de focus en tournage exigerait plusieurs points calibrés.