# TODO-FINAL — mise à jour de 11h45 (cinq minutes)

Ce fichier est une liste de contrôle interne : **le supprimer avant de rendre le dépôt public** (ainsi que la ligne
`TODO-FINAL.md` de l'arborescence et l'encadré « TODO-FINAL » en tête du `README.md` principal).
Vérification finale : `grep -rn "TODO-FINAL" .` ne doit plus rien renvoyer.

État écrit dans le dépôt (lu dans les journaux de soumission à 07h25) :
tâche 1 = 0,7705 (soumission de 07h11) · tâche 2 = 1,00 · tâche 3 = 0,4038 (06h54).
Attention : le relevé transmis pour 07h25 indiquait 0,7685 pour la tâche 1 ; le journal de soumission de la tâche 1
contient 0,7705 à 07h11. Reprendre le chiffre du classement à 11h45.

## 1. README principal — `README.md`

| Ligne | Quoi mettre |
|---|---|
| 6-8 | Supprimer l'encadré « TODO-FINAL » |
| 12 | En-tête de colonne : remplacer « at 07:25, 22 Nov » par l'heure du relevé final, ou garder la colonne et remplir « Final » |
| 14 | Tâche 1 : score public final (colonne **Final**) |
| 15 | Tâche 2 : score public final (colonne **Final**) |
| 16 | Tâche 3 : score public final (colonne **Final**) |
| 46 | Phrase « (0.3724 → 0.4038) » de la tâche 3 : mettre le score final si un meilleur run existe |
| 53 | Retirer la ligne `TODO-FINAL.md` de l'arborescence |

## 2. Tâche 1 — `task1_watermark_localization/README.md`

| Ligne | Quoi mettre |
|---|---|
| 6 | Score public final et heure (aujourd'hui « 0.7705 … 07:11 ») ; retirer le commentaire |
| 79-80 | Les deux dernières commandes (`assemble.sh …`, `common.py …`) : recette exacte de la meilleure soumission finale (membres de l'ensemble, `FULLSHIFT`, `KMAX`, `DELTA`, `THR`, fichier de masques SAM) |
| 83-84 | Retirer le commentaire une fois la recette à jour |
| 86-90 | Paragraphe « Best submission at 07:11 » : nom du fichier final, membres, nombres de rectangles (71 / 66 / 36), nombre d'images « full » décalées |
| 96-97 | Confirmer `KMAX` / `DELTA` / `THR` de la meilleure soumission et le modèle SAM de `objmasks2_all.npz` (le journal dit ViT-H, la ligne de job enregistrée utilise la valeur par défaut ViT-B), puis retirer le commentaire |
| 115 | Dernière ligne du tableau : heure, nom, score, ce qui a changé (ou supprimer la ligne si 0,7705 reste le meilleur) |

## 3. Tâche 2 — `task2_model_equality/README.md`

| Ligne | Quoi mettre |
|---|---|
| 8-9 | Score public final ; dire quelle configuration soumise est « notre réponse » (aujourd'hui : `subs/k01_H7.csv`, la plus probable) |
| 102 | Dernière ligne du tableau des soumissions (ou la supprimer) |
| 104-107 | Compteurs de budget par endpoint si de nouvelles réponses ont été collectées ; retirer le commentaire |
| — | Si la configuration retenue change : copier son CSV dans `task2_model_equality/subs/` et mettre à jour le tableau « Structure retained » |

## 4. Tâche 3 — `task3_unlearning/README.md`

| Ligne | Quoi mettre |
|---|---|
| 10 | Score public final, heure, nom du run |
| 45 | Titre « Reproduce (best submission at 06:54: run `pisr1_cap35mic_j`) » : heure et nom du run final |
| 62-65 | Lignes `make_cap.py` et `train_v3.py` : valeur du plafond et arguments du run final (ils sont dans le registre des lancements, colonne commande) |
| 72 | Retirer le commentaire |
| 88 | Dernière ligne du tableau des résultats (ou la supprimer) |
| 97 | Ligne « Cap on the true-class probability » : ajouter les nouvelles valeurs testées |

## 5. Diapositives — `slides/build_slides.py`, puis reconstruire

Tous les chiffres des diapositives sont dans le bloc `NUMBERS` en tête du script :

| Ligne | Quoi mettre |
|---|---|
| 22 | `FINAL = True` (retire la mention « TODO-FINAL: refresh before noon » du pied des diapositives 1, 2, 4, 6, 8, 9) |
| 23 | `AS_OF` : « Public leaderboard, 22 Nov, HH:MM » |
| 24-26 | `T1_SCORE`, `T2_SCORE`, `T3_SCORE` |
| 27-36 | `T1_LADDER` : dernière ligne (libellé, score, gain) si le meilleur score a changé |
| 44-52 | `T3_LADDER` : dernière ligne si le meilleur score a changé |
| 298, 303 | Diapositive 6 : « 4 endpoints out of 33 » et « 35 to 71 » si l'état de la tâche 2 a changé |
| 347 | Diapositive 9 : « 0.37 → 0.40 on task 3 » si le score final de la tâche 3 change d'arrondi |

Reconstruire (le dossier `.build/` est ignoré par git ; le supprimer ensuite) :

```bash
python3 -m venv .build/venv && .build/venv/bin/pip install -q python-pptx reportlab
.build/venv/bin/python slides/build_slides.py      # écrit slides/Fable4.2_slides.pptx et slides/Fable4.2_slides.pdf
rm -rf .build
```

## 6. Notes de l'orateur — `slides/NOTES_ORATEUR.md`

| Ligne | Quoi mettre |
|---|---|
| 8-9 | Heure du relevé et les trois scores |
| 29-33 | Diapositive 1 : les trois scores dits à l'oral (0,77 · 1,00 · 0,40) |
| 72-74 | Diapositive 4 : « pour arriver à 0,7705 » |
| 99-106 | Diapositive 6 : « 1,00 », « quatre endpoints sur trente-trois », « entre 35 et 71 réponses » |
| 136-141 | Diapositive 8 : « de 0,37 à 0,40 » |
| 159 | Diapositive 9 : retirer la mention une fois la diapositive à jour |

## 7. Dernier contrôle avant publication

```bash
grep -rn "TODO-FINAL" .                                   # doit être vide
grep -rnI -i -E 'KGAT_|API_KEY\s*=\s*["'"'"'][0-9a-f]|[0-9a-f]{32}|/p/scratch|/home/' .   # doit être vide
git status --short                                        # aucun .env, .pt, .npz, .jsonl, .pkl
```
