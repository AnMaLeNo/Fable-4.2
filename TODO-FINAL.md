# TODO-FINAL — ce qui reste à faire à 11h30 (trois gestes)

Liste de contrôle interne, à supprimer à la fin (geste 3). État écrit dans le dépôt, lu dans les journaux de soumission
à 09h40 : tâche 1 = 0,7965 (`00_G_C16_pure.pt`, 09h37) · tâche 2 = 1,00 · tâche 3 = 0,4162 (`wHcap40_j`, 09h38, une
seule soumission ; palier confirmé 0,4013-0,4038) · classement général 4es sur 44 à 0,92 (relevé de 09h25, non revérifié).

## 1. Relire les trois scores et le rang, et les reporter

| Fichier | Ligne | Quoi mettre |
|---|---|---|
| `README.md` | 13-15 | colonne **Final** des trois tâches (et corriger la colonne « 09:40 » si le meilleur a changé) |
| `README.md` | 17-18 | rang général (ou supprimer la phrase) |
| `task1_watermark_localization/README.md` | 6 | score, nom du fichier, heure ; si le meilleur a changé, ajouter UNE ligne au tableau « Results » (l. 137) |
| `task2_model_equality/README.md` | 8-9 | score (1,00 aujourd'hui) |
| `task3_unlearning/README.md` | 10-11 | score, run, heure ; si le meilleur a changé : titre « Reproduce » (l. 49), commandes (l. 63-71), une ligne au tableau « Results » (l. 97) |
| `slides/build_slides.py` | 23-27 | `AS_OF`, `T1_SCORE`, `T2_SCORE`, `T3_SCORE`, `RANK` ; dernière ligne de `T1_LADDER` (l. 28-39) et de `T3_LADDER` (l. 47-56) si le meilleur a changé |
| `slides/NOTES_ORATEUR.md` | 30-31, 71, 140 | les scores dits à l'oral (0,80 · 1,00 · 0,42 ; rang ; 0,7965 ; 0,42) |

Si le run de la tâche 3 à 0,4162 n'est pas confirmé par une seconde graine, garder la mention « single run » /
« une seule soumission » telle quelle.

## 2. Passer `FINAL = True` et reconstruire les diapositives

```bash
sed -i 's/^FINAL = False/FINAL = True /' slides/build_slides.py     # retire le rappel « TODO-FINAL » du pied des diapositives
python3 -m venv .build/venv && .build/venv/bin/pip install -q python-pptx reportlab
.build/venv/bin/python slides/build_slides.py && rm -rf .build       # écrit le .pptx et le .pdf ; ouvrir le .pptx une fois
```

## 3. Retirer les marqueurs et ce fichier

```bash
sed -i 's/ *<!-- TODO-FINAL[^>]*-->//' README.md task*/README.md
sed -i '/^> \*\*TODO-FINAL\*\*/,/^> What remains/d; /^TODO-FINAL.md /d' README.md   # encadré d'en-tête et ligne de l'arborescence
sed -i '/^`TODO-FINAL`/d; /^\*\*Chiffres à vérifier/,/^$/d' slides/NOTES_ORATEUR.md
git rm -q TODO-FINAL.md
grep -rn "TODO-FINAL" --include='*.md' . ; pdftotext slides/Fable4.2_slides.pdf - | grep -c "TODO-FINAL"   # rien, puis 0
```

(Les commentaires `# TODO-FINAL` de `slides/build_slides.py` sont de simples repères dans le script ; ils peuvent rester.)
Dernier contrôle avant de rendre public : `git status --short` ne montre aucun `.env`, `.pt`, `.npz`, `.jsonl`, `.pkl`.
