# Notes de l'orateur — Fable 4.2 (6 minutes, 9 diapositives)

Mode d'emploi : le texte en retrait (« À dire ») se lit tel quel, à voix posée. Il fait 894 mots, soit environ
5 min 58 à 150 mots par minute : c'est juste. Si le chronomètre dérive, sauter la dernière phrase des diapositives 3, 5
et 7 (on gagne environ 25 secondes). Les lignes « Pour comprendre » ne se lisent pas : elles expliquent les mots
techniques au cas où le jury pose une question.

**Chiffres à vérifier à 11h30** : les scores cités ont été lus à 09h40 dans les journaux de soumission
(0,7965 · 1,00 · 0,4162) et le rang à 09h25 (4es sur 44). Chaque passage concerné est signalé par `TODO-FINAL`.

| Diapositive | Durée | Temps cumulé |
|---|---|---|
| 1. Titre | 0 min 20 | 0 min 20 |
| 2. Vue d'ensemble | 0 min 36 | 0 min 56 |
| 3. Tâche 1, l'idée | 0 min 45 | 1 min 41 |
| 4. Tâche 1, ce qui a fait la différence | 0 min 44 | 2 min 25 |
| 5. Tâche 2, l'idée | 0 min 43 | 3 min 08 |
| 6. Tâche 2, le résultat | 0 min 41 | 3 min 49 |
| 7. Tâche 3, l'idée | 0 min 44 | 4 min 33 |
| 8. Tâche 3, ce qui a fait la différence | 0 min 40 | 5 min 13 |
| 9. Conclusion | 0 min 45 | 5 min 58 |

---

## 1. Titre (20 secondes)

À dire :

> Bonjour, nous sommes l'équipe Fable 4.2. En vingt-quatre heures, nous avons traité les trois tâches. Nos scores
> publics de ce matin : 0,80 sur la localisation de watermark, 1,00 sur l'égalité de modèles, 0,42 sur le
> désapprentissage, soit la quatrième place sur quarante-quatre équipes. Voici comment, deux minutes par tâche.

`TODO-FINAL` : les trois scores (0,7965 · 1,00 · 0,4162, lus à 09h40) et le rang (4es sur 44 à 09h25).

## 2. Vue d'ensemble (35 secondes)

À dire :

> D'abord, notre façon de travailler. Tout le travail technique a été fait par des agents Claude : un orchestrateur,
> qui fixe les priorités et partage les cartes graphiques, et un agent par tâche, qui tient son journal. Des agents
> de recherche viennent en renfort, pour relire ou chercher des idées. Nous, les humains, avons fourni les accès et
> pris les décisions. Pour le calcul : un nœud de quatre A100 sur JURECA et une carte RTX 3080 locale. À droite,
> l'idée de chaque tâche ; je les reprends une par une.

Pour comprendre : « agent » = un programme Claude qui travaille seul pendant des heures (il lit, écrit du code, lance
des calculs, soumet). L'orchestrateur ne résout pas les tâches, il coordonne les trois autres.

## 3. Tâche 1, l'idée (45 secondes)

À dire :

> Tâche 1. On reçoit une image retouchée, la méthode de watermark et le message caché ; il faut dessiner la zone
> encore tatouée. Nous avons trouvé trois types de retouche, et un outil pour chacun. Si toute l'image est tatouée,
> avec un trou : un réseau de segmentation, entraîné sur des images synthétiques fabriquées avec les outils fournis.
> Si seul un rectangle est tatoué : on décode des morceaux de l'image, et quand on retrouve le message connu avec au
> moins 85 % des bits justes, on tient le rectangle exact. Si le watermark ne reste que sur un objet : on recale la
> prédiction sur les contours donnés par Segment Anything.

Pour comprendre : « segmentation » = le réseau colorie les pixels tatoués. « Décoder » = relire le message caché dans
un morceau d'image ; cela ne marche bien que si le morceau est exactement le rectangle tatoué, ce qui le localise.
Segment Anything (SAM) est un modèle public qui découpe les objets d'une image.

## 4. Tâche 1, ce qui a fait la différence (45 secondes)

À dire :

> Ce qui a fait la différence, c'est une découverte sur les données. Les vraies images ont été légèrement agrandies,
> de zéro à sept pour cent, puis recadrées, après la pose du watermark. Un modèle entraîné sur nos images
> synthétiques propres ne voyait donc presque rien dans les vraies : 0,07. Avec le même zoom à l'entraînement, il
> monte à 0,83, sans aucune image réelle étiquetée. Ensuite : les rectangles trouvés par décodage, le recalage sur les
> objets, puis trois entraînements longs par méthode, pour arriver à 0,7965. Ces modèles n'utilisent aucune étiquette
> devinée sur le jeu de test : cet essai a été écarté, le modèle y recopiait son professeur.

`TODO-FINAL` : le score final (0,7965, lu à 09h37).

Pour comprendre : 0,07 et 0,83 sont des scores (IoU moyen) mesurés chez nous sur de vraies images StegaStamp
entièrement tatouées, par un modèle qui n'a vu aucune image réelle. Les scores du tableau sont ceux du classement public.
« Étiquettes devinées » (pseudo-labels) : on fait étiqueter les images de test par un premier modèle, puis on entraîne
un second modèle dessus. En entraînement long, ce second modèle recopie le premier au lieu d'apprendre : nous l'avons
retiré, et les neuf modèles de la solution finale sont entraînés sans.

## 5. Tâche 2, l'idée (45 secondes)

À dire :

> Tâche 2. Trente-trois modèles de langage cachés derrière des API, cent réponses par modèle, et il faut dire
> lesquels sont le même modèle. Comparer les réponses ne suffit pas : un modèle compressé ressemble plus à l'original
> qu'un modèle à qui on a seulement donné un personnage. Notre idée : reconstruire le modèle chez nous. Un
> questionnaire sépare d'abord trois familles. Puis on cherche, parmi les modèles publics, celui qui rend les réponses
> observées les plus probables : il se détache nettement. Avec cette copie locale, on connaît la probabilité exacte de
> chaque mot, et on teste chaque endpoint : température, fidélité question par question, compression, watermark.

Pour comprendre : « endpoint » = une adresse d'API derrière laquelle tourne un modèle inconnu. « Compressé »
(quantifié, « fp8 ») = mêmes poids stockés avec moins de précision : c'est un modèle différent selon l'énoncé.
Personnage, consigne système, température et watermark de texte ne changent pas l'identité du modèle. Le tableau du bas
donne le modèle trouvé pour chaque famille (Qwen 3B, Qwen 1,5B, Llama 1B) et son avance sur le second candidat.

## 6. Tâche 2, le résultat (40 secondes)

À dire :

> Résultat : chaque règle que nous pouvions justifier a fait monter le score. 0,10 avec les familles seules ; 0,75 en
> ne fusionnant que les déguisements identifiés avec certitude ; 0,85 avec les tests sur les copies locales ; puis
> 1,00 sur le classement public. Tout a d'abord été validé sur les neuf endpoints d'entraînement, dont la réponse
> est connue. Soyons honnêtes sur la limite : pour quatre endpoints sur trente-trois, nos tests ne tranchent pas, et
> le score public, déjà à 1, ne peut plus les départager. Nous avons dépensé entre 35 et 71 réponses sur les 100
> autorisées par endpoint.

`TODO-FINAL` : le score (1,00), le nombre d'endpoints non tranchés (4) et le budget (35 à 71).

Pour comprendre : le score public n'est calculé que sur une petite partie des paires d'endpoints. Les quatre
endpoints ouverts sont les numéros 01, 12, 21 et 19 : ils sont soit le modèle de base, soit une version à peine
modifiée. Le classement final retient notre meilleure soumission sur les paires cachées ; nous avons donc soumis
les lectures plausibles, de la plus probable à la moins probable.

## 7. Tâche 3, l'idée (45 secondes)

À dire :

> Tâche 3. Un modèle de vision doit oublier 3 500 images et se comporter comme un modèle réentraîné sans elles. Le
> score compare les prédictions image par image. D'où notre question : que répondrait un modèle qui n'a jamais vu
> cette image ? Pour le deviner, nous interrogeons quatre familles de modèles de substitution : des classifieurs
> simples sur des réseaux pré-entraînés, le modèle d'origine sur des variantes régénérées de l'image, des élèves
> entraînés sur des images générées, et le même réseau aux dernières couches remises à zéro. Leur consensus donne une
> cible par image. Il reste à ajuster le modèle vers ces cibles, en le gardant inchangé ailleurs.

Pour comprendre : le modèle d'origine connaît par cœur les 3 500 images (100 % de bonnes réponses). Un modèle qui ne
les a jamais vues se trompe sur une partie d'entre elles : nos cibles reproduisent ces erreurs (environ 45 % des images
reçoivent volontairement une mauvaise réponse) et le niveau d'hésitation attendu. Les « images ressemblantes » sont
générées par un modèle de diffusion ; elles servent à vérifier que le reste du comportement ne bouge pas.

## 8. Tâche 3, ce qui a fait la différence (40 secondes)

À dire :

> Chaque source d'information indépendante a ajouté du score : 0,15 au départ, 0,23 avec la première famille, 0,29
> avec deux, 0,35 avec trois, 0,36 avec quatre. Le serveur ne renvoyant qu'un nombre, nous n'avons changé qu'un
> réglage par soumission : c'est le tableau de droite. Le gain de 0,37 à 0,40 vient d'un second agent chargé de
> relire le premier. Son constat : une moyenne de modèles réentraînés n'est jamais tout à fait sûre d'elle ; il faut
> donc plafonner aussi la confiance de notre modèle. Ce matin, un cinquième signal nous amène à 0,42, sur une seule
> soumission pour l'instant.

`TODO-FINAL` : le score final (0,4162, lu à 09h38 ; une seule soumission, le palier confirmé est 0,4013 à 0,4038).

Pour comprendre : dans le tableau, « loss level » règle à quel point le modèle doit hésiter ou se tromper sur les
images à oublier (le mieux : ×2,0) ; « images wrong » est la part d'images auxquelles on demande une mauvaise réponse
(le mieux : 45 %). Le « cinquième signal » est le modèle d'origine dont on a supprimé une partie des poids (élagage) :
ce qu'il oublie en premier ressemble à ce qu'il n'avait fait que mémoriser. Ce qui n'a pas aidé : un modèle statistique de la référence ajusté sur nos scores passés (il
annonçait +12 %, le score a baissé de 0,3533 à 0,3010), les moyennes de poids, un ancrage plus fort.

## 9. Conclusion (45 secondes)

À dire :

> Pour conclure. Ce qui a marché : exploiter ce qui est connu, le message caché, les modèles publics, la formule du
> score ; mesurer avant de croire ; faire relire un agent par un autre. Ce qui n'a pas marché : le filtrage adapté
> sur le watermark, pas meilleur que le hasard ; les pseudo-étiquettes du jeu de test, que le modèle finit par
> recopier ; notre modèle de la référence, faux dès qu'on extrapole. Avec plus de temps : mieux choisir les contours
> d'objets en tâche 1, trancher les quatre endpoints restants en tâche 2, ajouter des familles en tâche 3. Le code
> et ces diapositives sont dans notre dépôt. Merci.

`TODO-FINAL` : les trois scores rappelés en bas de la diapositive.

---

## Si le jury pose une question

- **« Vos scores sont-ils fiables ? »** Ce sont des scores publics, calculés sur une partie des données ; le classement
  final se fait sur une partie cachée. Pour la tâche 1 nous avions en plus un contrôle local sur de vraies images
  étiquetées, pour la tâche 2 les neuf endpoints d'entraînement.
- **« Avez-vous réglé vos solutions sur le classement public ? »** Oui en partie, et nous le disons dans le README :
  en tâche 3 le classement est la seule mesure disponible, nous avons changé un seul réglage par soumission ; en tâche 2
  nous avons testé une hypothèse par soumission pour les derniers endpoints.
- **« Quel calcul ? »** Un nœud de quatre A100 sur JURECA, une RTX 3080 locale.
- **« Qui a écrit le code ? »** Les agents Claude. L'équipe humaine a fourni les accès et les décisions.
- **« Où est le détail ? »** Dans le dépôt : un README par tâche, avec le fichier et la fonction de chaque étape, les
  commandes, les résultats et ce qui n'a pas marché.
