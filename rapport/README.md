# Rapport modifiable

- **Lecture :** `rapport.pdf`.
- **Overleaf :** importer `overleaf.zip` comme projet existant, choisir `main.tex`
  et le compilateur **pdfLaTeX**. Le ZIP contient les sources et figures utiles,
  sans fichiers auxiliaires ni copie du PDF final.
- **Auteurs :** Wael AKIL, Jinyu CONG et Furkan NARIN, dans `metadata.tex`.
- **Couverture et en-têtes :** affiliation, formation et titre court dans
  `metadata.tex` ; logo institutionnel et provenance dans `assets/`.
- **Texte :** modifier les fichiers de `sections/` ; sources bibliographiques
  dans `references.tex`, tableaux dans `tables/`, figures vectorielles dans `figures/`.

Compilation locale depuis la racine du dépôt :

```bash
python rapport/build.py
```

La compilation nécessite Python et pdfLaTeX, effectue trois passes et actualise
`rapport.pdf` et `overleaf.zip`.
La mise en page est A4, 11 points, marges de 2,15 cm ; le contenu numéroté doit
rester à 10 pages maximum, références comprises. Les pages préliminaires sont
comptées séparément. Le dossier `build/` est temporaire.
La couverture, le résumé, la table des matières, la liste des huit figures et
la liste des cinq tableaux occupent chacun une page distincte. Les trois listes
ont des titres centrés, une même taille de police et un même espacement ; leurs
entrées comportent des liens internes. L'introduction et les
sections suivantes s'enchaînent sans saut de page imposé. Les pages préliminaires
sont séparées de la numérotation du corps.
Les pages du corps portent le titre court et « Master 2 MLSD » en en-tête,
avec un numéro centré en pied de page. La couverture n'est pas numérotée.
Les références occupent une page séparée, avec un titre centré.

## Mettre à jour les résultats

La version actuelle décrit `fisherDP.py` au commit `180de7c`, sans le modifier.
Le protocole principal reprend **20 départs** des K-means, graine **42**.
Les variantes d'initialisation et de standardisation sont séparées dans le texte.

Après modification du script par l'équipe :

1. Régénérer les résultats et analyses complémentaires avec
   `python rapport/tools/generate_assets.py`. Cet outil lit les fonctions et
   paramètres du script sans exécuter son interface graphique ; il écrit
   seulement dans `rapport/`.
2. Vérifier les calculs avec `python rapport/tools/validate_results.py`, puis
   actualiser les commentaires numériques dans les sections. La vérification
   couvre 270 cas exhaustifs, les entrées invalides, l'alignement, les empreintes
   des sources et la décomposition de Huygens.
3. Vérifier les figures avec `python rapport/tools/verify_figures.py` (dépendances
   du script, dont seaborn). Cette commande exécute le script sans le modifier,
   écrit ses sorties dans `build/script-figure-check/` et compare les coordonnées,
   labels, courbes, coupures, moyennes et matrices aux figures du rapport.
4. Recompiler et contrôler les chiffres, légendes, références et nombre de pages.

`resultats.json` conserve les valeurs, versions et empreintes des entrées utilisées.
`analyses-complementaires.json` conserve les profils, changements d'échelle et essais
sur dix graines. Les deux fichiers sont recalculés par l'outil de génération ;
les résultats des vérifications sont dans `build/validation.json`.
Les données, le script et les figures originales restent inchangés.

## Figures et traçabilité

Les huit figures du rapport couvrent les sept vues produites par le script :
inertie, espace et ordre des observations pour chaque jeu de données, puis
matrices de confusion sur `aiguillage`. Une comparaison supplémentaire des
labels selon l'ordre d'acquisition complète les vues temporelles.

Les figures sont régénérées en PDF vectoriel avec une mise en page homogène.
Le style conserve les couleurs et marqueurs des classes, ajoute des repères de
segmentation discrets et une échelle commune aux matrices de confusion ; les
erreurs non nulles hors diagonale sont encadrées.
Les données, paramètres et partitions restent ceux du script. L'affichage des
observations utilise la base 1, contre la base 0 du script ; la conversion
concerne également les coupures. Les couleurs sont alignées sur Fisher pour
la simulation et sur la référence pour les données réelles. Les ajustements
utilisent les variables brutes ; l'ACP centrée-réduite ne sert qu'à l'affichage.
Le rapport emploie une SVD complète déterministe pour l'ACP. Le solveur automatique
du script choisit une SVD aléatoire sur `aiguillage` : les coordonnées peuvent
différer légèrement, sans changement de partition. La vérification exige les
mêmes labels et admet uniquement pour les coordonnées ACP une tolérance absolue
de `1e-4` et relative de `1e-5` ; la variance expliquée doit coïncider à 0,1 point
près. Voir la [documentation officielle de PCA](https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.PCA.html).

`figure-provenance.json` conserve les marques effectivement tracées et les
empreintes des sources. `build/figure-validation.json` conserve le résultat de
la comparaison avec une exécution fraîche du script. Sur les données réelles,
les courbes d'inertie sont descriptives : les quatre classes sont imposées par
l'énoncé.

## Références de format

Les contraintes évaluées proviennent de `../projet.pdf`, page 2. Les choix LaTeX
suivent les guides officiels Overleaf sur le
[français](https://www.overleaf.com/learn/latex/French), les
[dimensions et marges](https://www.overleaf.com/learn/latex/Page_size_and_margins) et
l'[import d'un projet ZIP](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project).
La bibliographie cite Fisher (1958), Lloyd (1982), Ward (1963), Hubert et Arabie
(1985), la documentation scikit-learn 1.8.0 et une référence regroupant les supports
officiels d'Allou Samé. Aucun cours annoté ni fichier de notes n'est cité.

Correspondance de cette référence de cours avec les fichiers locaux :

| Support officiel | Fichier | Pages PDF utilisées |
| --- | --- | --- |
| Chapitre 1 : Introduction | `../../../Cours/chapitre1.pdf` | 31, 53–67 |
| Chapitre 2 : Classification hiérarchique | `../../../Cours/chapitre2.pdf` | 36–39 |
| Chapitre 3 : Méthodes par partitionnement | `../../../Cours/chapitre4.pdf` | 4–13, 47–49, 61–64 |

Les diapositives ne portent pas de date vérifiable : la référence indique « s. d. ».
