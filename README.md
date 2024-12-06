## Rapport PSVO-2024: NOUKON Gilberto Charbel Kayodé

## Projet de Segmentation et de Classification des maladies de feuilles de pommiers

## Objectifs du projet

L'objectif principal de ce projet est de développer une application web permettant aux utilisateurs de soumettre des images de leurs plantes pour une analyse automatique, afin d'obtenir des résultats sur l'état de santé des plantes et de segmenter les différentes parties de la plante pour une meilleure évaluation des anomalies. Suite à cette segmentation l’application doit etre capable d’émettre un diagnostic basé sur les prédictions d’un modèle de classification

## Ce que j'ai appris

Au cours de ce projet, j'ai appris à :
- Prétraiter des données d'images pour améliorer la performance du modèle.
- Organiser son code de manière à simplifier la lecture
- Utiliser des architectures simple de machine Learning tels que Kmeans pour effectuer la classification, meme si les performance laisse à désirer

## Description du projet

Le projet consiste à Utiliser des algorithmes de segmentation (comme U-Net) pour identifier et isoler les parties des plantes (feuilles, tiges, etc.) et détecter d'éventuelles anomalies. Nous avons utilisé TensorFlow pour construire un modèle de segmentation d'images à l'aide du jeu de données **leaf_disease_segmentation_dataset** qui contient 588 images de feuilles de pommiers malades et 588 masques correspondants à ces feuilles. Le modèle utilise l'architecture U-Net. Après l’entrainement, le modèle a atteint une précision de 82.68% sur l'ensemble de test prévu, démontrant ainsi son efficacité dans la segmentation des différentes feuilles de pommiers. Nous avons continué notre parcours en développant un modèle de classification en utilisant le dataset **plantvillageapplecolor** et l'algorithme **KMeans**. Notre modèle a atteint une précision de 65.75%.

## Liens Utiles
- **vidéo de démonstration :** [Lien vers la vidéo]



