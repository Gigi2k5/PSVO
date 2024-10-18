## Rapport PSVO-2024: NOUKON Gilberto Charbel Kayodé

## Projet de Segmentation et de Classification des maladies de feuilles de pommiers

## Objectifs du projet

L'objectif principal de ce projet est de développer une application web permettant aux utilisateurs de soumettre des images de leurs plantes pour une analyse automatique, afin d'obtenir des résultats sur l'état de santé des plantes et de segmenter les différentes parties de la plante pour une meilleure évaluation des anomalies. Suite à cette segmentation l’application doit etre capable d’émettre un diagnostic basé sur les prédictions d’un modèle de classification

## Ce que j'ai appris

Au cours de ce projet, j'ai appris à :
Prétraiter des données d'images pour améliorer la performance du modèle.
Utiliser des architectures de réseaux de neurones convolutionnels (CNN) pour la classification d'images.
Mettre en place un modèle de segmentation basé sur l’architecture U-Net
Évaluer la performance des modèles à l'aide de métriques appropriées telles que l'accuracy, le dice coefficient et l’iOU.
Implémenter des techniques de régularisation pour éviter le surapprentissage.

## Description du projet

Le projet consiste à Utiliser des algorithmes de segmentation (comme U-Net) pour identifier et isoler les parties des plantes (feuilles, tiges, etc.) et détecter d'éventuelles anomalies. Nous avons utilisé TensorFlow pour construire un modèle de segmentation d'images à l'aide du jeu de données **leaf_disease_segmentation_dataset** qui contient 588 images de feuilles de pommiers malades et 588 masques correspondants à ces feuilles. Le modèle utilise l'architecture U-Net. Après l’entrainement, le modèle a atteint une précision de 83.338% sur l'ensemble de test prévu, démontrant ainsi son efficacité dans la segmentation des différentes feuilles de pommiers.

## Liens Utiles
- **Notebook Colab :** [Lien vers le notebook](https://www.kaggle.com/code/charbelnoukon/notebookb9738a1d8c/edit/run/201647076)


