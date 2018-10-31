****************************************************************
*****************************READ ME****************************
****************************PROJET 6****************************
************CATEGORISER AUTOMATIQUEMENT DES QUESTIONS***********
****************************************************************


L'url de l'API est :https://tagprediction.herokuapp.com/predict_tags/

elle fonctionne en ajoutant la question en paramètre de la manière
suivante:
https://tagprediction.herokuapp.com/predict_tags?question="VOTRE QUESTION"


Dans ce dossier, vous pourrez trouvez:

Les différents scripts qui m'ont permis d'élaborer la solution : 
- final_solution.py : le script qui met en place la solution finale
- bag_of_words.py: le script qui permet à partir du dataset créé
une représentation bag of words
- tag_creation.py: le script qui permet de supprimer les tags les 
moins fréquents de notre dataset
- lda.py : le script permettant à partir de la représentation bag of 
words de générer des tags de manière non supervisé

Les deux notebooks d'explorations et d'études pour trouver une solution
à notre problème

Le rapport

Pour retrouver le code de l'api, vous pouvez consulter l'url:
https://github.com/mzackaria/ApiTaggingStackoverflowquestion

Vous pouvez également trouver la vidéo de la présentation à ce lien : http://youtu.be/AcCDlCp5CCo
