# TOUNDRA
**Modèle de forêt dynamique.**

___

## Présentation
Le script présent dans ce dépôt est un outil de simulation d'évolution de forêt.<br>
Ces évolutions sont issues de modèles développés et étudiés par l'équipe VELO du LS2N à Nantes, dans 
le cadre du projet TOUNDRA.<br>


* [Plus d'informations sur le projet TOUNDRA](https://pagesperso.ls2n.fr/~cantin-g/toundra.html)
* [L'équipe VELO](https://velo.pythonanywhere.com/)
* [On the degradation of forest ecosystems by extreme events: Statistical Model Checking of a hybrid model](https://hal.science/hal-04069502v1)


#### Le modèle

Le modèle dit "à deux équations" modélise l'évolution de la densité de population d'une espèce
d'arbre et de ses graines.   
On ajoute à ce processus déterministe (EDP) un dérèglement stochastique par 
l'apparition de feux de forêt aléatoires.

___

## Utilisation 

Actuellement, l'utilisation du programme se fait en trois temps :
0. Installer les dépendances requises pour python (fichier "requirements.txt")
1. Configurer la simulation à l'aide de "toundra_config.yaml", à ouvrir avec un éditeur de texte quelconque.
2. Lancer le programme principal avec "toundra.py".


#### Configuration

**Il n'est normalement pas nécessaire de modifier toundra.py**. Le seul fichier modifié par l'utilisateur est le .yaml.  
On trouvera dans le fichier config_toundra.yaml de nombreux commentaires explicatifs. 

___

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Les programmes de ce dépôt contiennent du code généré par une intelligence artificielle.

