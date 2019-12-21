#modification du dossier par défaut
# import os
# os.chdir("...")


### Fichier de données Importation, statistiques descriptives et graphiques

#importation des données
import pandas
fromage = pandas.read_table("fromage.txt",sep="\t",header=0,index_col=0)

#dimension des données
print(fromage.shape)
print("\n")

#statistiques descriptives
print(fromage.describe())
print("\n")

#graphique - croisement deux à deux des variables
import matplotlib
from pandas.plotting import scatter_matrix
scatter_matrix(fromage,figsize=(9,9))




### CAH

#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#générer la matrice des liens
Z = linkage(fromage,method='ward',metric='euclidean')
#plt.show()

#affichage du dendrogramme
plt.figure(2)
plt.title("CAH")
dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=0)
#plt.show()




### Classification ascendante hiérarchique
plt.figure(3)
plt.plot([1, 2, 3])
#matérialisation des 4 classes (hauteur t = 7)
plt.title('CAH avec matérialisation des 4 classes')
dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=7)
plt.show()




### Classification ascendante hiérarchique Découpage en classes – Matérialisation des groupes

#découpage à la hauteur t = 7 ==> identifiants de 4 groupes obtenus
groupes_cah = fcluster(Z,t=7,criterion='distance')
print(groupes_cah)
print("\n")

#index triés des groupes
import numpy as np
idg = np.argsort(groupes_cah)

#affichage des observations et leurs groupes
print(pandas.DataFrame(fromage.index[idg],groupes_cah[idg]))
print("\n")




### K-MEANS

#k-means sur les données centrées et réduites
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(fromage)

#index triés des groupes
idk = np.argsort(kmeans.labels_)

#affichage des observations et leurs groupes
print(pandas.DataFrame(fromage.index[idk],kmeans.labels_[idk]))
print("\n")

#distances aux centres de classes des observations
print(kmeans.transform(fromage))
print("\n")

#correspondance avec les groupes de la CAH
pandas.crosstab(groupes_cah,kmeans.labels_)




### Méthode des centres mobiles Aide à la détection du nombre adéquat de groupes

 #librairie pour évaluation des partitions
from sklearn import metrics

#utilisation de la métrique "silhouette"
# #faire varier le nombre de clusters de 2 à 10
res = np.arange(9,dtype="double")
for k in np.arange(9):
    km = cluster.KMeans(n_clusters=k+2)
    km.fit(fromage)
    res[k] = metrics.silhouette_score(fromage,km.labels_)
print(res)
print("\n")

#graphique
import matplotlib.pyplot as plt
plt.figure(4)
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,11,1),res)
plt.show()




#  INTERPRÉTATION DES CLASSES

#moyenne par variable
m = fromage.mean()
#TSS
TSS = fromage.shape[0]*fromage.var(ddof=0) 
print(TSS)
print("\n")

#data.frame conditionnellement aux groupes 
gb = fromage.groupby(kmeans.labels_)
#effectifs conditionnels
nk = gb.size()
print(nk)
print("\n")

#moyennes conditionnelles
mk = gb.mean()
print(mk)
print("\n")

#pour chaque groupe écart à la moyenne par variable
EMk = (mk-m)**2
#pondéré par les effectifs du groupe
EM = EMk.multiply(nk,axis=0)
#somme des valeurs => BSS
BSS = np.sum(EM,axis=0)
print(BSS)
print("\n")
#carré du rapport de corrélation
#variance expliquée par l'appartenance aux groupes #pour chaque variable
R2 = BSS/TSS
print(R2)
print("\n")




'''
### Style

# plt.plot(x, y, c='red', lw=5, ls='--')


### figure
x=2
y=4
plt.figure(4)
plt.plot(x, y, label='quadratique') #pour creer une legende
plt.plot(x, x**3, label='cubique')
plt.title('figure 1')
plt.xlabel('axe x')
plt.ylabel('axe y')
plt.legend()
plt.show()

### subplot

# plt.subplot(221)

# # equivalent but more general
# ax1=plt.subplot(2, 2, 1)

# # add a subplot with no frame
# ax2=plt.subplot(222, frameon=False)

# # add a polar subplot
# plt.subplot(223, projection='polar')

# # add a red subplot that shares the x-axis with ax1
# plt.subplot(224, sharex=ax1, facecolor='red')

# # delete ax2 from the figure
# plt.delaxes(ax2)

# # add ax2 to the figure again
# plt.subplot(ax2)

# plt.show()
'''