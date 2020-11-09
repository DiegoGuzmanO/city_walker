
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import geopandas as gpd
import contextily as ctx
from sklearn.cluster import SpectralClustering
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import OPTICS
from shapely.geometry import Point, Polygon

st.set_option('deprecation.showPyplotGlobalUse', False)

# Lecture du dataframe preprocessed
df_paris = pd.read_csv('df_paris.csv', sep = ';')

# Transformation du DataFrame en GeoDataFrame
crs={'init':'epsg:4326'}
gdf_paris = gpd.GeoDataFrame(
    df_paris,crs=crs, geometry=gpd.points_from_xy(df_paris.X, df_paris.Y))

#Création du SubSet Patrimoine
gdf_patrimoine=gdf_paris[ (gdf_paris['categorie']=='patrimoine')]

st.title('City Walker')
st.header("Optimisateur d'itinéraire touristique - Paris")
st.text('Danyl Delaisser, Diego Guzman')
st.text(' ')


# Définition des sites populaires auprès des touristes - PAS SUR DE GARDER CA
top_patrimoine=['Palais du Louvre','Tour Eiffel','Cathédrale Notre-Dame','Basilique du Sacré-Cœur','Orsay',
                'Galerie Raulin-Pompidou','Arc de Triomphe']
for top in top_patrimoine:
  df_paris.loc[df_paris.name==top, 'categorie']='patrimoine'

top_shopping=['Galeries Lafayette Haussmann','Beaugrenelle - Magnetic','Forum les Halles','Bercy Village'
              ,'Marché Saint-Germain' ,'Le Bon Marché', 'Le BHV Marais','Arcade des Champs-Élysées']
for top in top_shopping:
  df_paris.loc[df_paris.name==top, 'categorie']='shopping'



# Sélection des paramètres utilisateur

st.sidebar.title("Planification de l'itinéraire") 
slider_clusters = st.sidebar.slider(label='Nombre de jours à Paris?',
                            min_value=1,
                            max_value=15,
                            value=3,
                            step=1)

if slider_clusters <11 : 
    modele_optimal = 'Spectral Clustering'
else : 
    modele_optimal = 'K-Means'

st.sidebar.text('Le modèle optimal pour ce nombre ')
st.sidebar.text('de jours est ')
st.sidebar.text(modele_optimal)

slider_modele = st.sidebar.select_slider(label='Confirmez le modèle souhaité :',
                                       options=['Spectral Clustering', 'K-Means'],value= modele_optimal)

slider_nb_restos=st.sidebar.slider(label='Nombre de restaurants proposés',
                                   min_value=1,
                                   max_value=15,
                                   value=3,
                                   step=1)

selectbox_type_restaurant = st.sidebar.selectbox(label='Type de restaurants', 
                               options=['Peu importe', 'Restaurant traditionnel','Fast Food'])

colors = ['red','darkblue','darkgreen','yellow','darkorange','cyan','deeppink','steelblue','lime','silver',
      'maroon','indigo','fuchsia','darkgoldenrod','peachpuff','mediumaquamarine','whitesmoke','black']


# Specral Clustering avec n_clusters 15
spectral1 = SpectralClustering(n_clusters=15,affinity = 'nearest_neighbors').fit(gdf_patrimoine.loc[:,['X','Y']])
labelsspec1=spectral1.labels_
gdf_patrimoine_spec=gdf_patrimoine.copy()
gdf_patrimoine_spec['label']=labelsspec1


# modèle Kmeans avec n_clusters = 15
kmeans2=KMeans(n_clusters=15).fit(gdf_patrimoine.loc[:,['X','Y']])
labels2=kmeans2.labels_
gdf_patrimoine_k=gdf_patrimoine.copy()
gdf_patrimoine_k['label']=labels2

if slider_modele =='Spectral Clustering' : 
    gdf_patrimoine_final=gdf_patrimoine_spec.copy()
elif slider_modele == 'K-Means' : 
    gdf_patrimoine_final=gdf_patrimoine_k.copy()



# On intègre les 5 top de pagerank de chaque cluster dans un dataframe 
tops_per_cluster=pd.DataFrame()
tops_coord=pd.DataFrame()

for i in range(len(gdf_patrimoine_final['label'].unique())):
  X = gdf_patrimoine_final[gdf_patrimoine_final['label'] == i]['X'].to_numpy().reshape(-1, 1)
  Y = gdf_patrimoine_final[gdf_patrimoine_final['label'] == i]['Y'].to_numpy().reshape(-1, 1)

  XY = np.concatenate([X, Y], axis = 1)


## Matrice des distances
  distances = cdist(XY, XY, metric = 'euclidean')


## Scaling entre 0 et 1
  normalized_dists = MinMaxScaler().fit_transform(distances)

## On inverse les distances pour que les lieux les mieux classés
## soient les lieux les plus proches des autres
  normalized_dists = 1 - normalized_dists

## On ne veut pas qu'un point soit fortement connecté avec lui même
  normalized_dists = normalized_dists - np.eye(len(X))

## Normalise sur les lignes pour obtenir une loi de probabilité sur chaque ligne
  normalized_dists /= normalized_dists.sum(axis = 1).reshape(-1, 1)

## Application du pagerank
  G = nx.from_numpy_matrix(normalized_dists) 
  rankings = nx.pagerank(G)

## Top nodes du cluster
  top_nodes = sorted(rankings.items(), key = lambda x: x[1], reverse = True)[:5]

## Enregistrement des coordonnées des top_nodes dans un dataframe "tops_coord"
  coord=[]
  for top in top_nodes:
    coord.append(XY[top[0]])
  tops_coord[i]=coord


## Enregistrement des top nodes dans le dataframe recapitulatif
  tops_per_cluster[i]=(top_nodes)


## Calcul et enregistrement des centroids des pagerank par cluster dans un dataframe "top_centroids"

top_centroids=pd.DataFrame(index=('X','Y'))
for cluster in tops_coord.columns:
  top_centroids[cluster]=tops_coord[cluster].mean()
top_centroids=top_centroids.transpose()

# création d'un subjet avec les coordonnées des restaurants 
if selectbox_type_restaurant=='Peu importe':
    gdf_restaurants=gdf_paris[gdf_paris['categorie']=='restaurant']
elif selectbox_type_restaurant=='Restaurant traditionnel':
    gdf_restaurants=gdf_paris[(gdf_paris['categorie']=='restaurant')
                              & (gdf_paris['type']=='restaurant')]
elif selectbox_type_restaurant=='Fast Food':
    gdf_restaurants=gdf_paris[(gdf_paris['categorie']=='restaurant')
                              & (gdf_paris['type']=='fast_food')]



gdf_restaurants.reset_index(inplace=True)
Xr = gdf_restaurants['X'].to_numpy().reshape(-1, 1)
Yr = gdf_restaurants['Y'].to_numpy().reshape(-1, 1)
XYr=np.concatenate([Xr,Yr], axis = 1)

## Identification des restaurants situés le plus près des top_centroids
restos_index=[]
for cluster in top_centroids.index:
  xy=np.array([top_centroids.iloc[cluster,0],top_centroids.iloc[cluster,1]])
  restos=cdist(XYr,[xy],metric='euclidean')
  dist_df=pd.DataFrame(restos)
  liste=dist_df.sort_values(by=0).head(slider_nb_restos).index.tolist()
  restos_index.append(liste)


# Identification des zones commerciales
gdf_shopping=gdf_paris[ (gdf_paris['categorie']=='shopping')]

optics_clf = OPTICS(min_samples=15,metric='euclidean',
                    cluster_method='xi').fit(gdf_shopping.loc[:,['X','Y']])
shop_labels = optics_clf.labels_

## création des polygones à partir des clusters issues du modèle optics_clf
gdf_shopping['labels']=shop_labels
gdf_shopping=gdf_shopping[gdf_shopping['labels'] > -1]
gdf_shopping_zones=gdf_shopping.copy()

## Transformations des clusters en Polygones géographiques
gdf_shopping_zones['geometry'] = gdf_shopping_zones['geometry'].apply(lambda x: x.coords[0])

gdf_shopping_zones =gdf_shopping_zones.groupby('labels')['geometry'].apply(lambda x: Polygon(x.tolist())).reset_index()

gdf_shopping_zones = gpd.GeoDataFrame(gdf_shopping_zones, geometry = 'geometry')

## Identification des centroids de chaque polygone (ou zone commerciale)
shop_centroids=gdf_shopping_zones.centroid

## identification des zones de Shopping les plus près des top_centroids
shop_centroids=gpd.GeoDataFrame(gdf_shopping_zones.centroid,columns=['geometry'])

Xs = shop_centroids['geometry'].x.to_numpy().reshape(-1, 1)
Ys = shop_centroids['geometry'].y.to_numpy().reshape(-1, 1)
XYs=np.concatenate([Xs,Ys], axis = 1)

shop_centroids_index=[]
for cluster in top_centroids.index:
  xy=np.array([top_centroids.iloc[cluster,0],top_centroids.iloc[cluster,1]])
  shop_dist=cdist(XYs,[xy],metric='euclidean')
  shop_dist=pd.DataFrame(shop_dist)
  liste=shop_dist.sort_values(by=0).head(5).index.tolist()
  shop_centroids_index.append(liste)




# Affichage des top 5 pagerank, restaurants et zones commerciales
for clusters in range(slider_clusters):
  colors = ['blue' for i in range(120)]
  for node in tops_per_cluster.iloc[:,clusters]:
    colors[node[0]] = 'red'
  ax = gdf_patrimoine_final[gdf_patrimoine_final['label'] == clusters].plot(color = colors, alpha = 0.8)
  gdf_restaurants.loc[restos_index[clusters]].plot(color = 'green',ax=ax, alpha=0.8)
  gdf_shopping_zones.loc[shop_centroids_index[clusters]].plot(color = 'orange',ax=ax,alpha=0.6)
  jour=clusters+1
  fig = plt.gcf()
  fig.set_size_inches((9,9))
  plt.title('Jour de visite n° %i' % jour)
  plt.axis('off')
  ctx.add_basemap(ax,crs=crs,zoom="auto")
  st.pyplot()
  
