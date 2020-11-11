
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
from shapely.geometry import LineString, Polygon

st.set_option('deprecation.showPyplotGlobalUse', False)

# Lecture du dataframe preprocessed
df_paris = pd.read_csv('df_paris.csv', sep = ';')

# Transformation du DataFrame en GeoDataFrame
crs={'init':'epsg:4326'}
gdf_paris = gpd.GeoDataFrame(
    df_paris,crs=crs, geometry=gpd.points_from_xy(df_paris.X, df_paris.Y))

#Création du SubSet Patrimoine
gdf_patrimoine=gdf_paris[ (gdf_paris['categorie']=='patrimoine')]

st.image('page_de_garde.png', caption=None, width=None, 
         use_column_width=False, clamp=False, channels='RGB', output_format='auto')



#######################################################################
# Sélection des paramètres utilisateur

## Nombre de jours
st.sidebar.title("Planification de l'itinéraire") 
slider_clusters = st.sidebar.slider(label='Nombre de jours à Paris?',
                            min_value=1,
                            max_value=15,
                            value=3,
                            step=1)

## Modèle du classificateur
if slider_clusters <11 : 
    modele_optimal = 'Spectral Clustering'
else : 
    modele_optimal = 'K-Means'

st.sidebar.text('Modèle recommandé')
st.sidebar.text(modele_optimal)
slider_modele = st.sidebar.select_slider(label='Confirmez le modèle souhaité :',
                                       options=['Spectral Clustering', 'K-Means'],value= modele_optimal)

## Nombre de restaurants à proposer
slider_nb_restos=st.sidebar.slider(label='Nombre de restaurants proposés',
                                   min_value=1,
                                   max_value=15,
                                   value=3,
                                   step=1)
##Type de restaurants à afficher
selectbox_type_restaurant = st.sidebar.selectbox(label='Type de restaurants', 
                               options=['Peu importe', 'Restaurant traditionnel','Fast Food'])

## Affichage des étiquettes des points prioritaires
selectbox_labels= st.sidebar.selectbox(label='Afficher les noms des lieux',
                                       options=['Non','Oui'])

#######################################################################
#Boucle de sélection du modèle de clusering

if modele_optimal == 'Spectral Clustering':
    # Specral Clustering avec n_clusters 15
    spectral1 = SpectralClustering(n_clusters=15,affinity = 'nearest_neighbors').fit(gdf_patrimoine.loc[:,['X','Y']])
    labelsspec1=spectral1.labels_
    gdf_patrimoine['label']=labelsspec1
elif modele_optimal == 'K-Means':
    # modèle Kmeans avec n_clusters = 15
    kmeans=KMeans(n_clusters=15).fit(gdf_patrimoine.loc[:,['X','Y']])
    labels_k=kmeans.labels_
    gdf_patrimoine['label']=labels_k
    

#######################################################################
# Identification des points prioritaires PageRank par cluster 
tops_per_cluster=pd.DataFrame()
tops_coord=pd.DataFrame()

for i in range(len(gdf_patrimoine['label'].unique())):
  X = gdf_patrimoine[gdf_patrimoine['label'] == i]['X'].to_numpy().reshape(-1, 1)
  Y = gdf_patrimoine[gdf_patrimoine['label'] == i]['Y'].to_numpy().reshape(-1, 1)

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

#######################################################################
#Traitement des restaurants

## création d'un subset avec les coordonnées des restaurants 
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

#######################################################################
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
  liste=shop_dist.sort_values(by=0).head(3).index.tolist()
  shop_centroids_index.append(liste)



#######################################################################
# Affichage des top 5 pagerank, restaurants et zones commerciales
for clusters in range(slider_clusters):
    line_coords=[]
    colors = ['blue' for i in range(120)]
    for node in tops_per_cluster.iloc[:,clusters]:
        colors[node[0]] = 'red'
        line_coords.append([gdf_patrimoine[gdf_patrimoine['label'] == clusters].iloc[node[0],1],
                            gdf_patrimoine[gdf_patrimoine['label'] == clusters].iloc[node[0],2]])
        
    ax = gdf_patrimoine[gdf_patrimoine['label'] == clusters].plot(color = colors, alpha = 0.8)
    gdf_restaurants.loc[restos_index[clusters]].plot(color = 'green',ax=ax, alpha=0.8)
    gdf_shopping_zones.loc[shop_centroids_index[clusters]].plot(color = 'orange',ax=ax,alpha=0.6)
    
    # Optimisation du trajet entre points prioritaires
    ##Calcule des distances
    distances = cdist(line_coords, line_coords, metric = 'euclidean')
    ## Scaling entre 0 et 1
    normalized_dists = MinMaxScaler().fit_transform(distances)
    ## On ne veut pas qu'un point soit fortement connecté avec lui même
    normalized_dists = normalized_dists - np.eye(len(line_coords))
    ## Application du Minimum Spanning Tree pour identifier le trajet optimal
    G = nx.from_numpy_matrix(normalized_dists) 
    mst=nx.tree.minimum_spanning_tree(G)
    order=[*mst.edges]
    ##Tracé du parcours optimal
    for edge in range(len(order)):
        line=LineString([(line_coords[order[edge][0]][0],line_coords[order[edge][0]][1]),
                               (line_coords[order[edge][1]][0],line_coords[order[edge][1]][1])])
        x,y=line.coords.xy
        ax.plot(x,y,color='r',alpha=0.8,ls='-.')
    
    ## Annotation des noms des points prioritaires
    if selectbox_labels == 'Oui':
        for node in tops_per_cluster.iloc[:,clusters]:
            ax.annotate(gdf_patrimoine[gdf_patrimoine['label'] == clusters].iloc[node[0],5],
                        xy=(gdf_patrimoine[gdf_patrimoine['label'] == clusters].iloc[node[0],1],
                            gdf_patrimoine[gdf_patrimoine['label'] == clusters].iloc[node[0],2]), 
                        xytext=(3,3), textcoords="offset points")
  
  
    jour=clusters+1
    fig = plt.gcf()
    fig.set_size_inches((9,9))
    plt.title('Jour de visite n° %i' % jour)
    plt.axis('off')
    ctx.add_basemap(ax,crs=crs,zoom="auto")
    st.pyplot()
  