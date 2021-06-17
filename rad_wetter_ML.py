# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:56:13 2021

@author: katha
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr


df_radwetter = pd.read_pickle('radwetter')
sns.distplot(df_radwetter.gesamt) # 

df_radwetter['UV'] = df_radwetter['UV'].astype('float')

# for categorical variables
df_radwetter['Werktag'] = df_radwetter['Werktag'].astype('category')
df_radwetter['Werktag'].cat.categories = [0,1]
df_radwetter['Werktag'] = df_radwetter['Werktag'].astype('float')

# classify gesamt 
def transform_gesamt(x):
    x = float(x)
    if x == 0:
        return 0
    elif (x > 0) & (x < 50):
        return 1
    elif (x >= 50) & (x < 100) :
        return 2
    elif x >= 100:
        return 3
df_radwetter['cgesamt'] = df_radwetter['gesamt'].apply(lambda x: transform_gesamt(x))


# ------------------------------------------------------------------------
# dimensionality reduction using multi-correlations
# ------------------------------------------------------------------------

df_radwetter_corr = abs(df_radwetter.corr(method = 'kendall'))
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_radwetter_corr, vmin=0, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

df_radwetter_reduced = df_radwetter.loc[df_radwetter['Tageslicht'] == 1]
del df_radwetter_reduced['Windböen']
del df_radwetter_reduced['Tageslicht']
del df_radwetter_reduced['Windchill']
del df_radwetter_reduced['UV']
del df_radwetter_reduced['Taupunkt']
del df_radwetter_reduced['Solarstrahlung']
del df_radwetter_reduced['Luftfeuchte']
df_radwetter_multicoll = df_radwetter_reduced
df_radwetter_corr = abs(df_radwetter_reduced.corr(method = 'kendall'))
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_radwetter_corr, vmin=0, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

# ------------------------------------------------------------------------
# dimensionality reduction fpr only continuous variables: pca
# ------------------------------------------------------------------------
#https://stats.stackexchange.com/questions/27300/using-principal-component-analysis-pca-for-feature-selection/27310#27310
#https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006907

# Select data due to meaningfulness
df_radwetter_reduced = df_radwetter.loc[df_radwetter['Tageslicht'] == 1]

# select data because of memory problems
train = df_radwetter_reduced[msk].reset_index(drop = True)
test = df_radwetter_reduced[~msk]

# select only continuous/ordinal variables
df = train[['Temperatur', 'Luftfeuchte', 'Luftdruck', 'Regen',
 'Sonnenschein','Taupunkt', 'Windchill', 'Windböen', 'Windgeschwindigkeit', 'Solarstrahlung','UV',
      'Werktag']]


X_features =  df#df.iloc[:,:-1]
y_label = df_radwetter.cgesamt
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)
#sns.distplot(X_features[:,1]) # cube transform
# pca
n_components = 2
pca = PCA(n_components = n_components)
reduced = pca.fit_transform(df)
pca_variance = pca.explained_variance_
eigenvalues=pca.components_# eigenvalues

from mlxtend.plotting import plot_pca_correlation_graph
figure, correlation_matrix = plot_pca_correlation_graph(X_features, 
                                                        list(df.columns),
                                                        X_pca = reduced,
                                                        explained_variance = pca_variance,
                                                        dimensions=(1, 2),
                                                        figure_axis_size=10)    

# scree plot
plt.figure()
plt.bar(range(len(pca_variance)), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

# scatter plot
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c = train.cgesamt)
plt.show()

# controbution of each variable to component: eigenvalues
for c in range(len(eigenvalues)):
    plt.figure()
    plt.bar(range(len(eigenvalues[c])), eigenvalues[c], alpha=0.5, align='center', label='variable controbition')
    plt.legend(['Component ' + str(c +1 )])
    plt.ylabel('Eigenvalue')
    plt.xlabel('Variable')
    plt.xticks(ticks = range(len(df.columns)), labels = list(df.columns), rotation=90)
    plt.show()

# for multicoll results

# select data because of memory problems
msk = np.random.rand(len(df_radwetter_multicoll)) < (30000/len(df_radwetter_multicoll))
train = df_radwetter_multicoll[msk].reset_index(drop = True)
test = df_radwetter_multicoll[~msk]
df = train
X_features =  train[['Temperatur', 'Luftdruck', 'Regen', 'Sonnenschein',
       'Windgeschwindigkeit', 'gesamt', 'Werktag']]
y_label = df_radwetter_multicoll.cgesamt
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)
#sns.distplot(X_features[:,1]) # cube transform
# pca
n_components = 2
pca = PCA(n_components = n_components)
reduced = pca.fit_transform(X_features)
pca_variance = pca.explained_variance_
eigenvalues=pca.components_# eigenvalues

from mlxtend.plotting import plot_pca_correlation_graph
figure, correlation_matrix = plot_pca_correlation_graph(X_features, 
                                                        list(X_features().columns),
                                                        X_pca = reduced,
                                                        explained_variance = pca_variance,
                                                        dimensions=(1, 2),
                                                        figure_axis_size=10)    

# scree plot 
plt.figure()
plt.bar(range(len(pca_variance)), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()



# -----------------------------------------------------------------------
# clustering
# -----------------------------------------------------------------------

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.get_dummies(train[['Temperatur', 'Luftdruck', 'Regen',
       'Sonnenschein', 'Windgeschwindigkeit', 'Werktag', 'gesamt']])
X_features =  df#df.iloc[:,:-1]
y_label = train.cgesamt
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

kmeans = KMeans(init="random", 
                n_clusters=4, 
                n_init=10 ,
                max_iter=300, 
                random_state=42)
clusters = kmeans.fit(X_features)
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(train.cgesamt)
clusters.labels_
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
preprocessor = Pipeline([("scaler", MinMaxScaler()),
                         ("pca", PCA(n_components=2, random_state=42)),])
n_clusters = 4
clusterer = Pipeline([("kmeans", 
                       KMeans(n_clusters=n_clusters,
                              init="k-means++",n_init=50,
                              max_iter=500,random_state=42,),), ] )
pipe = Pipeline(  [("preprocessor", preprocessor),("clusterer", clusterer)])
pipe.fit(X_features)
predicted_labels = pipe["clusterer"]["kmeans"].labels_



pcadf = pd.DataFrame(
     pipe["preprocessor"].transform(X_features),
     columns=["component_1", "component_2"],
 )
pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)


plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(
 "component_1",
"component_2",
 s=50,
data=pcadf,
 hue="predicted_cluster",
 style="true_label",
palette="Set2", )

scat.set_title(
    "Clustering results from TCGA Pan-Cancer\nGene Expression Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()



sns.scatterplot(train.Temperatur, train.gesamt, s=50,
 hue=predicted_labels,
palette="Set2", )

# svc
