import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.manifold import Isomap
# Visualize the embedding vectors
from matplotlib import offsetbox
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import manifold
from time import time
from sklearn.manifold import TSNE
from ast import literal_eval
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score 
#MNIST Dataset was used and it was downloaded from Kaggle.com
#It can be found here ----> https://drive.google.com/drive/folders/1qsfntkfwAH3xMtu_eIdMtW7NRoTbBdm4?usp=sharing



df_origin = pd.read_csv("mnist_train.csv", header = None, nrows=5000)

print(df_origin)


df_labels = df_origin.iloc[1:, 0]
df_images = df_origin.iloc[1:, 1:]

#print(df_labels)
#print(df_images)



labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

X_train, X_test, y_train, y_test = train_test_split(df_images, df_labels, test_size=0.2)


X_train_standardized = StandardScaler().fit_transform(X_train)
print("X_train_standardized", X_train_standardized)

X_test_standardized = StandardScaler().fit_transform(X_test)
print("X_test_standardized" , X_test_standardized)

X_train_normalized = Normalizer().fit_transform(X_train_standardized)
print("X_train_normalized", X_train_normalized)

X_test_normalized = Normalizer().fit_transform(X_test_standardized)
print("X_test_normalized", X_test_normalized)

X_train_tsne = TSNE(n_components=2).fit_transform(X_train_normalized)

print("t_SNE:", X_train_tsne)

colors = ['rgb(0,31,63)', 'rgb(255,133,27)', 'rgb(255,65,54)', 'rgb(0,116,217)', 'rgb(133,20,75)', 'rgb(57,204,204)',
'rgb(240,18,190)', 'rgb(46,204,64)', 'rgb(1,255,112)', 'rgb(255,220,0)',
'rgb(76,114,176)', 'rgb(85,168,104)', 'rgb(129,114,178)', 'rgb(100,181,205)']


def plot_embedding_v1(X_embeded, y):
    plt.rcParams["figure.figsize"] = [21, 18]
    for k, i in enumerate(np.unique(y.astype(np.int))):
        plt.scatter(X_embeded[y == i, 0],
                   X_embeded[y == i, 1],
                   color = '#%02x%02x%02x' % literal_eval(colors[k][3:]), 
                    label = labels[k])
    plt.legend()
    plt.show()

plot_embedding_v1(X_train_tsne, y_train)

def plot_embedding_v2(X, X_origin, title=None, dims=[None, 28, 28]):
    dims[0] = X.shape[0]
    X_origin = X_origin.values.astype(np.float).reshape(dims)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y_train.values[i]),
                 color=plt.cm.Set1(y_train.values.astype(np.int)[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 3e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_origin[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.show()

plot_embedding_v2(X_train_tsne, X_train, "t-SNE embedding of the digits)")

X_train_normalized = pd.DataFrame(X_train_normalized) 
print("X_train_normalized", X_train_normalized)


print(" Isomap embedding ")
n_neighbors = 30
t0 = time()
X_iso = Isomap(n_neighbors, n_components=2).fit_transform(X_train_normalized)
print("Done.")
plot_embedding_v2(X_iso, X_train,"Isomap (time %.2fs)" % (time() - t0))

print(X_iso)


print(" LLE ")
n_neighbors = 30
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
t0 = time()
X_lle = clf.fit_transform(X_train_normalized)
plot_embedding_v2(X_lle, X_train, "LLE (time %.2fs)" % (time() - t0))


### CLUSTERING ###

X_iso = pd.DataFrame(X_iso) 
print(X_iso)

print(X_iso.columns)



# Building the clustering model 
spectral_model = SpectralClustering(n_clusters = 20, affinity ='nearest_neighbors')   
# Training the model and Storing the predicted cluster labels 
labels_sp = spectral_model.fit_predict(X_iso)

plt.scatter(X_iso.iloc[:,0] , X_iso.iloc[:,1], c=labels_sp, cmap = 'rainbow')
plt.show()




X_iso = pd.DataFrame(X_iso)
df_labels = df_origin.iloc[1:, 0].values


print(df_labels)
print(df_labels[0])


#X_iso["cluster"] = labels_sp
N_CLUSTERS = 20
clusters = [X_iso[labels_sp == i] for i in range(N_CLUSTERS)]

print (type(clusters))

for c in clusters:
	zero, one, two, three, four, five, six, seven, eight, nine = 0,0,0,0,0,0,0,0,0,0
	index = c.index
	print(index)
	for i in index:
		if df_labels[i] == 0:
			zero += 1
		elif df_labels[i] == 1:
			one += 1
		elif df_labels[i] == 2:
			two += 1
		elif df_labels[i] == 3:
			three += 1
		elif df_labels[i] == 4:
			four += 1
		elif df_labels[i] == 5:
			five += 1
		elif df_labels[i] == 6:
			six += 1
		elif df_labels[i] == 7:
			seven += 1
		elif df_labels[i] == 8:
			eight += 1
		elif df_labels[i] == 9:
			nine += 1
	
	print ('zero: %s one: %s two: %s three: %s four: %s five: %s six: %s seven: %s eight: %s nine: %s'%(zero,one,two,three,four,five,six,seven,eight,nine))
 


for i, c in enumerate(clusters):
    print('Cluster {} has {} members: {}...'.format(i, len(c), c))



# Building the clustering model 
spectral_model = SpectralClustering(n_clusters = 20, affinity ='rbf')   
# Training the model and Storing the predicted cluster labels 
labels_rbf = spectral_model.fit_predict(X_iso)

plt.scatter(X_iso.iloc[:,0] , X_iso.iloc[:,1], c=labels_rbf, cmap = 'rainbow')
plt.show()

X_iso = pd.DataFrame(X_iso) 

N_CLUSTERS = 20
clusters = [X_iso[labels_rbf == i] for i in range(N_CLUSTERS)] 

for c in clusters:
	zero, one, two, three, four, five, six, seven, eight, nine = 0,0,0,0,0,0,0,0,0,0
	index = c.index
	print(index)
	for i in index:
		if df_labels[i] == 0:
			zero += 1
		elif df_labels[i] == 1:
			one += 1
		elif df_labels[i] == 2:
			two += 1
		elif df_labels[i] == 3:
			three += 1
		elif df_labels[i] == 4:
			four += 1
		elif df_labels[i] == 5:
			five += 1
		elif df_labels[i] == 6:
			six += 1
		elif df_labels[i] == 7:
			seven += 1
		elif df_labels[i] == 8:
			eight += 1
		elif df_labels[i] == 9:
			nine += 1
	
	print ('zero: %s one: %s two: %s three: %s four: %s five: %s six: %s seven: %s eight: %s nine: %s'%(zero,one,two,three,four,five,six,seven,eight,nine))
 


for i, c in enumerate(clusters):
    print('Cluster {} has {} members: {}...'.format(i, len(c), c[0]))


# List of different values of affinity 
affinity = ['rbf', 'nearest-neighbours']  
# List of Silhouette Scores 
s_scores = []   
# Evaluating the performance 
s_scores.append(silhouette_score(X_iso, labels_rbf)) 
s_scores.append(silhouette_score(X_iso, labels_sp)) 
  
print(s_scores) 


# Plotting a Bar Graph to compare the models 
plt.bar(affinity, s_scores) 
plt.xlabel('Affinity') 
plt.ylabel('Silhouette Score') 
plt.title('Comparison of different Clustering Models') 
plt.show() 

