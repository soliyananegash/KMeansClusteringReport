#importing what's needed 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataFourCircle = pd.read_csv('/data/classes/2024/spring/cs425/course_files/Assignment_1_Files/fourCircles.txt', header=None) #reads in file
dataFourCircle = dataFourCircle[0].str.split(expand=True) #split each row into two columns
dataFourCircle = dataFourCircle.astype(float) #converts to float
dataFourCircle #displays file

dataIris = pd.read_csv('/data/classes/2024/spring/cs425/course_files/Assignment_1_Files/iris.txt', header=None)
dataIris = dataIris[0].str.split(expand=True)
dataIris = dataIris.astype(float)
dataIris

dataTFour = pd.read_csv('/data/classes/2024/spring/cs425/course_files/Assignment_1_Files/t4.8k.txt', header=None)
dataTFour = dataTFour[0].str.split(expand=True) #split each row into two columns
dataTFour = dataTFour.astype(float) #converts to float
dataTFour

dataTwoCircle = pd.read_csv('/data/classes/2024/spring/cs425/course_files/Assignment_1_Files/twoCircles.txt', header=None)
dataTwoCircle = dataTwoCircle[0].str.split(expand=True) #split each row into two columns
dataTwoCircle = dataTwoCircle.astype(float) #converts to float
dataTwoCircle

dataTwoEllipse = pd.read_csv('/data/classes/2024/spring/cs425/course_files/Assignment_1_Files/twoEllipses.txt', header=None)
dataTwoEllipse = dataTwoEllipse[0].str.split(expand=True) #split each row into two columns
dataTwoEllipse = dataTwoEllipse.astype(float) #converts to float
dataTwoEllipse

#finding out the how many clusters needed for fourCircles.txt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = dataFourCircle.values  #putting values from the dataset into X
X_scaled = StandardScaler().fit_transform(X) #scaling the data

sse = [] #stores SSE vals for each k (difference btwn the observed and predicted values)
#loop over every potential cluster num
for k in range(1,11):
    kmeans = KMeans(n_clusters=k) #initializes amount of clusters and centroids
    kmeans.fit(X_scaled) #fits this model to our dataset
    sse.append(kmeans.inertia_) #the math doing the work
    
#visualize
plt.plot(range(1,11), sse) #plots points
plt.xticks(range(1,11)) #sets range on x-axis
plt.xlabel('Num of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Best Num of Clusters for fourCircles.txt')
plt.show() #appears to be a bend at x=4 so will use 4 clusters

#finding out the how many clusters needed for t4.8k.txt
X = dataTFour.values  #putting values from the dataset into X
X_scaled = StandardScaler().fit_transform(X) #scaling the data

sse = [] #stores SSE vals for each k (difference btwn the observed and predicted values)
#loop over every potential cluster num
for k in range(1,11):
    kmeans = KMeans(n_clusters=k) #initializes amount of clusters and centroids
    kmeans.fit(X_scaled) #fits this model to our dataset
    sse.append(kmeans.inertia_) #the math doing the work
    
#visualize
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('Num of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Best Num of Clusters for t4.8k.txt')
plt.show() #appears to be a bend at x=3 so will use 3 clusters

#finding out the how many clusters needed for twoCircles.txt
X = dataTwoCircle.values  #putting values from the dataset into X
X_scaled = StandardScaler().fit_transform(X) #scaling the data

sse = [] #stores SSE vals for each k (difference btwn the observed and predicted values)
#loop over every potential cluster num
for k in range(1,11):
    kmeans = KMeans(n_clusters=k) #initializes amount of clusters and centroids
    kmeans.fit(X_scaled) #fits this model to our dataset
    sse.append(kmeans.inertia_) #the math doing the work
    
#visualize
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('Num of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Best Num of Clusters for twoCircles.txt')
plt.show() #appears to be a bend at x=4 so will use 4 clusters

#finding out the how many clusters needed for twoEllipses.txt
X = dataTwoEllipse.values  #putting values from the dataset into X
X_scaled = StandardScaler().fit_transform(X) #scaling the data

sse = [] #stores SSE vals for each k (difference btwn the observed and predicted values)
#loop over every potential cluster num
for k in range(1,11):
    kmeans = KMeans(n_clusters=k) #initializes amount of clusters and centroids
    kmeans.fit(X_scaled) #fits this model to our dataset
    sse.append(kmeans.inertia_) #the math doing the work
    
#visualize
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('Num of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Best Num of Clusters for twoEllipses.txt')
plt.show() #appears to be a bend at x=4 so will use 4 clusters

def kmeans(dataset, clusterAmt):
    max_i = 20  #max number of iterations if it never converges
    epsilon = 0.001  #convergence threshold
    i = 0  #actual iterations happening
    mean = [] #array to hold mean values
    size = [] #array to hold data points
    
    centroids = dataset[np.random.choice(len(dataset), clusterAmt, replace=False)] #randomly selecting cluster centers based on dataset length and cluster amount asked
    
    #iteration begins
    while i < max_i: #while current iteration is < 20
        clusterDistance = np.sqrt(np.sum((dataset[:, np.newaxis] - centroids) ** 2, axis=2)) #calculates euclidean distance
        labels = np.argmin(clusterDistance, axis=1) #assigns the clusters to each data point
        
        newCentroids = [] #to store the new centroids when updating
        
        for j in range(clusterAmt): #looping through clusters
            dataPts = dataset[labels == j] #setting the data points in a cluster
            calculateCentroid = np.mean(dataPts, axis=0) #calculates new centroid in cluster
            newCentroids.append(calculateCentroid) #adds new centroids to newCentroids list declared above
            
        newCentroids = np.array(newCentroids) #converts from list to numpy array
        
        delta = np.sum((centroids - newCentroids)**2) #comparing old and new means
        if delta <= epsilon: #convergence testing
            break #stop if converged
        
        centroids = newCentroids #updates the centroids
        i += 1 #increment i
    
    for j in range(clusterAmt): #going through each cluster
        mean.append(np.mean(dataset[labels == j], axis=0))  #calculates the mean of each cluster
        size.append(np.sum(labels == j))  #calculates the cluster size of each cluster
        
    #printing out all relevant values
    print("Final mean and size for each cluster:")
    for i in range(clusterAmt):
        print(f"Cluster {i + 1}: Mean = {mean[i]}, Size (amount of data points) = {size[i]}")
    print(f"Final Cluster Assignments: {labels}")
    print(f"Number of Iterations: {i}")
    print(f"Final Delta: {delta}")
    
    #returns the cluster assignments and centroids
    return labels, centroids

#test to see if kMeans fxn works (it does!!)
kmeans(dataFourCircle.values,4)
kmeans(dataTwoCircle.values,4)
kmeans(dataTFour.values,3)
kmeans(dataTwoEllipse.values,4)

def makeGraphs(dataset, finalClusterAssignment, centroids):
    for i in np.unique(finalClusterAssignment): #going through all the centroids
        cluster = dataset[finalClusterAssignment == i] #looking through data for current cluster
        plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], label=f'Cluster {i + 1}') #actually plots points from above
    
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids') #plotting the centroids

    #graph design
    plt.title('Cluster Graph')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

labels, centroids = kmeans(dataFourCircle.values,4) #pulls  cluster assignments and centroids from kMeans fxn and applies it for graph making
makeGraphs(dataFourCircle, labels, centroids)

labels, centroids = kmeans(dataTwoCircle.values,4) 
makeGraphs(dataTwoCircle, labels, centroids)

labels, centroids = kmeans(dataTFour.values,3) 
makeGraphs(dataTFour, labels, centroids)

labels, centroids = kmeans(dataTwoEllipse.values,4) 
makeGraphs(dataTwoEllipse, labels, centroids)

labels, centroids = kmeans(dataIris.values,3)
makeGraphs(dataIris,labels,centroids)
