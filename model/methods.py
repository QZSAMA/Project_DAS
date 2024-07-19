# methods
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os
from sklearn.datasets import load_iris, load_diabetes

from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors

#E1


def apply_em(X, y, n_clusters, dataset_name,random_seed):
    # Apply Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_seed)
    y_pred = gmm.fit_predict(X)
    
    # Evaluate the clustering
    ari = adjusted_rand_score(y, y_pred)
    silhouette_avg = silhouette_score(X, y_pred)
    
    print(f"EM on {dataset_name}:")
    print(f"Adjusted Rand Index: {ari:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.2f}\n")
    
    return y_pred,ari,silhouette_avg

def apply_kmeans(X, y, n_clusters, dataset_name,random_seed):
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed,n_init='auto')
    y_pred = kmeans.fit_predict(X)
    
    # Evaluate the clustering
    ari = adjusted_rand_score(y, y_pred)
    silhouette_avg = silhouette_score(X, y_pred)
    
    print(f"K-Means on {dataset_name}:")
    print(f"Adjusted Rand Index: {ari:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.2f}\n")
    
    return y_pred,ari,silhouette_avg

def loan_datasets():
    iris = load_iris()    
    diabetes = load_diabetes()
    # Standardize the datasets
    scaler = StandardScaler()
    X_iris = scaler.fit_transform(iris.data)
    X_diabetes = scaler.fit_transform(diabetes.data)
    return   X_iris,iris.target,X_diabetes, diabetes.target
def plot_clusters_e1(X, y_pred, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar()
    # plt.legend()
    if not os.path.exists('E1'):
        os.mkdir('E1')
    plt.savefig('E1\\'+title+'.png')
    return
def e1(x,y,n,ds_name,random_seed):
    # Apply EM on digits dataset
    y_pred_digits_em,_,_ = apply_em(x, y, n, ds_name,random_seed)
    # Apply K-Means on digits dataset
    y_pred_digits_kmeans,_,_ = apply_kmeans(x, y, n,ds_name,random_seed)
    # Visualize the results for digits dataset
    plot_clusters_e1(x, y_pred_digits_em, "EM Clustering on "+ds_name+" Dataset")
    plot_clusters_e1(x, y_pred_digits_kmeans, "K-Means Clustering on "+ds_name+" Dataset")

#E2
def apply_randomized_projections(X, n_components, dataset_name,seed=0):
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    X_rp = rp.fit_transform(X)
    
    print(f"Randomized Projections on {dataset_name}:")
    print(f"Explained Variance Score: {explained_variance_score(X, rp.inverse_transform(X_rp)):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(X, rp.inverse_transform(X_rp)):.2f}\n")
    
    return X_rp


def apply_pca(X, n_components, dataset_name,seed=0):
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA on {dataset_name}:")
    print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(X, pca.inverse_transform(X_pca)):.2f}\n")
    
    return X_pca

def apply_ica(X, n_components, dataset_name,seed=0):
    ica = FastICA(n_components=n_components, random_state=seed)
    X_ica = ica.fit_transform(X)
    
    print(f"ICA on {dataset_name}:")
    print(f"Explained Variance Score: {explained_variance_score(X, ica.inverse_transform(X_ica)):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(X, ica.inverse_transform(X_ica)):.2f}\n")
    
    return X_ica

def apply_lle(X, n_components, dataset_name, seed=0):
    lle = LocallyLinearEmbedding(n_neighbors=30, n_components=n_components, random_state=seed)
    X_lle = lle.fit_transform(X)
    
    # 计算重构误差
    nbrs = NearestNeighbors(n_neighbors=30).fit(X_lle)
    distances, indices = nbrs.kneighbors(X_lle)
    
    X_reconstructed = np.zeros_like(X)
    for i in range(len(X)):
        weights = np.exp(-distances[i] ** 2)
        weights /= np.sum(weights)
        X_reconstructed[i] = np.dot(weights, X[indices[i]])
    
    reconstruction_error = np.mean(np.linalg.norm(X - X_reconstructed, axis=1))
    
    print(f"LLE on {dataset_name}:")
    print(f"Reconstruction Error: {reconstruction_error:.2f}\n")
    # print(f"Explained Variance Ratio: {np.sum(lle.explained_variance_ratio_):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(X, X_reconstructed):.2f}\n")
    
    return X_lle

def plot_2d_reduction_e2(X, y, title):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter)
    # plt.legend()
    if not os.path.exists('E2'):
        os.mkdir('E2')
    plt.savefig('E2\\'+title+'.png')
    return

def e2(X,y,n_components,ds_name,gtid):
    # Randomized Projections
    X_rp = apply_randomized_projections(X, n_components, ds_name,gtid)

    # PCA
    X_pca = apply_pca(X, n_components,ds_name,gtid)

    # ICA
    X_ica = apply_ica(X, n_components, ds_name,gtid)

    # LLE
    X_lle = apply_lle(X, n_components, ds_name,gtid)

    # Plot results for iris dataset
    plot_2d_reduction_e2(X_rp, y, "Randomized Projections on "+ds_name+" Dataset")
    plot_2d_reduction_e2(X_pca, y, "PCA on "+ds_name+" Dataset")
    plot_2d_reduction_e2(X_ica, y, "ICA on "+ds_name+" Dataset")
    plot_2d_reduction_e2(X_lle, y, "LLE on "+ds_name+" Dataset")

def plot_2d_reduction_e3(X, y, title):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter)
    # plt.legend()
    if not os.path.exists('E3'):
        os.mkdir('E3')
    plt.savefig('E3\\'+title+'.png')
    return

def e3(drs,dss,gtid):
    results = []

    # Apply dimensionality reduction and clustering algorithms
    for dr_name, dr_func in drs.items():
        for dataset_name, (X, y, n_components, n_clusters) in dss.items():
            X_dr = dr_func(X, n_components,dataset_name,gtid)
            
            # Apply EM
            y_pred_em, ari_em, silhouette_em = apply_em(X_dr, y, n_clusters, dataset_name,gtid)
            results.append((dataset_name, dr_name, "EM", ari_em, silhouette_em))
            
            # Apply K-Means
            y_pred_kmeans, ari_kmeans, silhouette_kmeans = apply_kmeans(X_dr, y, n_clusters, dataset_name,gtid)
            results.append((dataset_name, dr_name, "K-Means", ari_kmeans, silhouette_kmeans))
    
    for dr_name, dr_func in drs.items():
        for dataset_name, (X, y, n_components, n_clusters) in dss.items():
            X_dr = dr_func(X, n_components,dataset_name,gtid)
            
            # Plot EM results
            y_pred_em, _, _ = apply_em(X_dr, y, n_clusters, dataset_name,gtid)
            plot_2d_reduction_e3(X_dr, y_pred_em, f"EM on {dataset_name} with {dr_name}")
            
            # Plot K-Means results
            y_pred_kmeans, _, _ = apply_kmeans(X_dr, y, n_clusters, dataset_name,gtid)
            plot_2d_reduction_e3(X_dr, y_pred_kmeans, f"K-Means on {dataset_name} with {dr_name}")
    for dataset_name, dr_name, clustering_method, ari, silhouette in results:
        print(f"{clustering_method} on {dataset_name} with {dr_name}:")
        print(f"  Adjusted Rand Index: {ari:.2f}")
        print(f"  Silhouette Score: {silhouette:.2f}\n")
