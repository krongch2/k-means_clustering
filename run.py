import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets

def plot(ax, X, labels, centroids, legend=False, ylabel=None, xlabel=None, title=None):
    d = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'label': labels,
        })
    palette = sns.color_palette()
    for i, label in enumerate(d['label'].unique()):
        color = palette[i]
        dd = d.loc[d['label'] == label, :]
        ax.scatter(dd['x'], dd['y'], color=color, label=label, lw=1, s=40, ec='white', alpha=0.7, zorder=0)
        ax.scatter(centroids[i, 0], centroids[i, 1], marker='D', s=70, color='k', zorder=20)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(title='label', fancybox=False)

def plot_multi(X, labels_all, centroids_all, change_all, ncols=6):
    niter = len(centroids_all)
    nrows = int(np.ceil(niter/ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=(ncols*3, nrows*3))
    for i in range(niter):
        row_idx = int(i // ncols)
        col_idx = int(i % ncols)
        if nrows == 1:
            ax = axs[col_idx]
        else:
            ax = axs[row_idx, col_idx]
        legend = True if row_idx == 0 and col_idx == ncols - 1 else False
        xlabel = 'x' if row_idx == nrows - 1 else None
        ylabel = 'y' if col_idx == 0 else None
        change = change_all[i]
        plot(ax, X, labels_all[i], centroids_all[i],
            legend=legend, xlabel=xlabel, ylabel=ylabel, title=f'iter={i}, change={change:.4f}')
    fig.tight_layout()
    plt.savefig('result.pdf', bbox_inches='tight')

def get_distance(a, b):
    '''
    Returns a Euclidean distance between two vectors (or two points in this case).
    '''
    return np.sqrt(sum((a - b)**2))

def init_centroids(k, X):
    '''
    Initializes `k` centroids for k-means.
    Args:
        k [int]: number of centroids
        X [array(N, 2)]: N is the number of data points
    Returns: array(k, 2)
    '''
    x_mean = np.random.uniform(low=X[:, 0].min(), high=X[:, 0].max(), size=(k,))
    y_mean = np.random.uniform(low=X[:, 1].min(), high=X[:, 1].max(), size=(k,))
    return np.array([x_mean, y_mean]).T

def get_labels(X, centroids):
    '''
    Assigns labels to each point.
    Args:
        X [array(N, 2)]:
        centroids [array(k, 2)]
    Returns: array(N)
    '''
    labels = []
    for point in X:
        distances = []
        for centroid in centroids:
            distance = get_distance(point, centroid)
            distances.append(distance)
        distances = np.array(distances)
        labels.append(np.argmin(distances))
    return np.array(labels)

def get_centroids(X, labels):
    '''
    Groups data points that have the same label
    and calculates the centroid for each group.
    Returns: array(k, 2)
    '''
    centroids = []
    for label in np.unique(labels):
        idx = np.where(labels == label)
        XX = X[idx]
        centroids.append(np.mean(XX, axis=0))
    return np.array(centroids)

def get_centroid_change(centroids_prev, centroids_new):
    '''
    Measures the sum of distance change between old and new centroids.
    '''
    change = 0
    for centroid_prev, centroid_new in zip(centroids_prev, centroids_new):
        change += get_distance(centroid_prev, centroid_new)
    return float(change)

def k_means(k, X, tol=1e-3):
    '''
    Runs k-means and returns the result of each iteration
    '''
    centroids_prev = init_centroids(k, X)
    centroids_all = []
    labels_all = []
    change_all = [np.nan]
    while True:
        labels = get_labels(X, centroids_prev)
        centroids_all.append(centroids_prev)
        labels_all.append(labels)
        centroids_new = get_centroids(X, labels)
        change = get_centroid_change(centroids_prev, centroids_new)
        if change < tol:
            break
        change_all.append(change)
        centroids_prev = centroids_new
    return centroids_all, labels_all, change_all

if __name__ == '__main__':
    X, thrue_labels = sklearn.datasets.make_blobs(n_samples=500, centers=3, n_features=2, random_state=0)
    centroids_all, labels_all, change_all = k_means(3, X)
    plot_multi(X, labels_all, centroids_all, change_all)
