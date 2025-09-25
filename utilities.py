import matplotlib.pyplot as plt
import numpy as np
from gmm import GMM
from kmeans import KMeans


def image_to_matrix(image_file, grays=False):
    """
    Convert .png image to matrix
    of values.
    params:
    image_file = str
    grays = Boolean
    returns:
    img = (color) np.ndarray[np.ndarray[np.ndarray[float]]]
    or (grayscale) np.ndarray[np.ndarray[float]]
    """
    img = plt.imread(image_file)
    if len(img.shape) == 3 and img.shape[2] > 3:
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = img[r, c, 0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = img[r, c, 0]
        img = new_img
    return img


def update_image_values(k, image_values, r, c, ch):
    kmeans = KMeans(image_values, k)
    centers, cluster_idx, loss = kmeans.train()
    updated_image_values = np.copy(image_values)
    for i in range(0, k):
        indices_current_cluster = np.where(cluster_idx == i)[0]
        updated_image_values[indices_current_cluster] = centers[i]
    updated_image_values = updated_image_values.reshape(r, c, ch)
    return updated_image_values


def plot_image(img_list, title_list, figsize=(15, 10)):
    fig, axes = plt.subplots(len(img_list) // 2, 2, figsize=figsize)
    p = 0
    for i, ax in enumerate(axes):
        for col in range(len(img_list) // 2):
            axes[i, col].imshow(img_list[p])
            axes[i, col].set_title(title_list[p])
            axes[i, col].axis("off")
            p += 1


def plot_images(img_list, title_list, figsize=(20, 10)):
    assert len(img_list) == len(title_list)
    fig, axes = plt.subplots(1, len(title_list), figsize=figsize)
    for i, ax in enumerate(axes):
        ax.imshow(img_list[i] / 255.0)
        ax.set_title(title_list[i])
        ax.axis("off")
    plt.show()


"""
    Creates 2-D KMeans plots
    Args:
        X: observations
        k: number of clusters
"""


def create_plots(X, k):
    np.random.seed(0)
    feature_indices = [(6, 12), (3, 4)]
    num_figures_per_row = 2
    num_columns = len(feature_indices) // num_figures_per_row
    if len(feature_indices) % num_figures_per_row != 0:
        num_columns += 1
    fig, axes = plt.subplots(
        num_columns,
        num_figures_per_row,
        figsize=(5 * num_figures_per_row, 5 * num_columns),
    )
    axes = axes.ravel()
    for i, (f1, f2) in enumerate(feature_indices):
        X_subset = X[:, [f1, f2]]
        kmeans = KMeans(X, k)
        centers, assignments, loss = kmeans.train()
        axes[i].scatter(
            X_subset[:, 0], X_subset[:, 1], c=assignments, cmap="viridis", alpha=0.6
        )
        axes[i].scatter(
            centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centers"
        )
        axes[i].scatter([], [], c="yellow", label="Cluster 0")
        axes[i].scatter([], [], c="purple", label="Cluster 1")
        axes[i].set_title(f"K-Means Clustering (Features {f1} & {f2})")
        axes[i].set_xlim(X_subset[:, 0].min() - 5, X_subset[:, 0].max() + 5)
        axes[i].set_ylim(X_subset[:, 1].min() - 5, X_subset[:, 1].max() + 5)
        axes[i].set_xlabel(f"Feature {f1}")
        axes[i].set_ylabel(f"Feature {f2}")
        axes[i].legend()
    plt.tight_layout()
    plt.show()
