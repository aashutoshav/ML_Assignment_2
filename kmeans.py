import numpy as np

class KMeans(object):

    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Use np.random.choice to initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        num_points = self.points.shape[0]
        random_indices = np.random.choice(num_points, self.K, replace=False)
        centers = self.points[random_indices]
        return centers

    def kmpp_init(self):
        """
            Use the intuition that points further away from each other will probably be better initial centers.
            To complete this method, refer to the steps outlined below:.
            1. Sample 1% of the points from dataset, uniformly at random (UAR) and without replacement.
            This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
            2. From the above sample, select only one random point to be the first cluster center.
            3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
            4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
            In other words, we will choose the next center based on the maximum of the minimum calculated distance
            instead of sampling randomly like in step 2. You may break ties arbitrarily.
            5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
        Return:
            self.centers : K x D numpy array, the centers.
        Hint:
            You could use functions like np.vstack() here.
        """
        num_points, D = self.points.shape
        sample_size = max(self.K, int(num_points * 0.01))
        sample_indices = np.random.choice(num_points, size=sample_size, replace=False)
        sample = self.points[sample_indices]

        centers = np.empty((self.K, D))

        first_idx = np.random.randint(sample.shape[0])
        centers[0] = sample[first_idx]
        
        min_dists_sq = np.full(sample.shape[0], np.inf)

        for i in range(1, self.K):
            last_center = centers[i-1][np.newaxis, :]
            
            dists_sq_to_last = pairwise_dist(sample, last_center)**2

            min_dists_sq = np.minimum(min_dists_sq, dists_sq_to_last.flatten())
            next_center_idx = np.argmax(min_dists_sq)
            centers[i] = sample[next_center_idx]
            
        return centers

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Do not use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        dist_matrix = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(dist_matrix, axis=1)
        return self.assignments

    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        new_centers = []
        for k in range(self.K):
            cluster_points = self.points[self.assignments == k]
            if len(cluster_points) > 0:
                new_center = np.mean(cluster_points, axis=0)
                new_centers.append(new_center)
                
        self.centers = np.array(new_centers)
        return self.centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        if self.assignments is None:
            self.loss = 0.0
            return self.loss
    
        assigned_centers = self.centers[self.assignments]
        diff = self.points - assigned_centers
        self.loss = np.sum(diff ** 2)
        return self.loss

    def train(self):
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!

        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.

        HINT: Do not loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        loss = np.inf
        for i in range(self.max_iters):
            self.update_assignment()
            self.update_centers()
            
            num_curr_centers = self.centers.shape[0]
            if num_curr_centers < self.K:
                num_missing = self.K - num_curr_centers
                random_indices = np.random.choice(self.points.shape[0], num_missing, replace=False)
                new_centers = self.points[random_indices]
                self.centers = np.vstack([self.centers, new_centers])
                self.update_assignment()
                
            curr_loss = self.get_loss()
            
            if i > 0 and loss != 0 and (loss - curr_loss) / loss < self.rel_tol:
                break
            
            loss = curr_loss
            
        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_distance function
    """
    xsq = np.sum(x**2, axis=1, keepdims=True)
    ysq = np.sum(y**2, axis=1)
    
    xy_dot = 2 * (x @ y.T)
    dist_sq = xsq + ysq - xy_dot
    dist_sq_clipped = np.maximum(dist_sq, 0)
    
    return np.sqrt(dist_sq_clipped)


def pairwise_dist_inf(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist[i, j] is the infinity or chebyshev distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_dist_inf function
    """
    x_reshaped = x[:, np.newaxis, :]
    diff = np.abs(x_reshaped - y)
    dist = np.max(diff, axis=2)
    
    return dist


def adjusted_rand_statistic(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        adjusted rand index value: final coefficient value of type np.float64

    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth data points.
        1. Iterate over all distinct pairs of points.
        2. Compare the prediction pair label with the ground truth pair.
        3. Based on the analysis, we can figure out whether both points fall under TP/FP/FN/FP
           i.e. if a pair falls under TP, increment by 2 (one for each point in the pair).
        4. Then calculate the adjusted rand index value
    """
    
    def combinations(n):
        return n * (n - 1) // 2 if n >= 2 else 0
    
    n_samples = len(xGroundTruth)
    if n_samples < 2:
        return 1.0
    
    true_labels = np.unique(xGroundTruth)
    pred_labels = np.unique(xPredicted)
    contingency_table = np.zeros((true_labels.size, pred_labels.size), dtype=int)
    
    true_map = {label: i for i, label in enumerate(true_labels)}
    pred_map = {label: i for i, label in enumerate(pred_labels)}
    
    for i in range(n_samples):
        true_idx = true_map[xGroundTruth[i]]
        pred_idx = pred_map[xPredicted[i]]
        contingency_table[true_idx, pred_idx] += 1
    
    sum_comb_nij = np.sum([combinations(n) for n in contingency_table.flatten()])
    
    sum_comb_ai = np.sum([combinations(n) for n in np.sum(contingency_table, axis=1)])
    sum_comb_bj = np.sum([combinations(n) for n in np.sum(contingency_table, axis=0)])
    
    total_pairs = combinations(n_samples)
    if total_pairs == 0:
        return 1.0

    expected_idx = (sum_comb_ai * sum_comb_bj) / total_pairs
    max_idx = 0.5 * (sum_comb_ai + sum_comb_bj)
    denominator = max_idx - expected_idx

    if denominator == 0:
        return 1.0 if sum_comb_nij == expected_idx else 0.0
    
    ari = (sum_comb_nij - expected_idx) / denominator
    
    return np.float64(ari)


def silhouette_score(X, labels):
    """
    Args:
        X : N x D numpy array, where N is # points and D is the dimensionality
        labels : 1D numpy array of predicted labels of length N where N = no. of test samples
    Return:
        silhouette score: final coefficient value of type np.float64

    HINT: You can use loops for this function.
    HINT: The idea is to calculate the mean distance between a point and the other points
    in its cluster (the intra cluster distance) and the mean distance between a point and the
    other points in its closest cluster (the inter cluster distance)
        1.  Calculate the pairwise_dist between all points to each other (N x N)
        2.  Loop over all points in the provided data (X) and for each point calculate:

            Intra Cluster Distances (point p to the other points in its own cluster)
                a. Identify all points in the same cluster as p (excluding p itself)
                b. Calculate the mean pairwise_dist between p and the other points
                c. If there are no other points in the same cluster, assign an
                   intra-cluster distance of 0

            Inter Cluster Distances (point p to the points in the closest cluster)
                a. Loop over all clusters except for p's cluster
                b. For each cluster, identify all points belonging to it
                c. Calculate the mean pairwise_dist between p and those points
                d. Set the inter-cluster distance to the minimum mean pairwise_dist
                   among all clusters. Again, if there are no other clusters, use
                   an inter-cluster distance of 0.

        3. Calculate the silhouette scores for each point using
                s_i = (mu_out(x_i) - mu_in(x_i)) / max(mu_out(x_i), mu_in(x_i))
        4. Average the silhouette score across all points

    Note: Refer to the Clustering Evaluation slides from Lecture
    """
    N = X.shape[0]
    if len(np.unique(labels)) < 2:
        return 0.0

    all_dists = pairwise_dist(X, X)
    
    scores = []
    for i in range(N):
        p_label = labels[i]
        
        in_cluster_mask = (labels == p_label) & (np.arange(N) != i)
        if not np.any(in_cluster_mask):
            a_i = 0
        else:
            a_i = np.mean(all_dists[i, in_cluster_mask])

        other_labels = np.unique(labels[labels != p_label])
        
        if len(other_labels) == 0:
            b_i = 0
        else:
            mean_other_dists = []
            for other_label in other_labels:
                other_cluster_mask = (labels == other_label)
                mean_dist = np.mean(all_dists[i, other_cluster_mask])
                mean_other_dists.append(mean_dist)
            b_i = np.min(mean_other_dists)

        denominator = max(a_i, b_i)
        if denominator == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / denominator
        scores.append(s_i)

    return np.mean(scores)
