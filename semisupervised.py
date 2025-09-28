import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32


def complete_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_complete: n x (D+1) array (n <= N) where values contain both complete features and labels
    """
    has_nan = np.isnan(data).any(axis=1)
    comp_mask = ~has_nan
    return data[comp_mask]


def incomplete_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x (D+1) array (n <= N) where values contain incomplete features but complete labels
    """
    features = data[:, :-1]
    labels = data[:, -1]
    
    nan_in_feat = np.isnan(features).any(axis=1)
    valid_label = ~np.isnan(labels)
    inc_mask = nan_in_feat & valid_label
    
    return data[inc_mask]


def unlabeled_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        unlabeled_complete: n x (D+1) array (n <= N) where values contain complete features but incomplete labels
    """
    features = data[:, :-1]
    labels = data[:, -1]
    
    has_comp_feat = ~np.isnan(features).any(axis=1)
    has_nan_label = np.isnan(labels)
    unl_mask = has_comp_feat & has_nan_label
    
    return data[unl_mask]


class CleanData(object):

    def __init__(self):
        pass

    def pairwise_dist(self, x, y):
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        x_sq = np.sum(x**2, axis=1, keepdims=True)
        y_sq = np.sum(y**2, axis=1)
        xy_dot = 2 * (x @ y.T)
        
        dist_sq = np.maximum(x_sq + y_sq - xy_dot, 0)
        
        return np.sqrt(dist_sq)

    def __call__(self, incomplete_points, complete_points, K, **kwargs):
        """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points.

        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points

        Notes:
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors, from the complete dataset, with the same class labels;
            (5) There may be missing values in any of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you MAY use a for-loop over the M labels and the D features (e.g. omit one feature at a time)
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        filled_points = incomplete_points.copy()
        D = complete_points.shape[1] - 1
        unique_labels = np.unique(complete_points[:, -1])
        
        feature_indices = np.arange(D)
        
        for d in range(D):
            for label in unique_labels:
                target_mask = np.isnan(incomplete_points[:, d]) & (incomplete_points[:, -1] == label)
                if not np.any(target_mask):
                    continue
                
                target_points = incomplete_points[target_mask]
                
                candidate_mask = complete_points[:, -1] == label
                candidate_neighbors = complete_points[candidate_mask]
                
                dist_feature_indices = feature_indices[feature_indices != d]

                distances = self.pairwise_dist(
                    target_points[:, dist_feature_indices],
                    candidate_neighbors[:, dist_feature_indices]
                )
                
                neighbor_indices = np.argsort(distances, axis=1)[:, :K]
                
                k_neighbors = candidate_neighbors[neighbor_indices]
                neighbor_values_d = k_neighbors[:, :, d]
                
                imputed_values = np.mean(neighbor_values_d, axis=1)
                
                filled_points[target_mask, d] = imputed_values

        clean_points = np.vstack([complete_points, filled_points])
        
        return clean_points


def median_clean_data(data):
    """
    Args:
        data: N x (D+1) numpy array where only last column is guaranteed non-NaN values and is the labels
    Return:
        median_clean: N x (D+1) numpy array where each NaN value in data has been replaced by the median feature value
    Notes:
        (1) When taking the median of any feature, do not count the NaN value
        (2) Return all values to max one decimal point
        (3) The labels column will never have NaN values
    """
    cleaned_data = data.copy()
    D = cleaned_data.shape[1] - 1
    
    for d in range(D):
        median_value = np.nanmedian(cleaned_data[:, d])
        nan_mask = np.isnan(cleaned_data[:, d])
        cleaned_data[nan_mask, d] = median_value
        
    return np.round(cleaned_data, decimals=1)


class SemiSupervised(object):

    def __init__(self):
        pass

    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        max_logits = np.max(logit, axis=1, keepdims=True)
        cent_logits = logit - max_logits
        exp_logits = np.exp(cent_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        prob = exp_logits / sum_exp
        return prob

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        max_logits = np.max(logit, axis=1, keepdims=True)
        cent_logits = logit - max_logits
        exp_logits = np.exp(cent_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        log_sum_exp = np.log(sum_exp) + max_logits
        
        return log_sum_exp
        

    def normalPDF(self, logit, mu_i, sigma_i):
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        mu = mu_i[0]
        sigma = sigma_i[0]
        D = mu.shape[0]
        
        variances = np.diagonal(sigma) + SIGMA_CONST
        term1 = -(D / 2.0) * np.log(2 * np.pi)
        term2 = -0.5 * np.sum(np.log(variances))
        centered_pts = logit - mu
        term3 = -0.5 * np.sum((centered_pts**2) / variances, axis=1)
        log_pdf = term1 + term2 + term3

        return np.exp(log_pdf).reshape(1, -1)

    def _init_components(self, points, K, **kwargs):
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K; contains the prior probabilities of each class k
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.

        Hint:
            1. Given that the data is labeled, what's the best estimate for pi?
            2. Using the labels, you can look at individual clusters and estimate the best value for mu, sigma
        """
        features = points[:, :-1]
        labels = points[:, -1]
        N, D = features.shape

        pi = np.zeros(K)
        mu = np.zeros((K, D))
        sigma = np.zeros((K, D, D))

        for k in range(K):
            points_k = features[labels == k]
            
            pi[k] = len(points_k) / N
            
            mu[k] = np.mean(points_k, axis=0)
            cov_matrix = np.cov(points_k, rowvar=False)
            sigma[k] = np.diag(np.diag(cov_matrix))

        return pi, mu, sigma

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        N, D = points.shape
        K = pi.shape[0]
        ll = np.zeros((N, K))
        
        for k in range(K):
            pdf = self.normalPDF(points, mu[k:k+1, :], sigma[k:k+1, :, :])
            ll[:, k] = np.log(pi[k] + LOG_CONST) + np.log(pdf[0] + LOG_CONST)
            
        return ll
            

    def _E_step(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        ll = self._ll_joint(points, pi, mu, sigma, **kwargs)
        gamma = self.softmax(ll)
        
        return gamma

    def _M_step(self, points, gamma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.

        Hint:  There are formulas in the slide.
        """
        N, D = points.shape
        K = gamma.shape[1]
        N_k = np.sum(gamma, axis=0)

        pi = N_k / N

        mu = (gamma.T @ points) / (N_k[:, np.newaxis] + LOG_CONST)

        sigma = np.zeros((K, D, D))
        for k in range(K):
            cent_points = points - mu[k]
            wt_sq_diff = gamma[:, k][:, np.newaxis] * (cent_points**2)
            diag_sig = np.sum(wt_sq_diff, axis=0) / (N_k[k] + LOG_CONST)
            sigma[k] = np.diag(diag_sig)
                
        return pi, mu, sigma

    def __call__(
        self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs
    ):
        """
        Args:
            points: N x (D+1) numpy array, where
                - N is # points,
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint: Look at Table 1 in the paper
        """
        labels = points[:, -1]
        labeled_mask = ~np.isnan(labels)
        unlabeled_mask = np.isnan(labels)
        
        labeled_data = points[labeled_mask]
        unlabeled_points = points[unlabeled_mask, :-1]

        pi, mu, sigma = self._init_components(labeled_data, K)
        
        prev_loss = None
        
        for i in range(max_iters):
            gamma_unlabeled = self._E_step(unlabeled_points, pi, mu, sigma)
            
            labeled_labels = labeled_data[:, -1].astype(int)
            gamma_labeled = np.eye(K)[labeled_labels]
            
            all_points = np.vstack([labeled_data[:, :-1], unlabeled_points])
            all_gamma = np.vstack([gamma_labeled, gamma_unlabeled])
            
            pi, mu, sigma = self._M_step(all_points, all_gamma)
            
            ll_joint = self._ll_joint(all_points, pi, mu, sigma)
            loss = -np.sum(self.logsumexp(ll_joint))
            
            if i > 0 and prev_loss is not None:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol or (diff / np.abs(prev_loss)) < rel_tol:
                    break
            prev_loss = loss

        return pi, mu, sigma


class ComparePerformance(object):

    def __init__(self):
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K: int) -> float:
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N_t is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number

        Note: validation_data will NOT include any unlabeled points
        """
        pi, mu, sigma = SemiSupervised()(training_data, K)
        classification_probs = SemiSupervised()._E_step(
            validation_data[:, :-1], pi, mu, sigma
        )
        classification = np.argmax(classification_probs, axis=1)
        semi_supervised_score = accuracy_score(validation_data[:, -1], classification)
        return semi_supervised_score

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float:
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: both training_data and validation_data will NOT include any unlabeled points
        """
        gnb_model = GaussianNB()
        gnb_model.fit(training_data[:, :-1], training_data[:, -1])
        gnb_score = gnb_model.score(validation_data[:, :-1], validation_data[:, -1])
        return gnb_score

    @staticmethod
    def accuracy_comparison():
        all_data = np.loadtxt("data/data.csv", delimiter=",")
        labeled_complete = complete_(all_data)
        labeled_incomplete = incomplete_(all_data)
        unlabeled = unlabeled_(all_data)
        cleaned_data = CleanData()(labeled_incomplete, labeled_complete, 10)
        cleaned_and_unlabeled = np.concatenate((cleaned_data, unlabeled), 0)
        labeled_data = np.concatenate((labeled_complete, labeled_incomplete), 0)
        median_cleaned_data = median_clean_data(labeled_data)
        print(f"All Data shape:                 {all_data.shape}")
        print(f"Labeled Complete shape:         {labeled_complete.shape}")
        print(f"Labeled Incomplete shape:       {labeled_incomplete.shape}")
        print(f"Labeled shape:                  {labeled_data.shape}")
        print(f"Unlabeled shape:                {unlabeled.shape}")
        print(f"Cleaned data shape:             {cleaned_data.shape}")
        print(f"Cleaned + Unlabeled data shape: {cleaned_and_unlabeled.shape}")
        validation = np.loadtxt("data/validation.csv", delimiter=",")
        accuracy_complete_data_only = ComparePerformance.accuracy_GNB(
            labeled_complete, validation
        )
        accuracy_cleaned_data = ComparePerformance.accuracy_GNB(
            cleaned_data, validation
        )
        accuracy_median_cleaned_data = ComparePerformance.accuracy_GNB(
            median_cleaned_data, validation
        )
        accuracy_semi_supervised = ComparePerformance.accuracy_semi_supervised(
            cleaned_and_unlabeled, validation, 2
        )
        print("===COMPARISON===")
        print(
            f"Supervised with only complete data, GNB Accuracy: {np.round(100.0 * accuracy_complete_data_only, 3)}%"
        )
        print(
            f"Supervised with KNN clean data, GNB Accuracy:     {np.round(100.0 * accuracy_cleaned_data, 3)}%"
        )
        print(
            f"Supervised with Median clean data, GNB Accuracy:    {np.round(100.0 * accuracy_median_cleaned_data, 3)}%"
        )
        print(
            f"SemiSupervised Accuracy:                          {np.round(100.0 * accuracy_semi_supervised, 3)}%"
        )
