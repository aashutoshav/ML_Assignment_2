import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):

    def __init__(self, X, K, max_iters=100, seed=5):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K
        self.num_iters = 1
        self.seed = seed

    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        raise NotImplementedError

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        raise NotImplementedError

    def normalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        raise NotImplementedError

    def multinormalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. Note the value in self.D may be outdated and not correspond to the current dataset.
            3. You may wanna check if the matrix is singular before implementing calculation process.
        """
        raise NotImplementedError

    def create_pi(self):
        """
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        raise NotImplementedError

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        raise NotImplementedError

    def create_mu_kmeans(self, kmeans_max_iters=1000, kmeans_rel_tol=1e-05):
        """
        Intialize centers for each gaussian using your KMeans implementation from Q1
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        raise NotImplementedError

    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        raise NotImplementedError

    def _init_components(self, kmeans_init=False, **kwargs):
        """
        Args:
            kmeans_init: whether to randomly initiate the centers or to use kmeans
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        # keep this seeding, so your random generation in create_mu is consistent with ours
        if self.seed is not None:
            np.random.seed(self.seed)

        raise NotImplementedError

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        raise NotImplementedError

    def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        raise NotImplementedError

    def __call__(
        self, full_matrix=FULL_MATRIX, kmeans_init=False, rel_tol=1e-16, **kwargs
    ):
        """
        Args:
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        disable_tqdm = kwargs.pop("disable_tqdm", False)
        pi, mu, sigma = self._init_components(kmeans_init, **kwargs)
        pbar = tqdm(range(self.max_iters), disable=disable_tqdm)
        prev_loss = None
        for it in pbar:
            tau = self._E_step(pi, mu, sigma, full_matrix)
            pi, mu, sigma = self._M_step(tau, full_matrix)
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if not disable_tqdm:
                pbar.set_description("iter %d, loss: %.4f" % (it, loss))
            self.num_iters += 1
        return tau, (pi, mu, sigma)


def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
    """
    Clusters pixels in the input image

    Each pixel can be considered as a separate data point (of length 3),
    which you can then cluster using GMM. Then, process the outputs into
    the shape of the original image, where each pixel is its most likely value.

    Args:
        image: input image of shape(H, W, 3)
        K: number of components
        max_iters: maximum number of iterations in GMM. Default is 10
        full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
    Return:
        clustered_img: image of shape(H, W, 3) after pixel clustering

    Hints:
        What do mu and tau represent?
    """
    raise NotImplementedError


def density(points, pi, mu, sigma, gmm):
    """
    Evaluate the density at each point on the grid.
    Args:
        points: (N, 2) numpy array containing the coordinates of the points that make up the grid.
        pi: (K,) numpy array containing the mixture coefficients for each class
        mu: (K, D) numpy array containing the means of each cluster
        sigma: (K, D, D) numpy array containing the covariance matrixes of each cluster
        gmm: an instance of the GMM model

    Return:
        densities: (N, ) numpy array containing densities at each point on the grid

    HINT: You should be using the formula given in the hints.
    """
    raise NotImplementedError


def rejection_sample(xmin, xmax, ymin, ymax, pi, mu, sigma, gmm, dmax=1, M=0.1):
    """
    Performs rejection sampling. Keep sampling datapoints until d <= f(x, y) / M
    Args:
        xmin: lower bound on x values
        xmax: upper bound on x values
        ymin: lower bound on y values
        ymax: upper bound on y values
        gmm: an instance of the GMM model
        dmax: the upper bound on d
        M: scale_factor. can be used to control the fraction of samples that are rejected

    Return:
        x, y: the coordinates of the sampled datapoint

    HINT: Refer to the links in the hints
    """
    raise NotImplementedError
