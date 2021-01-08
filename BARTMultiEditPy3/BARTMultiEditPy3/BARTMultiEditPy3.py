
import numpy as np
from enum import Enum
from jpype.types import *
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from BARTMultiEdit import BartRegression, Hyperparam, DataContext


class MHMode(Enum):
    OneStep = 0
    MultiStep = 1

class VerboseLevel(Enum):
    NoReporting = 0
    ReportLoss = 1
    ReportDetails = 2


class BARTRegression(BaseEstimator, RegressorMixin):

    def __init__(self, T=50, gibbs_rounds=1250, burn_in=250, mh_mode=MHMode.OneStep,
                 alpha=0.95, beta=2.0, k=2.0, nu=3.0, q=0.9, prob_grow=3./9., prob_prune=3./9.,
                 seed=2020, mean_stride=1, verbose=VerboseLevel.NoReporting):

        self.T = T
        self.gibbs_rounds = gibbs_rounds
        self.burn_in = burn_in
        self.mh_mode = mh_mode
        mh_mode_to_java = mh_mode.value
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.nu = nu
        self.q = q
        self.prob_grow = prob_grow
        self.prob_prune = prob_prune
        self.seed = seed
        self.mean_stride = mean_stride
        self.verbose = verbose
        verbose_to_jave = verbose.value
        self.hyperParam = Hyperparam(seed, T, gibbs_rounds, burn_in, mh_mode_to_java, mean_stride,
                                     False, alpha, beta, k, nu, q, prob_grow, prob_prune, verbose_to_jave)
        self.X_ = None
        self.y_ = None
        self.data_context = None
        self.bart_model = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        jX = JArray.of(X)
        jy = JArray.of(y)
        self.data_context = DataContext(jX, jy, True)
        self.bart_model = BartRegression(self.hyperParam, self.data_context)
        self.bart_model.initialize()
        self.bart_model.doGibbsSampling()
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        predictions = self.bart_model.getPredictionsFromGibbsTreeSamples(JArray.of(X), False)
        return np.array(predictions)



