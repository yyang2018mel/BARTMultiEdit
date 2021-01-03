
import numpy as np
from enum import Enum
from jpype.types import *
from BARTMultiEdit import BartRegression, Hyperparam, DataContext


class MHMode(Enum):
    OneStep = 1
    MultiStep = 2


class BART:

    def __init__(self, T, gibbs_rounds, burn_in, is_classification,
                 mh_mode, alpha, beta, k, nu, q, prob_grow, prob_prune,
                 seed=2020, mean_stride=1, verbose=Hyperparam.VerboseLevel.NoReporting):

        self.T = T
        self.gibbsRounds = gibbs_rounds
        self.burnIn = burn_in
        self.classification = is_classification
        self.mhMode = Hyperparam.MHMode.OneStep if mh_mode == MHMode.OneStep else Hyperparam.MHMode.MultiStep
        self.hyperParam = Hyperparam(seed, T, gibbs_rounds, burn_in, self.mhMode, mean_stride,
                                     is_classification, alpha, beta, k, nu, q, prob_grow, prob_prune, verbose)
        self.X = None
        self.y = None
        self.dataContext = None
        self.bartModel = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        jX = JArray.of(X)
        jy = JArray.of(y)
        self.dataContext = DataContext(jX, jy, (not self.classification))
        if not self.classification:
            self.bartModel = BartRegression(self.hyperParam, self.dataContext)
        self.bartModel.initialize()
        self.bartModel.doGibbsSampling()
        return self

    def predict(self, X):
        predictions = self.bartModel.getPredictionsFromGibbsTreeSamples(JArray.of(X), False)
        return np.array(predictions)



