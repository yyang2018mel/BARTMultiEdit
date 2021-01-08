package BARTMultiEdit;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.exception.MathRuntimeException;

import java.util.Random;

public class Hyperparam {

    public enum MHMode {
        OneStep, MultiStep;

        public static MHMode fromInteger(int x) {
            switch(x) {
                case 0: return OneStep;
                case 1: return MultiStep;
            }
            return null;
        }
    }

    public enum VerboseLevel {
        NoReporting, ReportLoss, ReportDetails;

        public static VerboseLevel fromInteger(int x) {
            switch (x) {
                case 0: return NoReporting;
                case 1: return ReportLoss;
                case 2: return ReportDetails;
            }
            return null;
        }
    }

    /** random seed */
    protected final int seed;
    /** mode to do MH tree modifying: OneStep, MultiStep */
    protected final MHMode mhMode;
    /** only used when mhMode == MultiStep */
    protected final double meanStride;
    /** the number of trees in a vBART model */
    protected int T;
    /** how many total Gibbs samples in a BART model creation */
    protected int numGibbsTotal;
    /** length of Gibbs burn-in period */
    protected int numGibbsBurnin;
    /** is for classification or regression */
    protected boolean isClassification;

    /** a hyperparameter that controls how easy it is to grow new nodes in a tree independent of depth */
    protected double alpha;
    /** a hyperparameter that controls how easy it is to grow new nodes in a tree dependent on depth which makes it more difficult as the tree gets deeper */
    protected double beta;
    /** the center of the prior of the terminal node prediction distribution */
    protected double mu_mu;
    /** the variance of the prior of the terminal node prediction distribution */
    protected double σ_mu_sq;
    /** this controls where to set <code>sigma_mu</code> by forcing the variance to be this number of standard deviations on the normal CDF */
    protected double k;
    /** half the shape parameter and half the multiplicand of the scale parameter of the inverse gamma prior on the variance */
    protected double nu;
    /** the multiplier of the scale parameter of the inverse gamma prior on the variance */
    protected double lambda;
    /** At a fixed <code>hyper_nu</code>, this controls where to set <code>lambda</code> by forcing q proportion to be at that value in the inverse gamma CDF */
    protected double q;
    protected final VerboseLevel verbose;

    protected final double probGrow;
    protected final double probPrune;

    /** The static field that controls the bounds on the transformed y variable which is between negative and positive this value */
    protected static final double YminAndYmaxHalfDiff = 0.5;

    public Hyperparam(int seed, int T, int num_gibbs_total, int num_gibbs_burnin, MHMode mhMode,
                      double meanStride, boolean classification, double alpha, double beta, double k, double nu, double q,
                      double prob_insert, double prob_delete, VerboseLevel verbose) {
        this.seed = seed;
        this.T = T;
        this.numGibbsTotal = num_gibbs_total;
        this.numGibbsBurnin = num_gibbs_burnin;
        this.mhMode = mhMode;
        this.meanStride = meanStride;
        this.isClassification = classification;
        this.alpha = alpha;
        this.beta = beta;
        this.k = k;
        this.nu = nu;
        this.q = q;
        this.probGrow = prob_insert;
        this.probPrune = prob_delete;
        this.verbose = verbose;
    }

    public Hyperparam(int seed, int T, int num_gibbs_total, int num_gibbs_burnin, int mhMode,
                      double meanStride, boolean classification, double alpha, double beta, double k, double nu, double q,
                      double prob_insert, double prob_delete, int verbose) {
        this.seed = seed;
        this.T = T;
        this.numGibbsTotal = num_gibbs_total;
        this.numGibbsBurnin = num_gibbs_burnin;
        this.mhMode = MHMode.fromInteger(mhMode);
        this.meanStride = meanStride;
        this.isClassification = classification;
        this.alpha = alpha;
        this.beta = beta;
        this.k = k;
        this.nu = nu;
        this.q = q;
        this.probGrow = prob_insert;
        this.probPrune = prob_delete;
        this.verbose = VerboseLevel.fromInteger(verbose);
    }

    private void calculateSigmaMu() {
        this.mu_mu = 0.;
        this.σ_mu_sq = !isClassification
                ? Math.pow(YminAndYmaxHalfDiff / (k * Math.sqrt(T)), 2)
                : Math.pow(3 / (k * Math.sqrt(T)), 2);
    }

    private void calculateLambda(double sample_var_y) {
        double ten_pctile_chisq_df_hyper_nu = 0;
        ChiSquaredDistribution chi_sq_dist = new ChiSquaredDistribution(nu);
        try {
            ten_pctile_chisq_df_hyper_nu = chi_sq_dist.inverseCumulativeProbability(1 - q);
        } catch (MathRuntimeException e) {
            System.err.println("Could not calculate inverse cum prob density for chi sq df = " + nu + " with q = " + q);
            System.exit(0);
        }
        this. lambda = ten_pctile_chisq_df_hyper_nu / nu * sample_var_y;
    }

    public void calculateExtendedHyperparams(double sample_var_y) {
        calculateSigmaMu();
        calculateLambda(sample_var_y);
    }

}
