package BARTMultiEdit;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

public abstract class BartBase {

    Hyperparam hyperParam;
    DataContext dataContext;
    Random rand;

    int currentGibbsIteration;
    BTreeNode[][] gibbsSamplesOfTrees;
    double[]  gibbsSamplesOfSigmaSq;
    boolean[][] acceptRejectMH;

    transient double[] sumEstimatesVector; // cached current sum of estimation vector

    public BartBase(Hyperparam hyperParam, DataContext dataContext) {
        this.hyperParam = hyperParam;
        this.dataContext = dataContext;
        this.rand = new Random(hyperParam.seed);
        this.currentGibbsIteration = 0;
        this.gibbsSamplesOfTrees = new BTreeNode[hyperParam.numGibbsTotal+1][hyperParam.T];
        this.gibbsSamplesOfSigmaSq = new double[hyperParam.numGibbsTotal+1];
        this.acceptRejectMH = new boolean[hyperParam.numGibbsTotal+1][hyperParam.T];
    }

    public void initialize() {
        RandomCache.populateRandomSampleCache(10000, hyperParam.seed, (int)(hyperParam.nu+ dataContext.N));
        var sample_var_response = calculate_sample_var_ytrans(dataContext.X, dataContext.yTransformed);
        this.hyperParam.calculateExtendedHyperparams(sample_var_response);
        initializeTreesAndMus();
        initializeSigmaSq();
        this.sumEstimatesVector = new double[dataContext.N]; // with default value 0.
    }

    public void doGibbsSampling(){
        var start_time = Instant.now();
        while (currentGibbsIteration <= hyperParam.numGibbsTotal){
            doOneGibbsSampling();
            for(var tree : gibbsSamplesOfTrees[currentGibbsIteration-1])
                tree.flushBartData();

            if(currentGibbsIteration % 100 == 0) {
                var loss_so_far = this.getInSampleLossToCurrentIteration();
                System.out.println(String.format("Iteration: %d\tInSample Loss: %2f", currentGibbsIteration, loss_so_far));
            }

            currentGibbsIteration++;
        }
        var end_time = Instant.now();
        var training_duration = Duration.between(start_time, end_time).toSeconds();
        System.out.println(String.format("Total training time: %d seconds", training_duration));
    }

    abstract void initializeSigmaSq();

    public abstract double getPredictionsFromGibbsTreeSamples(double[] record);

    abstract double[] getInSamplePredictionToCurrentIteration();

    abstract double getInSampleLossToCurrentIteration();

    private void initializeTreesAndMus() {
        for(int i = 0; i <  hyperParam.T; i++) {
            var init_tree_root = BTreeNode.createStump(rand, dataContext, hyperParam);
            for(var terminal : init_tree_root.getTerminalsBelowInclusive())
                terminal.initializeMu();
            gibbsSamplesOfTrees[0][i] = init_tree_root;
        }
        currentGibbsIteration = 1;
    }

    private BTreeNode runMHSamplingForNewTree(int gibbs_iter, int tree_idx, BTreeNode old_tree, double[] R_minus_j, double σ_sq) {

        // we save a shallow copy of the current tree in case MH rejects the proposal
        var copy_of_tree = old_tree.clone();
        var proposal_tree = copy_of_tree.clone();
        var proposal =
            switch (hyperParam.mhMode) {
                case OneStep -> BTreeEdit.performOneStepRandomWalk(rand, old_tree, proposal_tree, R_minus_j);
                case MultiStep -> BTreeEdit.performMultiStepRandomWalk(rand, old_tree, proposal_tree, hyperParam.meanStride, R_minus_j); // to be implemented
            };

        proposal_tree = proposal.getValue0();
        var log_forward_backward = proposal.getValue1();
        var log_transition_ratio = log_forward_backward[1] - log_forward_backward[0];

        if(proposal_tree != null && proposal_tree.tryPopulateDataAndDepth(R_minus_j)) {
            var log_likelihood_ratio = BTreeProb.getTreeLogLikelihoodRatio(proposal_tree, old_tree, σ_sq, hyperParam.σ_mu_sq);
            var log_structure_ratio = BTreeProb.getTreeStructureLogRatio(proposal_tree, old_tree);
            var log_metropolis_ratio = log_likelihood_ratio + log_transition_ratio + log_structure_ratio;
            var log_u_0_1 = Math.log(rand.nextDouble());

            if(log_metropolis_ratio > log_u_0_1) {
                // accept
                this.acceptRejectMH[gibbs_iter][tree_idx] = true;
                return proposal_tree;
            }
        }

        this.acceptRejectMH[gibbs_iter][tree_idx] = false;
        copy_of_tree.tryPopulateDataAndDepth(R_minus_j); // must succeed as the old tree has a valid structure
        return copy_of_tree;
    }

    private double[] doOneMHTreeSampling(int gibbs_iter, int tree_idx, double σ_sq) {
        // try to sample a new tree and the corresponding mus on its terminal nodes
        // 1. we get the predictions from the same tree in last iteration
        var tree_j_old = gibbsSamplesOfTrees[gibbs_iter-1][tree_idx];
        var pred_j_old = BTreeNode.getInSamplePredictions(tree_j_old);
        // 2. we create the response variable values this tree is supposed to see
        double[] R_minus_j = VectorTools.add_arrays(VectorTools.subtract_arrays(dataContext.yTransformed, sumEstimatesVector), pred_j_old);
        // 3. run Metropolis-Hasting to sample a new tree with R_j as the new tree's response variable
        var sampled_tree = runMHSamplingForNewTree(gibbs_iter, tree_idx, tree_j_old, R_minus_j, σ_sq);
        sampled_tree.getTerminalsBelowInclusive().forEach(t -> t.sampleMu(rand, σ_sq));
        var pred_j_new = BTreeNode.getInSamplePredictions(sampled_tree);

        // after the j-th tree is sampled (be it old or new),
        // we need to replace the prediction of the j-th tree from previous iteration with the ones from this iteration
        // (they can be the same between iterations, e.g. when the proposal tree is rejected
        // technically, we pull 'pred_j_old' out from sum_estimates_vec and plug 'pred_j_new' in
        sumEstimatesVector = VectorTools.subtract_arrays(sumEstimatesVector, pred_j_old);
        sumEstimatesVector = VectorTools.add_arrays(sumEstimatesVector, pred_j_new);
        gibbsSamplesOfTrees[gibbs_iter][tree_idx] = sampled_tree;

        var residual = VectorTools.subtract_arrays(dataContext.yTransformed, sumEstimatesVector);
        return residual;
    }

    private void doSamplingForSigmaSq(int gibbs_sample_iter, double[] full_residual) {
        //first calculate the SSE
        double sse = 0;
        for (double e : full_residual) {
            sse += e * e;
        }
        double nu = hyperParam.nu;
        double lambda = hyperParam.lambda;
        //we're sampling from σ_sq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
        //which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
        double new_σ_sq =
                StatToolbox.sample_from_inv_gamma((nu + full_residual.length) / 2,
                        2 / (sse + nu * lambda), this.rand);

        gibbsSamplesOfSigmaSq[gibbs_sample_iter] = new_σ_sq;
    }

    private void doOneGibbsSampling(){
        // 1. we get the σ² to be used to sample trees for this iteration
        var σ_sq = gibbsSamplesOfSigmaSq[currentGibbsIteration - 1];
        // this is the residual after all T trees with their mus are sampled for this iteration
        var full_model_residuals = new double[dataContext.N];
        // 2. we sample T trees and their respective mus
        for(int j = 0; j < hyperParam.T; j++) {
            // keep updating full_model_residuals as trees are sampled
            full_model_residuals = doOneMHTreeSampling(currentGibbsIteration, j, σ_sq);
        }
        // 3.now we have the full residual vector which we pass on to sample a new σ²
        doSamplingForSigmaSq(currentGibbsIteration, full_model_residuals);
    }

    private static double calculate_sample_var_ytrans(double[][] X, double[] y_trans) {
        var N = X.length;
        var m = X[0].length;
        if (N > m) {
            // the majority case, run a multivariate linear regression and return the Mean Squared Error

            // TODO - we need to process X and y (probably by omitting) such that
            //  X_for_regression does not have any missing value
            var X_for_regression = X;
            var y_trans_for_regression = y_trans;
            var reg = new OLSMultipleLinearRegression();
            reg.newSampleData(y_trans_for_regression, X_for_regression);
            var residuals = reg.estimateResiduals();
            var mse = StatToolbox.sample_variance(residuals);
            return mse;
        }

        // otherwise (the rare cases)
        return StatToolbox.sample_variance(y_trans);
    }

}
