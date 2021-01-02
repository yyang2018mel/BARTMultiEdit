package BARTMultiEdit;

import java.util.Arrays;
import java.util.stream.IntStream;

public class BartRegression extends BartBase {

    public BartRegression(Hyperparam hyperParam, DataContext dataContext) {
        super(hyperParam, dataContext);
    }

    @Override
    void initializeSigmaSq() {
        double nu = hyperParam.nu;
        double lambda = hyperParam.lambda;
        this.gibbsSamplesOfSigmaSq[0] = StatToolbox.sample_from_inv_gamma(nu / 2, 2 / (nu * lambda), this.rand);
    }

    @Override
    public double[] getPredictionsFromGibbsTreeSamples(double[][] records, boolean till_current_iteration) {
        int iteration_end = till_current_iteration ? currentGibbsIteration : hyperParam.numGibbsTotal;
        int num_post_burn_in = iteration_end - hyperParam.numGibbsBurnin;
        int n = records.length;
        double[] result = new double[n];

        if (num_post_burn_in <= 0) {
            System.out.println("BART model still in burn-in period; returning 0-vector.");
            return result;
        }

        double[][] ys_trans_gibbs_sample = new double[num_post_burn_in][n];
        for(int g = hyperParam.numGibbsBurnin; g < iteration_end; g++) {
            double[] preds_trans_g = new double[n];
            var bart_trees = gibbsSamplesOfTrees[g];
            for(var tree : bart_trees) {
                for(int i = 0; i < n; i++) {
                    var record = records[i];
                    preds_trans_g[i] += tree.getPredictionForData(record);
                }
            }
            ys_trans_gibbs_sample[g-hyperParam.numGibbsBurnin] = preds_trans_g;
        }

        var ys_trans_gibbs_sample_T = VectorTools.transformMatrix(ys_trans_gibbs_sample);
        for(int i = 0; i < n; i++) {
            result[i] = dataContext.restoreYForRegression(
                        StatToolbox.sample_average(ys_trans_gibbs_sample_T[i]));
        }

        return result;
    }

    @Override
    double getInSampleLossToCurrentIteration() {
        var insample_predictions = getPredictionsFromGibbsTreeSamples(dataContext.X, true);
        var l2 = (IntStream.range(0, dataContext.N).boxed()
                .mapToDouble(i -> Math.pow(insample_predictions[i]-dataContext.y[i], 2))
                .sum())/ dataContext.N;
        return Math.sqrt(l2);
    }
}
