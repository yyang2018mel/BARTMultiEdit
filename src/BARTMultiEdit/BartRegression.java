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
    double[] getPredictionsFromGibbsTreeSamples(double[] record) {

        int num_post_burn_in = hyperParam.numGibbsTotal - hyperParam.numGibbsBurnin;
        double[] y_gibbs_samples = new double[num_post_burn_in];
        for (int g = hyperParam.numGibbsBurnin+1; g < hyperParam.numGibbsTotal; g++) {
            var bart_trees = gibbsSamplesOfTrees[g];
            double yt_g = 0;
            for (var tree : bart_trees) {
                yt_g += tree.getPredictionForData(record);
            }
            y_gibbs_samples[g] = dataContext.restoreYForRegression(yt_g);
        }
        return y_gibbs_samples;

    }

    @Override
    double[] getInSamplePredictionToCurrentIteration() {
        int num_post_burnin_to_current = currentGibbsIteration - hyperParam.numGibbsBurnin;
        if (num_post_burnin_to_current <= 0) {
            return new double[dataContext.N];
        }

        double[][] ys_gibbs_sample = new double[num_post_burnin_to_current][dataContext.N];
        double[] ys_insample_avg = new double[dataContext.N];

        for(int g = hyperParam.numGibbsBurnin; g < currentGibbsIteration; g++) {
            var bart_trees = gibbsSamplesOfTrees[g];
            double[] ys_g = new double[dataContext.N];
            for (var tree : bart_trees) {
                var tree_predictions = new double[dataContext.N];
                for(int i = 0; i < dataContext.N; i++)
                    tree_predictions[i] = tree.getPrediction(dataContext.X[i]);
                ys_g = VectorTools.add_arrays(ys_g, tree_predictions);
            }
            ys_gibbs_sample[g-hyperParam.numGibbsBurnin] = Arrays.stream(ys_g).map(v -> dataContext.restoreYForRegression(v)).toArray();
        }
        var ys_gibbs_sample_T = VectorTools.transformMatrix(ys_gibbs_sample);
        for(int i = 0; i < dataContext.N; i++) {
            ys_insample_avg[i] = StatToolbox.sample_average(ys_gibbs_sample_T[i]);
        }
        return ys_insample_avg;
    }

    @Override
    double getInSampleLossToCurrentIteration() {
        var insample_predictions = getInSamplePredictionToCurrentIteration();
        var l2 = (IntStream.range(0, dataContext.N).boxed()
                .mapToDouble(i -> Math.pow(insample_predictions[i]-dataContext.y[i], 2))
                .sum())/ dataContext.N;
        return Math.sqrt(l2);
    }
}
