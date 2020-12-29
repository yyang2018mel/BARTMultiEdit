package BARTMultiEdit;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import java.util.Random;
import java.util.stream.IntStream;

class RandomCache {

    /** A cached library of chi-squared with degrees of freedom nu plus n (used for Gibbs sampling the variance) */
    static double[] chiSquaredSamples;
    /** A cached library of standard normal values (used for Gibbs sampling the posterior means of the terminal nodes) */
    static double[] stdNormalSamples;

    static void populateRandomSampleCache(int size, int seed, int dof_chi_sq) {
        var rand = new Random(seed);
        var chi_sq = new ChiSquaredDistribution(dof_chi_sq);
        chi_sq.reseedRandomGenerator(seed);
        var normal_samples = IntStream.range(0, size).mapToDouble(i -> rand.nextGaussian()).toArray();
        var chi_sq_samples = chi_sq.sample(size);
        stdNormalSamples = normal_samples;
        chiSquaredSamples = chi_sq_samples;
    }
}
