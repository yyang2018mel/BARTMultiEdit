package BARTMultiEdit;

import java.util.Random;

import static OpenSourceExtensions.StatUtil.refine;
import org.apache.commons.math3.distribution.PoissonDistribution;

public class StatToolbox {
    /* ********************************************
     * Original algorythm and Perl implementation can
     * be found at:
     * http://www.math.uio.no/~jacklam/notes/invnorm/index.html
     * Author:
     *  Peter John Acklam
     *  jacklam@math.uio.no
     * ****************************************** */
    private static final double P_LOW  = 0.02425D;
    private static final double P_HIGH = 1.0D - P_LOW;

    // Coefficients in rational approximations.
    private static final double ICDF_A[] =
            { -3.969683028665376e+01,  2.209460984245205e+02,
                    -2.759285104469687e+02,  1.383577518672690e+02,
                    -3.066479806614716e+01,  2.506628277459239e+00 };

    private static final double ICDF_B[] =
            { -5.447609879822406e+01,  1.615858368580409e+02,
                    -1.556989798598866e+02,  6.680131188771972e+01,
                    -1.328068155288572e+01 };

    private static final double ICDF_C[] =
            { -7.784894002430293e-03, -3.223964580411365e-01,
                    -2.400758277161838e+00, -2.549732539343734e+00,
                    4.374664141464968e+00,  2.938163982698783e+00 };

    private static final double ICDF_D[] =
            { 7.784695709041462e-03,  3.224671290700398e-01,
                    2.445134137142996e+00,  3.754408661907416e+00 };

    public static double getInvCDF(double d, boolean highPrecision)
    {
        // Define break-points.
        // variable for result
        double z = 0;

        if(d == 0) z = Double.NEGATIVE_INFINITY;
        else if(d == 1) z = Double.POSITIVE_INFINITY;
        else if(Double.isNaN(d) || d < 0 || d > 1) z = Double.NaN;

            // Rational approximation for lower region:
        else if( d < P_LOW )
        {
            double q  = Math.sqrt(-2*Math.log(d));
            z = (((((ICDF_C[0]*q+ICDF_C[1])*q+ICDF_C[2])*q+ICDF_C[3])*q+ICDF_C[4])*q+ICDF_C[5]) / ((((ICDF_D[0]*q+ICDF_D[1])*q+ICDF_D[2])*q+ICDF_D[3])*q+1);
        }

        // Rational approximation for upper region:
        else if ( P_HIGH < d )
        {
            double q  = Math.sqrt(-2*Math.log(1-d));
            z = -(((((ICDF_C[0]*q+ICDF_C[1])*q+ICDF_C[2])*q+ICDF_C[3])*q+ICDF_C[4])*q+ICDF_C[5]) / ((((ICDF_D[0]*q+ICDF_D[1])*q+ICDF_D[2])*q+ICDF_D[3])*q+1);
        }
        // Rational approximation for central region:
        else
        {
            double q = d - 0.5D;
            double r = q * q;
            z = (((((ICDF_A[0]*r+ICDF_A[1])*r+ICDF_A[2])*r+ICDF_A[3])*r+ICDF_A[4])*r+ICDF_A[5])*q / (((((ICDF_B[0]*r+ICDF_B[1])*r+ICDF_B[2])*r+ICDF_B[3])*r+ICDF_B[4])*r+1);
        }
        if(highPrecision) z = refine(z, d);
        return z;
    }

    /** A flag that indicates an illegal value or failed operation */
    public static final double ILLEGAL_FLAG = -999999999;

    /**
     * Draws a sample from an inverse gamma distribution.
     *
     * @param k			The shape parameter of the inverse gamma distribution of interest
     * @param theta		The scale parameter of the inverse gamma distribution of interest
     * @return			The sampled value
     */
    public static double sample_from_inv_gamma(double k, double theta, Random rand){
        return (1 / (theta / 2)) / RandomCache.chiSquaredSamples[(int)Math.floor(rand.nextDouble()
                * RandomCache.chiSquaredSamples.length)];
    }

    /**
     * Compute the sample variance of a vector of data
     *
     * @param y	The vector of data values
     * @return	The sample variance
     */
    public static final double sample_variance(double[] y){
        return sample_sum_sq_err(y) / ((double)y.length - 1);
    }

    /**
     * Compute the sum of squared error (the squared deviation from the sample average) of a vector of data
     *
     * @param y	The vector of data values
     * @return	The sum of squared error
     */
    public static final double sample_sum_sq_err(double[] y){
        double y_bar = sample_average(y);
        double sum_sqd_deviations = 0;
        for (int i = 0; i < y.length; i++){
            sum_sqd_deviations += Math.pow(y[i] - y_bar, 2);
        }
        return sum_sqd_deviations;
    }

    /**
     * Compute the sample average of a vector of data
     *
     * @param y	The vector of data values
     * @return	The sample average
     */
    public static final double sample_average(double[] y){
        double y_bar = 0;
        for (int i = 0; i < y.length; i++){
            y_bar += y[i];
        }
        return y_bar / (double)y.length;
    }

    /**
     * Draws a sample from a normal distribution.
     *
     * @param mu		The mean of the normal distribution of interest
     * @param σ_sq		The variance of the normal distribution of interest
     * @return			The sample value
     */
    public static double sample_from_norm_dist(double mu, double σ_sq, Random rand){
        double std_norm_realization = RandomCache.stdNormalSamples[(int)Math.floor(rand.nextDouble() * RandomCache.stdNormalSamples.length)];
        return mu + Math.sqrt(σ_sq) * std_norm_realization;
    }

    // constants for the {@link normal_cdf} function
    private static double NORM_CDF_a1 =  0.254829592;
    private static double NORM_CDF_a2 = -0.284496736;
    private static double NORM_CDF_a3 =  1.421413741;
    private static double NORM_CDF_a4 = -1.453152027;
    private static double NORM_CDF_a5 =  1.061405429;
    private static double NORM_CDF_p  =  0.3275911;

    /**
     * Calculate the cumulative density under a standard normal to a point of interest.
     *
     * @param x	The point of interest on the standard normal density support
     * @return	The probability of interest
     *
     */
    public static double normal_cdf(double x) {
        // Save the sign of x
        int sign = 1;
        if (x < 0){
            sign = -1;
        }
        x = Math.abs(x) / Math.sqrt(2.0);

        // A&S formula 7.1.26
        double t = 1.0 / (1.0 + NORM_CDF_p * x);
        double y = 1.0 - (((((NORM_CDF_a5 * t + NORM_CDF_a4) * t) + NORM_CDF_a3) * t + NORM_CDF_a2) * t + NORM_CDF_a1) * t * Math.exp(-x * x);

        return 0.5 * (1.0 + sign * y);
    }

    /**
     * Compute the sample minimum of a vector of data
     *
     * @param y	The vector of data values
     * @return	The sample minimum
     */
    public static double sample_minimum(double[] y){
        double min = Double.MAX_VALUE;
        for (double y_i : y){
            if (y_i < min){
                min = y_i;
            }
        }
        return min;
    }

    /**
     * Compute the sample maximum of a vector of data
     *
     * @param y	The vector of data values
     * @return	The sample maximum
     */
    public static double sample_maximum(double[] y){
        double max = Double.NEGATIVE_INFINITY;
        for (double y_i : y){
            if (y_i > max){
                max = y_i;
            }
        }
        return max;
    }

    public static int sample_poisson(double mean) {
        var poisson = new PoissonDistribution(mean);
        return poisson.sample();
    }

}