package BARTMultiEdit;

import java.util.Arrays;

public class VectorTools {

    /**
     * Subtracts one array from the other
     *
     * @param arr1	The array of minuends
     * @param arr2	The array of subtrahends
     * @return		The array of differences
     */
    public static double[] subtract_arrays(double[] arr1, double[] arr2) {
        int n = arr1.length;
        double[] diff = new double[n];
        for (int i = 0; i < n; i++){
            diff[i] = arr1[i] - arr2[i];
        }
        return diff;
    }

    /**
     * Adds one array to another
     *
     * @param arr1	The array of first addends
     * @param arr2	The array of seconds addends
     * @return		The array of sums
     */
    public static double[] add_arrays(double[] arr1, double[] arr2) {
        int n = arr1.length;
        double[] sum = new double[n];
        for (int i = 0; i < n; i++){
            sum[i] = arr1[i] + arr2[i];
        }
        return sum;
    }

    /**
     * Returns the max of a vector
     *
     * @param values	The values of interest
     * @return			The maximum of those values
     */
    public static double max(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (value > max){
                max = value;
            }
        }
        return max;
    }

    /**
     * Returns if the vector contains 0s only
     *
     * @param values	The values of interest
     * @return			Whether or not the vector is [0,...,0]
     */
    public static boolean isZeroVec(double[] values) {
        return Arrays.stream(values).allMatch(d -> d == 0.);
    }

    public static double[][] transformMatrix(double[][] matrix) {
        var n_row = matrix.length;
        var n_col = matrix[0].length;
        var transformed = new double[n_col][n_row];
        for(int i = 0; i < n_row; i++)
            for(int j = 0; j < n_col; j++)
                transformed[j][i] = matrix[i][j];
        return transformed;
    }

}