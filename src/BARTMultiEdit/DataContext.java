package BARTMultiEdit;

import javax.swing.*;
import java.util.Arrays;

public class DataContext {

    int N; // number of records in the dataset
    int m; // number of features per record

    double[][] X;
    double[][] XByColumn;
    double[] y;
    double[] yTransformed;
    String[] headers;

    public DataContext(double[][] X, double[] y, boolean is_regression) {
        this.X = X;
        this.y = y;
        this.XByColumn = VectorTools.transformMatrix(X);
        this.N = X.length;
        this.m = X[0].length;
        this.yTransformed =
            is_regression
            ? Arrays.stream(y).map(y_i -> transformYForRegression(y_i)).toArray()
            : y;
    }

    double transformYForRegression(double y_i) {
        var y_min = StatToolbox.sample_minimum(y);
        var y_max = StatToolbox.sample_maximum(y);
        var y_min_and_max_half_diff = Hyperparam.YminAndYmaxHalfDiff;
        return (y_i - y_min) / (y_max - y_min) - y_min_and_max_half_diff;
    }

    double restoreYForRegression(double y_trans_i) {
        var y_min = StatToolbox.sample_minimum(y);
        var y_max = StatToolbox.sample_maximum(y);
        var y_min_and_max_half_diff = Hyperparam.YminAndYmaxHalfDiff;
        return (y_trans_i + y_min_and_max_half_diff) * (y_max - y_min) + y_min;
    }

}
