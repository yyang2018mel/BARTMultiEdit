package BARTMultiEdit;

import DataUtils.DataUtils;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.*;

public class BartRegressionTest {

    @Test
    public void runRegressionModel() {
        try {
            var context = DataUtils.readCsv(new File("datasets", "r_rand.csv"), true, true);
            var params = new Hyperparam(2020, 50, 2000, 100, Hyperparam.MHMode.OneStep,
                                        0, false, .95, 2., 2., 3., .9, .3, .3,
                                         Hyperparam.VerboseLevel.NoReporting);
            var bart = new BartRegression(params, context);
            bart.initialize();
            bart.doGibbsSampling();
        }
        catch (Exception e) {
            System.out.println(e.getMessage());
        }


    }
}