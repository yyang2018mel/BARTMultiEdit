package DataUtils;


import BARTMultiEdit.DataContext;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

public class DataUtils {

    public static DataContext readCsv(File file, boolean has_header, boolean is_regression) throws IOException {

        ArrayList<double[]> X = new ArrayList<>();
        ArrayList<Double> y = new ArrayList<>();
        ArrayList<String> header = new ArrayList<>();

        //begin by iterating over the file
        BufferedReader in = new BufferedReader(new FileReader(file));
        int line_num = 0;
        while (true){
            String datum = in.readLine();
            if (datum == null) {
                break;
            }
            String[] datums = datum.split(",");
            if (line_num == 0 && has_header){
                for (int i = 0; i < datums.length-1; i++){
                    header.add(datums[i]); //default for now
                }
            }
            else {
                var record = IntStream.range(0, datums.length).boxed()
                        .map(i -> {
                            try {
                                return Double.parseDouble(datums[i]);
                            } catch (NumberFormatException e) {
                                return Double.NaN;
                            }
                        }).mapToDouble(e -> e).toArray();
                X.add(Arrays.copyOfRange(record,0, record.length-1));
                y.add(record[record.length-1]);
            }
            line_num++;
        }
        in.close();
        return new DataContext(X.toArray(double[][]::new), y.stream().mapToDouble(e->e).toArray(), is_regression);
    }
}


