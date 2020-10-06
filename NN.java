package p1;

//import com.sun.org.apache.bcel.internal.generic.NEW;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.lang.Math;
import java.text.DecimalFormat;



// Todo: you need to change the activation function from relu (current) version to logistic, remember, not only the activation function, but the weight update part as well.

public class NN {

    // Todo: change hyper-parameters below, like MAX_EPOCHS, learning_rate, etc.

    private static final String COMMA_DELIMITER = ",";
    private static final String PATH_TO_TRAIN = "./mnist_train.csv";
    private static final String PATH_TO_TEST = "./mnist_test.csv";
    private static final String NEW_TEST = "./test.txt";
    private static final int MAX_EPOCHS = 1;
    static Double learning_rate = 0.1;

    static Double[][] wih = new Double[392][785];
    static Double[] who = new Double[393];

    static String first_digit = "5";
    static String second_digit = "6";
    static Random rng = new Random();


    public static double[][] parseRecords(String file_path) throws FileNotFoundException, IOException {
        double[][] records = new double[20000][786];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {

                String[] string_values = line.split(COMMA_DELIMITER);
                if (!string_values[0].equals(first_digit) && !string_values[0].contentEquals(second_digit)) continue;
                if (first_digit.equals(string_values[0])) records[k][0] = 0.0; // label 0
                else records[k][0] = 1.0; // label 1
                for (int i = 1; i < string_values.length; i++) {
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                }
                records[k][785] = 1.0;

                k += 1;
            }

            double[][] res = new double[k][786];
            for (int i= 0; i < k; i ++){
                System.arraycopy(records[i], 0, res[i], 0, 786);
            }
            return res;
        }

    }


    public static double[][] NewTest(String file_path) throws FileNotFoundException, IOException {
        double[][] records = new double[20000][785];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {

                String[] string_values = line.split(COMMA_DELIMITER);
                for (int i = 0; i < string_values.length; i++) {
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                }
                records[k][784] = 1.0;

                k += 1;
            }

            double[][] res = new double[k][785];
            for (int i= 0; i < k; i ++){
                System.arraycopy(records[i], 0, res[i], 0, 785);
            }
            return res;
        }

    }

    public static double logistic(double z) {
    	return 1.0 / (1.0 + +Math.exp(-z));
    }
    
    public static double firstLayer(double[] row) {
    	Double[] output = new Double[392];
    	for(int i = 0; i < wih.length; i++) {
    		Double sum = 0.0;
    		for(int j = 0; j < wih[0].length-1; j++) {
    			sum += row[j] * wih[i][j];
    		}
    		sum += wih[i][wih[0].length-1];
    		output[i] = logistic(sum);
    		
    	}
    	
    	double sum = 0;
    	for(int x = 0; x < output.length; x++) {
    		sum += output[x] * who[x];
    	}
    	return logistic(sum);
    }

    public static void main(String[] args) throws IOException {
        double[][] train = parseRecords(PATH_TO_TRAIN);
        double[][] test = parseRecords(PATH_TO_TEST);

        double[][] new_test = NewTest(NEW_TEST);


        int num_train = train.length;
        int num_test = test.length;

        for(int i = 0; i < wih.length; i ++){
            for (int j = 0; j < wih[0].length; j++){
                wih[i][j] = 2 * rng.nextDouble() - 1;
            }
        }
        for(int i = 0; i < who.length; i ++){
            who[i] = 2 * rng.nextDouble() - 1;
        }


        for(int epoch = 1; epoch <= MAX_EPOCHS; epoch++ ){
            double[] out_o = new double[num_train];
            double[][] out_h = new double[num_train][393];
            for(int i = 0; i < num_train; ++ i)
                out_h[i][392] = 1.0;

            for(int ind = 0; ind < num_train; ++ ind){
                double[] row = train[ind];
                double label = row[0];


                //calc out_h[ind, :-1]
                for(int i = 0; i < 392; ++ i) {
                    double s = 0.0;
                    for (int j = 0; j < 785; ++j) {
                        s += wih[i][j] * row[j+1];
                    }
                    out_h[ind][i] = logistic(s);
                }

                //calc out_o[ind]
                double s = 0.0;
                for(int i = 0; i < 393; ++ i){
                    s += out_h[ind][i] * who[i];
                }
                out_o[ind] = 1.0 / (1.0 + Math.exp(-s));

                //calc delta
                double[] delta = new double[393];
                for(int i = 0; i < 393; ++i){
                    delta[i] = logistic(out_h[ind][i]) * who[i] * (label - out_o[ind]);
                }

                //update wih
                for(int i = 0; i < 392; ++i){
                    for(int j = 0; j < 785; ++ j){
                        wih[i][j] += learning_rate * delta[i] * row[j+1];
                    }
                }

                //update who
                for(int i = 0; i < 393; ++ i){
                    who[i] += learning_rate * (label - out_o[ind]) * out_h[ind][i];
                }
            }


            //calc error
            double error = 0;
            for(int ind = 0; ind < num_train; ind ++){
                double[] row = train[ind];
                error += -row[0] * Math.log(out_o[ind]) - (1-row[0]) * Math.log(1- out_o[ind]);
            }

            //correct
            double correct = 0.0;
            for(int ind = 0; ind < num_train; ind ++){
                if ((train[ind][0] == 1.0 && out_o[ind] >=0.5) || (train[ind][0] ==0.0 && out_o[ind] < 0.5) )
                    correct += 1.0;
            }
            double acc = correct / num_train;

            System.out.println("Epoch: " + epoch + ", error: " + error + ", acc: " + acc);

        }
        
        // I received help on this portion from another student in the class. We wroked together to solve the problem. 
        
        DecimalFormat format1 = new DecimalFormat("##.##");
		DecimalFormat format2 = new DecimalFormat("##.####");

		

		Double prediction[] = new Double[new_test.length];
															

		for (int i = 0; i < prediction.length; i++) {
			prediction[i] = firstLayer(new_test[i]);
		}

		for (int i = 0; i < prediction.length; i++) {
			System.out.print(format1.format(prediction[i]) + ",");
		}
		
		PrintWriter writer = new PrintWriter("Q5.txt");

		for (int i = 0; i < wih[0].length; i++) {

			for (int k = 0; k < wih.length; k++) {
				if (k == wih.length - 1) {
					writer.print(format2.format(wih[k][i]));
				} else
					writer.print(format2.format(wih[k][i]) + ", ");

			}
			writer.println();

		}
		System.out.println();
		// q6
		PrintWriter writer2 = new PrintWriter("Q6.txt");
		for (int i = 0; i < who.length; i++) {
			if (i == who.length - 1)
				writer2.print(format2.format(who[i]));
			else
				writer2.print(format2.format(who[i]) + ", ");
		}
		writer2.close();
		writer.close();

		writer.close();
		
		for(Double p : prediction) {
			if(p < 0.5) {
				System.out.print("0, ");
			}
			else {
				System.out.print("1, ");
			}
		}



    }



}

