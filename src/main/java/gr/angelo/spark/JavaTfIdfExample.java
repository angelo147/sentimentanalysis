package gr.angelo.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.*;

public class JavaTfIdfExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaTfIdfExample")
                .config("spark.master", "local")
                .getOrCreate();

        StructType schemab = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("type", DataTypes.StringType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty()),
                new StructField("label", DataTypes.StringType, false, Metadata.empty()),
                new StructField("file", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> dataFrame =
                spark.read().format("csv").schema(schemab).option("header", "true").load("C:\\Users\\Aggelos\\Desktop\\projects\\sentimentanalysis\\imdb_master.csv").filter(col("label").as("String").equalTo("neg").or(col("label").as("String").equalTo("pos")));

        UserDefinedFunction cleanUDF = udf(
                (String strVal) -> strVal.replaceAll("[\"]", ""), DataTypes.StringType
        );

        Dataset<Row> changed = dataFrame.withColumn("label",when(dataFrame.col("label").as("String").equalTo("neg"), 0).otherwise(1)).withColumn("sentence", cleanUDF.apply(col("sentence")));
        //changed.show();

        //Tokenize
        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsDataa = tokenizer.transform(changed);
        //Remove StopWords
        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered");

        Dataset<Row> wordsData = remover.transform(wordsDataa);
        //English stemming
        Stemmer stemmer = new Stemmer().setInputCol("filtered").setOutputCol("stemmed").setLanguage("English");
        Dataset<Row> wordData = stemmer.transform(wordsData);

        List<String[]> listOne = wordData.select("stemmed").collectAsList().stream().map(r->r.getList(0)).collect(Collectors.toList()).stream().map(e->e.toArray(new String[0])).collect(Collectors.toList());
        int count = (int) listOne.stream().flatMap(Arrays::stream).distinct().count();

        HashingTF hashingTF = new HashingTF()
                .setInputCol("stemmed")
                .setOutputCol("rawFeatures")
                .setNumFeatures(count);

        CountVectorizerModel cvModel = new CountVectorizer()
                .setInputCol("stemmed")
                .setOutputCol("rawFeatures")
                //.setVocabSize(5)
                //.setMinDF(2)
                .fit(wordData);

        Dataset<Row> featurizedData = cvModel.transform(wordData);
        // alternatively, CountVectorizer can also be used to get term frequency vectors

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        rescaledData.select( "label", "features").show();
        // $example off$

        Dataset<Row>[] splits = rescaledData.randomSplit(new double[]{0.9, 0.1}, 1234L);
        Dataset<Row> train = splits[0];//rescaledData.filter(col("type").as("String").equalTo("train"));
        Dataset<Row> test = splits[1];//rescaledData.filter(col("type").as("String").equalTo("test"));
        // create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();//.setModelType("bernoulli");

        // train the model
        NaiveBayesModel model = nb.fit(train);

        // Select example rows to display.
        Dataset<Row> predictions = model.transform(test);
        predictions.show(100); //.where(col("prediction").equalTo(0.0))

        // compute accuracy on the test set
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        //BinaryClassificationEvaluator ev = new BinaryClassificationEvaluator();
        //ev.setLabelCol("label").setRawPredictionCol("rawPrediction");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy = " + accuracy);

        /*LinearSVC lsvc = new LinearSVC()
                .setMaxIter(10)
                .setRegParam(0.1);*/

        // Fit the model
        /*LinearSVCModel lsvcModel = lsvc.fit(train);

        // Print the coefficients and intercept for LinearSVC
        System.out.println("Coefficients: "
                + lsvcModel.coefficients() + " Intercept: " + lsvcModel.intercept());
        Dataset<Row> predictionss = lsvcModel.transform(test);
        predictionss.show();
        double accuracyy = evaluator.evaluate(predictionss);
        System.out.println("Test set accuracy = " + accuracyy);*/
        spark.stop();
    }
}
