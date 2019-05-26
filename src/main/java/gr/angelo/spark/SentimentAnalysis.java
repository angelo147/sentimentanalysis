package gr.angelo.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;

public class SentimentAnalysis {
    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir","C:\\hadoop" );
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaTfIdfExample")
                .config("spark.master", "local")
                .getOrCreate();

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("type", DataTypes.StringType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty()),
                new StructField("result", DataTypes.StringType, false, Metadata.empty()),
                new StructField("file", DataTypes.StringType, false, Metadata.empty())
        });

        UserDefinedFunction cleanUDF = udf(
                (String strVal) -> strVal.replaceAll("[\"]", ""), DataTypes.StringType
        );

        Dataset<Row> dataFrame =
                spark.read().format("csv").schema(schema).option("header", "true").load("C:\\Users\\Aggelos\\Desktop\\projects\\sentimentanalysis\\imdb_master.csv").filter(col("result").as("String").equalTo("neg").or(col("result").as("String").equalTo("pos")))
                        .withColumn("sentence", cleanUDF.apply(col("sentence")));
                        //.withColumn("label",when(col("result").as("String").equalTo("neg"), 0).otherwise(1));

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        StringIndexer indexer = new StringIndexer()
                .setInputCol("result")
                .setOutputCol("label");
        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered");
        MyStemmer stemmer = new MyStemmer().setInputCol("filtered").setOutputCol("stemmed").setLanguage("English");//new Stemmer().setInputCol("filtered").setOutputCol("stemmed").setLanguage("English");
        CountVectorizer cv = new CountVectorizer()
                .setInputCol("stemmed")
                .setOutputCol("rawFeatures");
                //.setVocabSize(5)
        HashingTF hashingTF = new HashingTF()
                .setInputCol("stemmed")
                .setOutputCol("rawFeatures");
        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, indexer, remover, stemmer, cv, idf});

        PipelineModel model = pipeline.fit(dataFrame);
        Dataset<Row> rescaledData = model.transform(dataFrame);

        rescaledData.cache();
        //rescaledData.show();
        Dataset<Row>[] splits = rescaledData.randomSplit(new double[]{0.9, 0.1}, 1234L);
        Dataset<Row> train = splits[0];//rescaledData.filter(col("type").as("String").equalTo("train"));
        Dataset<Row> test = splits[1];//rescaledData.filter(col("type").as("String").equalTo("test"));

        BinaryClassificationEvaluator ev = new BinaryClassificationEvaluator();
        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(5)
                .setRegParam(0.1);
        LinearSVCModel lsvcModel = lsvc.fit(train);
        try {
            model.write().overwrite().save("file:///C:/Users/Aggelos/Desktop/projects/pipemodel");
            lsvcModel.write().overwrite().save("file:///C:/Users/Aggelos/Desktop/projects/model");
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Coefficients: "
                + lsvcModel.coefficients() + " Intercept: " + lsvcModel.intercept());

        /*NaiveBayes nb = new NaiveBayes().setSmoothing(0.000000001);//.setModelType("bernoulli");
        NaiveBayesModel nbmodel = nb.fit(train);*/

        Dataset<Row> predictions = lsvcModel.transform(test);
        predictions.show();
        double accuracyy = ev.evaluate(predictions);
        double v = predictions.filter(col("label").equalTo(col("prediction"))).count() / (double) test.count();
        System.out.println("Test set accuracy = " + accuracyy + " " + v);
        spark.stop();
    }
}
    /*ParamMap[] paramGrid = new ParamGridBuilder()
            .addGrid(lsvc.regParam(), new double[] {0.1, 0.01})
            .addGrid(lsvc.maxIter(), new int[] {5, 10})
            .build();
    TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
            .setEstimator(lsvc)
            .setEvaluator(new BinaryClassificationEvaluator())
            .setEstimatorParamMaps(paramGrid)
            .setTrainRatio(0.8)
            .setParallelism(1);
    TrainValidationSplitModel modell = trainValidationSplit.fit(train);
    Dataset<Row> predictionss = modell.transform(test);*/