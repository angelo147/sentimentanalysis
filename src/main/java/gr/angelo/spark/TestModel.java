package gr.angelo.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

public class TestModel {
    public static void main(String[] args) {
        Configs configs = new Configs();
        System.setProperty("hadoop.home.dir", configs.getHadoopPath());
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaTfIdfExample")
                .config("spark.master", "local")
                .getOrCreate();

        List<Row> data = Arrays.asList(RowFactory.create(55, "", configs.getReviewText(), configs.getReviewResult(), "file44.txt"));

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("type", DataTypes.StringType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty()),
                new StructField("result", DataTypes.StringType, false, Metadata.empty()),
                new StructField("file", DataTypes.StringType, false, Metadata.empty())
        });
        Dataset<Row> sentenceData = spark.createDataFrame(data, schema);
        PipelineModel model2 = PipelineModel.load("file:///"+configs.getPipepath());
        Dataset<Row> mytest = model2.transform(sentenceData);

        LinearSVCModel lsvcModel = LinearSVCModel.load("file:///"+configs.getModelpath());

        Dataset<Row> predictions = lsvcModel.transform(mytest);
        predictions.show();
    }
}
