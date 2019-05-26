package gr.angelo.spark;

import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.util.DefaultParamsWritable;
import org.apache.spark.mllib.feature.Stemmer;

public class MyStemmer extends Stemmer implements HasInputCol, HasOutputCol, DefaultParamsWritable {
    @Override
    public MyStemmer setLanguage(String value) {
        super.setLanguage(value);
        return this;
    }

    @Override
    public MyStemmer setInputCol(String value) {
        super.setInputCol(value);
        return this;
    }

    @Override
    public MyStemmer setOutputCol(String value) {
        super.setOutputCol(value);
        return this;
    }
}
