package gr.angelo.spark;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;

public class Configs {
    String hadoopPath;
    String csvpath;
    String pipepath;
    String modelpath;
    String reviewText;
    String reviewResult;
    Configuration config;

    public Configs() {
        try {
            config = new PropertiesConfiguration("config.properties");
            hadoopPath = config.getString("hadoop.path");
            csvpath = config.getString("csv.path");
            pipepath = config.getString("pipemodel.save.path");
            modelpath = config.getString("trainedmodel.save.path");
            reviewText = config.getString("testReview.text");
            reviewResult = config.getString("testReview.result");
        } catch (ConfigurationException e) {
            e.printStackTrace();
        }
    }

    public String getCsvpath() {
        return csvpath;
    }

    public void setCsvpath(String csvpath) {
        this.csvpath = csvpath;
    }

    public String getPipepath() {
        return pipepath;
    }

    public void setPipepath(String pipepath) {
        this.pipepath = pipepath;
    }

    public String getModelpath() {
        return modelpath;
    }

    public void setModelpath(String modelpath) {
        this.modelpath = modelpath;
    }

    public Configuration getConfig() {
        return config;
    }

    public void setConfig(Configuration config) {
        this.config = config;
    }

    public String getReviewText() {
        return reviewText;
    }

    public void setReviewText(String reviewText) {
        this.reviewText = reviewText;
    }

    public String getReviewResult() {
        return reviewResult;
    }

    public void setReviewResult(String reviewResult) {
        this.reviewResult = reviewResult;
    }

    public String getHadoopPath() {
        return hadoopPath;
    }

    public void setHadoopPath(String hadoopPath) {
        this.hadoopPath = hadoopPath;
    }
}
