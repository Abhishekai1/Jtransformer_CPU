package com.jtransformer.config;

import com.jtransformer.transformer.model.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Application configuration loader using application.properties
 */
public class AppConfig {

    private static final Logger logger = LoggerFactory.getLogger(AppConfig.class);
    private final Properties properties = new Properties();
    private ModelConfig modelConfig;
    private int epochs;
    private int batchSize;
    private double learningRate;

    public AppConfig() {
        loadProperties();
        this.modelConfig = createModelConfig();
        this.epochs = getIntProperty("training.epochs", 5);
        this.batchSize = getIntProperty("training.batch_size", 32);
        this.learningRate = getDoubleProperty("training.learning_rate", 0.001);
    }

    private void loadProperties() {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("application.properties")) {
            if (is != null) {
                properties.load(is);
                logger.info("Configuration loaded successfully");
            }
        } catch (IOException e) {
            logger.warn("Could not load application.properties", e);
        }
    }

    private ModelConfig createModelConfig() {
        int dim = getIntProperty("model.dim", 128);
        int heads = getIntProperty("model.heads", 4);
        int layers = getIntProperty("model.layers", 2);
        int seqLength = getIntProperty("seq.length", 64);

        logger.info("ModelConfig from properties: dim={}, heads={}, layers={}, seqLength={}", dim, heads, layers, seqLength);
        return new ModelConfig(dim, heads, layers, seqLength);
    }

    private int getIntProperty(String key, int defaultVal) {
        String v = properties.getProperty(key);
        if (v == null || v.isEmpty()) return defaultVal;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            logger.warn("Invalid integer for property {}: {}. Using default {}", key, v, defaultVal);
            return defaultVal;
        }
    }

    /**
     * Get a double property from configuration.
     *
     * @param key        the property key
     * @param defaultVal the default value
     * @return the property value or default
     */
    private double getDoubleProperty(String key, double defaultVal) {
        String v = properties.getProperty(key);
        if (v == null || v.isEmpty()) return defaultVal;
        try {
            return Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            logger.warn("Invalid double for property {}: {}. Using default {}", key, v, defaultVal);
            return defaultVal;
        }
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    public String getProperty(String key) {
        return properties.getProperty(key);
    }

    /**
     * Get training epochs.
     *
     * @return number of epochs
     */
    public int getEpochs() {
        return epochs;
    }

    /**
     * Set training epochs.
     *
     * @param epochs the number of epochs
     */
    public void setEpochs(int epochs) {
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs must be positive");
        }
        this.epochs = epochs;
    }

    /**
     * Get batch size.
     *
     * @return batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Set batch size.
     *
     * @param batchSize the batch size
     */
    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        this.batchSize = batchSize;
    }

    /**
     * Get learning rate.
     *
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Set learning rate.
     *
     * @param learningRate the learning rate
     */
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0 || learningRate >= 1) {
            throw new IllegalArgumentException("Learning rate must be between 0 and 1");
        }
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "AppConfig{" +
                "epochs=" + epochs +
                ", batchSize=" + batchSize +
                ", learningRate=" + learningRate +
                ", modelConfig=" + modelConfig +
                '}';
    }
}