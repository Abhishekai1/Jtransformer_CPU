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

    public AppConfig() {
        loadProperties();
        this.modelConfig = createModelConfig();
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
        // TODO: Parse properties into ModelConfig
        return new ModelConfig(128, 4, 2, 64);
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    public String getProperty(String key) {
        return properties.getProperty(key);
    }
}