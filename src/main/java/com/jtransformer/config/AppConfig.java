package com.jtransformer.config;

import com.jtransformer.transformer.model.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Application configuration manager.
 */
public class AppConfig {

    private static final Logger logger = LoggerFactory.getLogger(AppConfig.class);

    private final ModelConfig modelConfig;

    public AppConfig() {
        this.modelConfig = new ModelConfig();
        logger.info("AppConfig initialized");
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    // TODO: Load from application.properties
}