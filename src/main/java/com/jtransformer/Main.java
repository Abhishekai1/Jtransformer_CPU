package com.jtransformer;

import com.jtransformer.config.AppConfig;
import com.jtransformer.transformer.model.TinyTransformer;
import com.jtransformer.training.Trainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main entry point for JTransformer research framework.
 */
public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        logger.info("Starting JTransformer...");

        AppConfig config = new AppConfig();
        TinyTransformer model = new TinyTransformer(config.getModelConfig());

        // Load dataset and train - research execution plan
        Trainer trainer = new Trainer(model, config);
        trainer.train();

        logger.info("JTransformer execution completed.");
    }
}