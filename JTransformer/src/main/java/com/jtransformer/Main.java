package com.jtransformer;

import com.jtransformer.config.AppConfig;
import com.jtransformer.transformer.model.TinyTransformer;
import com.jtransformer.training.Trainer;
import com.jtransformer.dataset.ShakespeareDatasetLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main entry point for JTransformer research framework.
 * Demonstrates a complete tiny transformer pipeline.
 */
public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        logger.info("Starting JTransformer CPU-Optimized Mini Framework");

        // TODO: Load configuration
        AppConfig config = new AppConfig();

        // Initialize model
        TinyTransformer model = new TinyTransformer(config.getModelConfig());

        // Load dataset
        ShakespeareDatasetLoader loader = new ShakespeareDatasetLoader();
        var dataset = loader.loadDataset();

        // Train
        Trainer trainer = new Trainer(model, config);
        trainer.train(dataset, 5); // epochs

        logger.info("JTransformer execution completed.");
    }
}