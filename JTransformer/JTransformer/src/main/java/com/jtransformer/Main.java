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
        try {
            logger.info("Starting JTransformer CPU-Optimized Mini Framework");

            // Load configuration from properties or use defaults
            AppConfig config = loadConfiguration(args);
            logger.info("Configuration loaded: {}", config);

            // Initialize model with config
            logger.info("Initializing TinyTransformer model...");
            TinyTransformer model = new TinyTransformer(config.getModelConfig());
            logger.info("Model initialized successfully");

            // Load dataset
            logger.info("Loading Shakespeare dataset...");
            ShakespeareDatasetLoader loader = new ShakespeareDatasetLoader();
            var dataset = loader.loadDataset();
            if (dataset == null || dataset.isEmpty()) {
                logger.warn("Dataset is empty");
                return;
            }
            logger.info("Dataset loaded: {} samples", dataset.size());

            // Train model
            logger.info("Starting training (5 epochs)...");
            Trainer trainer = new Trainer(model, config);
            trainer.train(dataset, 5); // epochs
            logger.info("Training completed successfully");

            logger.info("JTransformer execution completed.");
        } catch (IllegalArgumentException e) {
            logger.error("Configuration error: {}", e.getMessage(), e);
            System.exit(1);
        } catch (RuntimeException e) {
            logger.error("Runtime error: {}", e.getMessage(), e);
            System.exit(2);
        } catch (Exception e) {
            logger.error("Unexpected error: {}", e.getMessage(), e);
            System.exit(3);
        }
    }

    /**
     * Load configuration from command-line arguments or defaults.
     *
     * @param args command-line arguments
     * @return loaded configuration
     * @throws IllegalArgumentException if configuration is invalid
     */
    private static AppConfig loadConfiguration(String[] args) throws IllegalArgumentException {
        AppConfig config = new AppConfig();

        // Parse command-line arguments if provided
        if (args != null && args.length > 0) {
            for (String arg : args) {
                if (arg.startsWith("--")) {
                    String[] parts = arg.substring(2).split("=");
                    if (parts.length == 2) {
                        String key = parts[0];
                        String value = parts[1];
                        applyConfigOption(config, key, value);
                    }
                }
            }
        }

        // Validate configuration
        if (config.getModelConfig() == null) {
            throw new IllegalArgumentException("Model configuration is null");
        }

        logger.info("Configuration loaded with model dim={}, heads={}, layers={}",
                config.getModelConfig().getDim(),
                config.getModelConfig().getHeads(),
                config.getModelConfig().getLayers());

        return config;
    }

    /**
     * Apply a single configuration option.
     *
     * @param config the configuration object
     * @param key    the configuration key
     * @param value  the configuration value
     */
    private static void applyConfigOption(AppConfig config, String key, String value) {
        try {
            switch (key.toLowerCase()) {
                case "epochs":
                    config.setEpochs(Integer.parseInt(value));
                    logger.info("Set epochs to {}", value);
                    break;
                case "batch_size":
                    config.setBatchSize(Integer.parseInt(value));
                    logger.info("Set batch size to {}", value);
                    break;
                case "learning_rate":
                    config.setLearningRate(Double.parseDouble(value));
                    logger.info("Set learning rate to {}", value);
                    break;
                default:
                    logger.debug("Unknown config option: {}", key);
            }
        } catch (NumberFormatException e) {
            logger.warn("Failed to parse config value for {}: {}", key, value);
        }
    }
}