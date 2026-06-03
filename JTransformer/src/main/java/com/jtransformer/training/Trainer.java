package com.jtransformer.training;

import com.jtransformer.transformer.model.TinyTransformer;
import com.jtransformer.dataset.TextDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main training orchestrator.
 */
public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);
    private final TinyTransformer model;
    private final Optimizer optimizer;

    public Trainer(TinyTransformer model, com.jtransformer.config.AppConfig config) {
        this.model = model;
        this.optimizer = new AdamOptimizer();
    }

    public void train(TextDataset dataset, int epochs) {
        logger.info("Starting training for {} epochs", epochs);
        for (int e = 0; e < epochs; e++) {
            // TODO: Full training loop
            logger.info("Epoch {} completed", e);
        }
    }
}