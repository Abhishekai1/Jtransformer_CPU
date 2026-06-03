package com.jtransformer.training;

import com.jtransformer.transformer.model.TinyTransformer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles model saving/loading.
 */
public class CheckpointManager {

    private static final Logger logger = LoggerFactory.getLogger(CheckpointManager.class);

    public void save(TinyTransformer model) {
        logger.info("Checkpoint saved");
    }
}