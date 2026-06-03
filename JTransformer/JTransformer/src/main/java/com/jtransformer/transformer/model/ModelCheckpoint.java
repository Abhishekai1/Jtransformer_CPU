package com.jtransformer.transformer.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Checkpoint management.
 */
public class ModelCheckpoint {

    private static final Logger logger = LoggerFactory.getLogger(ModelCheckpoint.class);

    public static void save(TinyTransformer model, String path) {
        logger.info("Saving checkpoint to {}", path);
        // TODO: Serialize weights
    }
}