package com.jtransformer.transformer.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Checkpoint management.
 */
public class ModelCheckpoint {

    private static final Logger logger = LoggerFactory.getLogger(ModelCheckpoint.class);

    public static void save(TinyTransformer model, String path) {
        if (model == null) {
            throw new IllegalArgumentException("Model must not be null");
        }
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("Checkpoint path must not be blank");
        }

        logger.info("Saving checkpoint to {}", path);
        ModelConfig config = model.getConfig();
        String checkpoint = String.format(
                "{\n  \"dim\": %d,\n  \"heads\": %d,\n  \"layers\": %d,\n  \"seqLength\": %d\n}",
                config.getDim(), config.getHeads(), config.getLayers(), config.getSeqLength());

        Path target = Paths.get(path);
        try {
            Path parent = target.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(target, checkpoint.getBytes(StandardCharsets.UTF_8));
            logger.info("Checkpoint file written successfully");
        } catch (IOException e) {
            logger.error("Failed to write checkpoint file", e);
        }
    }
}