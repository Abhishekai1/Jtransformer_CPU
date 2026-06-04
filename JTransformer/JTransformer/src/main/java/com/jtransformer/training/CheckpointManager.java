package com.jtransformer.training;

import com.jtransformer.transformer.model.TinyTransformer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;

/**
 * Handles model saving/loading.
 */
public class CheckpointManager {

    private static final Logger logger = LoggerFactory.getLogger(CheckpointManager.class);

    public void save(TinyTransformer model) {
        if (model == null) {
            throw new IllegalArgumentException("Model must not be null");
        }

        String filename = String.format("checkpoint-%s.json", Instant.now().toEpochMilli());
        Path checkpointDir = Paths.get("checkpoints");
        Path checkpointPath = checkpointDir.resolve(filename);

        try {
            Files.createDirectories(checkpointDir);
            model.saveCheckpoint(checkpointPath.toString());
            logger.info("Checkpoint saved to {}", checkpointPath.toAbsolutePath());
        } catch (Exception e) {
            logger.error("Failed to save checkpoint", e);
        }
    }
}