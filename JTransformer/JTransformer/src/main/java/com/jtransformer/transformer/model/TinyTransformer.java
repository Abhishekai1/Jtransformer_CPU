package com.jtransformer.transformer.model;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.encoder.TransformerEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main Tiny Transformer model class.
 */
public class TinyTransformer {

    private static final Logger logger = LoggerFactory.getLogger(TinyTransformer.class);
    private final TransformerEncoder encoder;
    private final ModelConfig config;

    public TinyTransformer(ModelConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("ModelConfig must not be null");
        }
        this.config = config;
        this.encoder = new TransformerEncoder(config.getLayers());
        logger.info("TinyTransformer initialized with dim={}, heads={}, layers={}",
                config.getDim(), config.getHeads(), config.getLayers());
    }

    public Tensor forward(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("Input tensor must not be null");
        }
        return encoder.forward(input);
    }

    public void saveCheckpoint(String path) {
        ModelCheckpoint.save(this, path);
    }

    public ModelConfig getConfig() {
        return config;
    }
}