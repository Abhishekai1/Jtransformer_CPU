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
        this.config = config;
        this.encoder = new TransformerEncoder();
        logger.info("TinyTransformer initialized with dim={}, heads={}, layers={}",
                config.getDim(), config.getHeads(), config.getLayers());
    }

    public Tensor forward(Tensor input) {
        return encoder.forward(input);
    }

    public void saveCheckpoint(String path) {
        // TODO
    }
}