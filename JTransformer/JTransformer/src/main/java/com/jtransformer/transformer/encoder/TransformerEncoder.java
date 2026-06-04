package com.jtransformer.transformer.encoder;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Full Transformer Encoder stack.
 */
public class TransformerEncoder {

    private static final Logger logger = LoggerFactory.getLogger(TransformerEncoder.class);
    private final EncoderLayer[] layers;

    public TransformerEncoder() {
        this(6);
    }

    public TransformerEncoder(int numLayers) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("Encoder must contain at least one layer");
        }
        this.layers = new EncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            this.layers[i] = new EncoderLayer();
        }
    }

    public Tensor forward(Tensor x) {
        if (x == null) {
            throw new IllegalArgumentException("Encoder input must not be null");
        }

        logger.debug("Encoder forward pass");
        Tensor output = x;
        for (EncoderLayer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
}