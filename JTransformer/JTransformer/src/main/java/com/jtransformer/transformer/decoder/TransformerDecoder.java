package com.jtransformer.transformer.decoder;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Transformer Decoder stack.
 */
public class TransformerDecoder {

    private static final Logger logger = LoggerFactory.getLogger(TransformerDecoder.class);
    private final DecoderLayer[] layers;

    public TransformerDecoder() {
        this(6);
    }

    public TransformerDecoder(int numLayers) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("Decoder must contain at least one layer");
        }

        this.layers = new DecoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            this.layers[i] = new DecoderLayer();
        }
    }

    public Tensor forward(Tensor x, Tensor encoderOutput) {
        if (x == null || encoderOutput == null) {
            throw new IllegalArgumentException("Decoder input and encoder output must not be null");
        }

        logger.debug("Decoder forward");
        Tensor output = x;
        for (DecoderLayer layer : layers) {
            output = layer.forward(output, encoderOutput);
        }
        return output;
    }
}