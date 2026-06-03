package com.jtransformer.transformer.encoder;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Full Transformer Encoder stack.
 */
public class TransformerEncoder {

    private static final Logger logger = LoggerFactory.getLogger(TransformerEncoder.class);

    public Tensor forward(Tensor x) {
        logger.debug("Encoder forward pass");
        // TODO: Stack of EncoderLayer
        return x;
    }
}