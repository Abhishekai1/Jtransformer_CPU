package com.jtransformer.transformer.decoder;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Transformer Decoder stack.
 */
public class TransformerDecoder {

    private static final Logger logger = LoggerFactory.getLogger(TransformerDecoder.class);

    public Tensor forward(Tensor x, Tensor encoderOutput) {
        logger.debug("Decoder forward");
        return x;
    }
}