package com.jtransformer.core.math;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Softmax activation for attention scores.
 */
public class Softmax {

    private static final Logger logger = LoggerFactory.getLogger(Softmax.class);

    public Tensor apply(Tensor input) {
        logger.debug("Applying softmax");
        // TODO: Implement stable softmax using ND4J
        return input;
    }
}