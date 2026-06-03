package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adam optimizer implementation.
 */
public class AdamOptimizer implements Optimizer {

    private static final Logger logger = LoggerFactory.getLogger(AdamOptimizer.class);

    @Override
    public void step(Tensor gradients) {
        logger.debug("Adam step");
        // TODO: Adam update rule
    }
}