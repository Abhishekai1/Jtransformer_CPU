package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Cross-entropy loss for language modeling.
 */
public class LossFunction {

    private static final Logger logger = LoggerFactory.getLogger(LossFunction.class);

    public Tensor compute(Tensor predictions, Tensor targets) {
        // TODO
        return predictions;
    }
}