package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Optimizer interface.
 */
public interface Optimizer {
    void step(Tensor gradients);
}