package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;

/**
 * Optimizer interface.
 */
public interface Optimizer {
    void step(Tensor gradients);
}