package com.jtransformer.core.math;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.core.tensor.TensorOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Optimized matrix multiplication with quantization awareness.
 */
public class MatrixMultiply {

    private static final Logger logger = LoggerFactory.getLogger(MatrixMultiply.class);

    public Tensor multiply(Tensor a, Tensor b) {
        logger.debug("Matrix multiplication called");
        return TensorOps.matmul(a, b);
    }
}