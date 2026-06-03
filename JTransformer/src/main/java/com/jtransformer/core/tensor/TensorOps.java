package com.jtransformer.core.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility operations for tensors.
 */
public class TensorOps {

    private static final Logger logger = LoggerFactory.getLogger(TensorOps.class);

    public static Tensor matmul(Tensor a, Tensor b) {
        logger.debug("Performing matrix multiplication");
        // TODO: Full implementation with ND4J
        INDArray result = a.getData().mmul(b.getData());
        return new Tensor(result);
    }

    public static Tensor transpose(Tensor tensor) {
        return new Tensor(tensor.getData().transpose());
    }
}