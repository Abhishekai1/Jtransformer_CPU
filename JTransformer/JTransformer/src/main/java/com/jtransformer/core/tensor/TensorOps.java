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
        if (a == null || b == null) {
            throw new IllegalArgumentException("Both input tensors must be non-null");
        }

        INDArray left = a.getData();
        INDArray right = b.getData();

        if (left.rank() != 2 || right.rank() != 2) {
            throw new IllegalArgumentException("Tensor matmul requires 2D matrices");
        }
        if (left.columns() != right.rows()) {
            throw new IllegalArgumentException(String.format(
                    "Incompatible shapes for matmul: [%d,%d] x [%d,%d]",
                    left.rows(), left.columns(), right.rows(), right.columns()));
        }

        logger.debug("Performing matrix multiplication: {}x{} * {}x{}",
                left.rows(), left.columns(), right.rows(), right.columns());
        INDArray result = left.mmul(right);
        return new Tensor(result);
    }

    public static Tensor transpose(Tensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("Tensor must be non-null");
        }
        return new Tensor(tensor.getData().transpose());
    }
}