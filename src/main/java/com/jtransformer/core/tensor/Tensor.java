package com.jtransformer.core.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Wrapper around ND4J INDArray for transformer tensors.
 */
public class Tensor {

    private static final Logger logger = LoggerFactory.getLogger(Tensor.class);
    private INDArray data;

    public Tensor(INDArray data) {
        this.data = data;
        logger.trace("Tensor created with shape: {}", data.shape());
    }

    public INDArray getData() {
        return data;
    }

    public Tensor add(Tensor other) {
        if (other == null) {
            throw new IllegalArgumentException("Other tensor must not be null");
        }
        return new Tensor(this.data.add(other.getData()));
    }
}