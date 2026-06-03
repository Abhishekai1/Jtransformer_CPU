package com.jtransformer.core.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Core Tensor abstraction using ND4J INDArray.
 */
public class Tensor {

    private static final Logger logger = LoggerFactory.getLogger(Tensor.class);
    private INDArray data;

    public Tensor(INDArray data) {
        this.data = data;
        logger.debug("Tensor created with shape: {}", data.shape());
    }

    public INDArray getData() {
        return data;
    }

    public Tensor add(Tensor other) {
        // TODO: Implement tensor addition with broadcasting
        return new Tensor(this.data.add(other.getData()));
    }

    // TODO: Add more operations
}