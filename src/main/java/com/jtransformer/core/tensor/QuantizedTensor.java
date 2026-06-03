package com.jtransformer.core.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tensor with quantization support.
 */
public class QuantizedTensor extends Tensor {

    private static final Logger logger = LoggerFactory.getLogger(QuantizedTensor.class);

    private final int bits;

    public QuantizedTensor(INDArray data, int bits) {
        super(data);
        this.bits = bits;
        logger.info("Created QuantizedTensor with {} bits", bits);
    }

    public int getBits() {
        return bits;
    }

    // TODO: Dequantize method
}