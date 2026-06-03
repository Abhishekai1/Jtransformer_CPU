package com.jtransformer.core.quantization;

import com.jtransformer.core.tensor.QuantizedTensor;
import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Int8 Quantizer for weights and activations.
 */
public class Int8Quantizer {

    private static final Logger logger = LoggerFactory.getLogger(Int8Quantizer.class);

    public QuantizedTensor quantize(Tensor tensor) {
        logger.info("Quantizing to INT8");
        // TODO: Implement scale + zero-point quantization
        return new QuantizedTensor(new byte[0], 1.0f, 0.0f);
    }
}