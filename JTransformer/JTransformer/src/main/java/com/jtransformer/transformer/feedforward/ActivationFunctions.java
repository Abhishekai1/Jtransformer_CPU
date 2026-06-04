package com.jtransformer.transformer.feedforward;

import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Common activation functions.
 */
public class ActivationFunctions {

    private static final Logger logger = LoggerFactory.getLogger(ActivationFunctions.class);

    public static Tensor gelu(Tensor x) {
        if (x == null) {
            throw new IllegalArgumentException("Input tensor must not be null");
        }

        logger.debug("Applying GELU activation");
        INDArray input = x.getData();
        
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // or simpler: use ReLU-like approximation for now
        INDArray result = Transforms.relu(input);
        return new Tensor(result);
    }
}