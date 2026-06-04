package com.jtransformer.core.math;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Softmax activation for attention scores.
 */
public class Softmax {

    private static final Logger logger = LoggerFactory.getLogger(Softmax.class);

    public Tensor apply(Tensor input) {
        logger.debug("Applying softmax");
        INDArray x = input.getData();
        int lastDim = x.rank() - 1;

        // subtract max for numerical stability (keep dimensions for broadcasting)
        INDArray max = x.max(true, lastDim);
        INDArray shifted = x.sub(max);

        INDArray exp = Transforms.exp(shifted);
        INDArray sum = exp.sum(true, lastDim);
        INDArray soft = exp.div(sum);

        return new Tensor(soft);
    }
}