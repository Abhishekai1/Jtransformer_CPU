package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Backpropagation implementation.
 */
public class BackPropagation {

    private static final Logger logger = LoggerFactory.getLogger(BackPropagation.class);
    private Tensor gradients;

    public void backward(Tensor loss) {
        if (loss == null || loss.getData() == null) {
            throw new IllegalArgumentException("Loss tensor must not be null");
        }

        logger.info("Running backpropagation");
        INDArray lossData = loss.getData();
        INDArray gradientData;

        if (lossData.isScalar()) {
            gradientData = Nd4j.scalar(1.0);
        } else {
            gradientData = lossData.dup();
        }

        gradients = new Tensor(gradientData);
        logger.debug("Computed backward gradient with shape {}", gradientData.shape());
    }

    public Tensor getGradients() {
        return gradients;
    }
}