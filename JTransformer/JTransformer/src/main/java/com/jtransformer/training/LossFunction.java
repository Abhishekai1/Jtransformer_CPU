package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Cross-entropy loss for language modeling.
 */
public class LossFunction {

    private static final Logger logger = LoggerFactory.getLogger(LossFunction.class);

    public Tensor compute(Tensor predictions, Tensor targets) {
        if (predictions == null || predictions.getData() == null) {
            throw new IllegalArgumentException("Predictions must not be null");
        }
        if (targets == null || targets.getData() == null) {
            throw new IllegalArgumentException("Targets must not be null");
        }

        INDArray logits = predictions.getData();
        INDArray targetData = targets.getData();
        int lastDim = logits.rank() - 1;

        INDArray maxLogits = logits.max(true, lastDim);
        INDArray shifted = logits.sub(maxLogits);
        INDArray exp = Transforms.exp(shifted);
        INDArray softmax = exp.div(exp.sum(true, lastDim));
        INDArray logSoftmax = Transforms.log(softmax);

        INDArray loss;
        if (targetData.shape().length == logits.shape().length && java.util.Arrays.equals(targetData.shape(), logits.shape())) {
            loss = targetData.mul(logSoftmax).sum(lastDim).negi();
        } else if (targetData.rank() == 1 || (targetData.rank() == 2 && targetData.size(1) == 1)) {
            INDArray indices = targetData;
            if (targetData.rank() == 2) {
                indices = targetData.reshape(targetData.size(0));
            }
            long batchSize = indices.size(0);
            INDArray gathered = Nd4j.create(DataType.FLOAT, batchSize);
            for (int i = 0; i < batchSize; i++) {
                int classIndex = indices.getInt(i);
                gathered.putScalar(i, logSoftmax.getFloat(i, classIndex));
            }
            loss = gathered.negi();
        } else {
            throw new IllegalArgumentException("Unsupported target shape for cross-entropy loss: " + java.util.Arrays.toString(targetData.shape()));
        }

        INDArray lossScalar = loss.mean();
        logger.debug("Computed cross-entropy loss value={}", lossScalar.getDouble(0));
        return new Tensor(lossScalar);
    }
}