package com.jtransformer.transformer.feedforward;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.core.tensor.TensorOps;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Position-wise Feed Forward Network.
 */
public class FeedForwardNetwork {

    private static final Logger logger = LoggerFactory.getLogger(FeedForwardNetwork.class);
    private INDArray w1;
    private INDArray b1;
    private INDArray w2;
    private INDArray b2;
    private int inputDim = -1;
    private int hiddenDim = -1;

    public Tensor forward(Tensor input) {
        logger.debug("FFN forward");
        if (input == null) {
            throw new IllegalArgumentException("FFN input must not be null");
        }

        INDArray x = input.getData();
        if (x.rank() != 2) {
            throw new IllegalArgumentException("FFN expects a 2D tensor [seqLen, dim]");
        }

        int dim = (int) x.size(1);
        if (inputDim != dim) {
            initializeWeights(dim);
        }

        Tensor hidden = TensorOps.matmul(input, new Tensor(w1));
        hidden = new Tensor(hidden.getData().addRowVector(b1));
        hidden = ActivationFunctions.gelu(hidden);

        Tensor output = TensorOps.matmul(hidden, new Tensor(w2));
        output = new Tensor(output.getData().addRowVector(b2));
        return output;
    }

    private void initializeWeights(int dim) {
        this.inputDim = dim;
        this.hiddenDim = dim * 4;
        double initScale = Math.sqrt(2.0 / dim);

        this.w1 = Nd4j.randn(DataType.FLOAT, dim, hiddenDim).muli(initScale);
        this.b1 = Nd4j.zeros(DataType.FLOAT, 1, hiddenDim);
        this.w2 = Nd4j.randn(DataType.FLOAT, hiddenDim, dim).muli(initScale);
        this.b2 = Nd4j.zeros(DataType.FLOAT, 1, dim);

        logger.debug("Initialized FFN weights dim={} hiddenDim={}", dim, hiddenDim);
    }
}