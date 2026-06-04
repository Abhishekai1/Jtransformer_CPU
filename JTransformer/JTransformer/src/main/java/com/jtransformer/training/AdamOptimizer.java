package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adam optimizer implementation.
 */
public class AdamOptimizer implements Optimizer {

    private static final Logger logger = LoggerFactory.getLogger(AdamOptimizer.class);
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private int step;
    private Tensor m;
    private Tensor v;

    public AdamOptimizer() {
        this.learningRate = 1e-3;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        this.step = 0;
    }

    @Override
    public void step(Tensor gradients) {
        if (gradients == null) {
            throw new IllegalArgumentException("Gradients tensor must not be null");
        }

        logger.debug("Adam step");
        INDArray gradData = gradients.getData();
        if (gradData == null) {
            throw new IllegalArgumentException("Gradient data must not be null");
        }

        if (m == null) {
            m = new Tensor(Nd4j.zerosLike(gradData));
        }
        if (v == null) {
            v = new Tensor(Nd4j.zerosLike(gradData));
        }

        step++;

        INDArray mData = m.getData();
        INDArray vData = v.getData();

        mData.muli(beta1).addi(gradData.mul(1.0 - beta1));
        vData.muli(beta2).addi(gradData.mul(gradData).mul(1.0 - beta2));

        double biasCorrection1 = 1.0 - Math.pow(beta1, step);
        double biasCorrection2 = 1.0 - Math.pow(beta2, step);

        INDArray mHat = mData.div(biasCorrection1);
        INDArray vHat = vData.div(biasCorrection2);
        INDArray update = mHat.div(Transforms.sqrt(vHat).addi(epsilon)).muli(learningRate);

        gradData.subi(update);
        logger.debug("Adam gradient update applied at step {}", step);
    }
}