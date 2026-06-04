package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Gradient accumulation for larger effective batch sizes.
 */
public class GradientAccumulator {

    private static final Logger logger = LoggerFactory.getLogger(GradientAccumulator.class);

    private Tensor accumulatedGradient;
    private int accumulationSteps;

    public void accumulate(Tensor gradient) {
        if (gradient == null || gradient.getData() == null) {
            throw new IllegalArgumentException("Gradient tensor must not be null");
        }

        if (accumulatedGradient == null) {
            accumulatedGradient = new Tensor(gradient.getData().dup());
        } else {
            accumulatedGradient = accumulatedGradient.add(gradient);
        }
        accumulationSteps++;
        logger.debug("Accumulated gradient step {}, shape={}", accumulationSteps, gradient.getData().shape());
    }

    public Tensor getAccumulatedGradient() {
        return accumulatedGradient;
    }

    public Tensor getAverageGradient() {
        if (accumulationSteps == 0 || accumulatedGradient == null) {
            return null;
        }
        return new Tensor(accumulatedGradient.getData().div(accumulationSteps));
    }

    public void reset() {
        accumulatedGradient = null;
        accumulationSteps = 0;
        logger.debug("Gradient accumulator reset");
    }

    public int getAccumulationSteps() {
        return accumulationSteps;
    }

    public void accumulate() {
        throw new UnsupportedOperationException("Gradient must be provided to accumulate()");
    }
}