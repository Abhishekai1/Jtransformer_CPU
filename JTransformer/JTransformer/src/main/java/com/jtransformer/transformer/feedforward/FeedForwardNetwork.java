package com.jtransformer.transformer.feedforward;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Position-wise Feed Forward Network.
 */
public class FeedForwardNetwork {

    private static final Logger logger = LoggerFactory.getLogger(FeedForwardNetwork.class);

    public Tensor forward(Tensor input) {
        logger.debug("FFN forward");
        // TODO: Linear -> ReLU/GELU -> Linear
        return input;
    }
}