package com.jtransformer.transformer.attention;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Causal and padding attention masks.
 */
public class AttentionMask {

    private static final Logger logger = LoggerFactory.getLogger(AttentionMask.class);
    private final Tensor mask;

    public AttentionMask(Tensor mask) {
        if (mask == null || mask.getData() == null) {
            throw new IllegalArgumentException("Attention mask cannot be null");
        }
        this.mask = mask;
        logger.debug("AttentionMask created with shape {}", mask.getData().shape());
    }

    public Tensor getMask() {
        return mask;
    }

    public boolean isCausal() {
        return mask.getData().rank() == 2;
    }
}