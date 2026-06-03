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
        this.mask = mask;
    }

    public Tensor getMask() {
        return mask;
    }
}