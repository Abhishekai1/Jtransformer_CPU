package com.jtransformer.core.threading;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Worker for parallel attention computation.
 */
public class AttentionWorker implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(AttentionWorker.class);
    private final Tensor query;
    private final Tensor key;
    private final Tensor value;

    public AttentionWorker(Tensor q, Tensor k, Tensor v) {
        this.query = q;
        this.key = k;
        this.value = v;
    }

    @Override
    public void run() {
        logger.debug("AttentionWorker processing head");
        // TODO: Compute attention scores in parallel
    }
}