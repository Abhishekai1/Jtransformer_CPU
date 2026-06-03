package com.jtransformer.dataset;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Generates batches for training.
 */
public class BatchGenerator {

    private static final Logger logger = LoggerFactory.getLogger(BatchGenerator.class);

    public Tensor[] nextBatch(TextDataset dataset, int batchSize) {
        // TODO
        return new Tensor[0];
    }
}