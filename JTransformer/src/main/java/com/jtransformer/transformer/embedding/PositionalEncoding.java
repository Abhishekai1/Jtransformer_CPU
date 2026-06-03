package com.jtransformer.transformer.embedding;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sinusoidal positional encoding.
 */
public class PositionalEncoding {

    private static final Logger logger = LoggerFactory.getLogger(PositionalEncoding.class);

    public Tensor encode(int seqLength, int dim) {
        logger.debug("Computing positional encodings");
        // TODO: Sinusoidal implementation
        return null;
    }
}