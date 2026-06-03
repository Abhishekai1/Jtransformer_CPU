package com.jtransformer.transformer.embedding;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Token embedding layer.
 */
public class TokenEmbedding {

    private static final Logger logger = LoggerFactory.getLogger(TokenEmbedding.class);

    public Tensor embed(int[] tokens) {
        logger.debug("Generating token embeddings");
        // TODO: Lookup table based embedding
        return null;
    }
}