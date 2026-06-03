package com.jtransformer.transformer.decoder;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Single Decoder Layer.
 */
public class DecoderLayer {

    private static final Logger logger = LoggerFactory.getLogger(DecoderLayer.class);

    public Tensor forward(Tensor x, Tensor encOutput) {
        // TODO: Masked self-attn + cross-attn + FFN
        return x;
    }
}