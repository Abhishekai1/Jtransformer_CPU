package com.jtransformer.transformer.encoder;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.attention.MultiHeadAttention;
import com.jtransformer.transformer.feedforward.FeedForwardNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Single Transformer Encoder Layer.
 */
public class EncoderLayer {

    private final MultiHeadAttention mha;
    private final FeedForwardNetwork ffn;

    public EncoderLayer() {
        this.mha = new MultiHeadAttention();
        this.ffn = new FeedForwardNetwork();
    }

    public Tensor forward(Tensor x) {
        // TODO: Self-attention + residual + norm + FFN + residual + norm
        return x;
    }
}