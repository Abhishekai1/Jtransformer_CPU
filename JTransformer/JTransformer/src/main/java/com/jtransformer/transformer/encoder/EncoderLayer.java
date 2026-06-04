package com.jtransformer.transformer.encoder;

import com.jtransformer.core.math.LayerNorm;
import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.attention.MultiHeadAttention;
import com.jtransformer.transformer.feedforward.FeedForwardNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Single Transformer Encoder Layer.
 */
public class EncoderLayer {

    private static final Logger logger = LoggerFactory.getLogger(EncoderLayer.class);
    private final MultiHeadAttention mha;
    private final FeedForwardNetwork ffn;
    private final LayerNorm layerNorm1;
    private final LayerNorm layerNorm2;

    public EncoderLayer() {
        this.mha = new MultiHeadAttention();
        this.ffn = new FeedForwardNetwork();
        this.layerNorm1 = new LayerNorm();
        this.layerNorm2 = new LayerNorm();
    }

    public Tensor forward(Tensor x) {
        if (x == null) {
            throw new IllegalArgumentException("Encoder input must not be null");
        }

        logger.debug("EncoderLayer forward");

        Tensor attentionOutput = mha.forward(x, x, x, null);
        Tensor attentionResidual = x.add(attentionOutput);
        Tensor normedAttention = layerNorm1.normalize(attentionResidual);

        Tensor ffnOutput = ffn.forward(normedAttention);
        Tensor ffnResidual = normedAttention.add(ffnOutput);
        Tensor output = layerNorm2.normalize(ffnResidual);

        return output;
    }
}