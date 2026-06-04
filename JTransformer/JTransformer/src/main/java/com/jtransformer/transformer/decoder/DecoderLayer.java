package com.jtransformer.transformer.decoder;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.attention.AttentionMask;
import com.jtransformer.transformer.attention.MultiHeadAttention;
import com.jtransformer.transformer.feedforward.FeedForwardNetwork;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Single Decoder Layer.
 */
public class DecoderLayer {

    private static final Logger logger = LoggerFactory.getLogger(DecoderLayer.class);
    private final MultiHeadAttention selfAttention;
    private final MultiHeadAttention crossAttention;
    private final FeedForwardNetwork ffn;

    public DecoderLayer() {
        this.selfAttention = new MultiHeadAttention();
        this.crossAttention = new MultiHeadAttention();
        this.ffn = new FeedForwardNetwork();
    }

    public Tensor forward(Tensor x, Tensor encOutput) {
        if (x == null || encOutput == null) {
            throw new IllegalArgumentException("Decoder input and encoder output must not be null");
        }

        logger.debug("DecoderLayer forward");

        Tensor selfMask = buildCausalMask(x);
        Tensor selfAttended = selfAttention.forward(x, x, x, new AttentionMask(selfMask));
        Tensor selfResidual = x.add(selfAttended);

        Tensor crossAttended = crossAttention.forward(selfResidual, encOutput, encOutput, null);
        Tensor crossResidual = selfResidual.add(crossAttended);

        Tensor ffOutput = ffn.forward(crossResidual);
        Tensor output = crossResidual.add(ffOutput);

        return output;
    }

    private Tensor buildCausalMask(Tensor input) {
        INDArray data = input.getData();
        if (data.rank() != 2) {
            throw new IllegalArgumentException("Decoder self-attention input must be a 2D tensor");
        }

        int seqLen = (int) data.size(0);
        INDArray mask = Nd4j.zeros(DataType.FLOAT, seqLen, seqLen);
        float negInf = -1e9f;

        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                mask.putScalar(i, j, negInf);
            }
        }

        return new Tensor(mask);
    }
}