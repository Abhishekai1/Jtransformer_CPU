package com.jtransformer.transformer.attention;

import com.jtransformer.core.math.Softmax;
import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Multi-Head Attention core implementation.
 */
public class MultiHeadAttention {

    private static final Logger logger = LoggerFactory.getLogger(MultiHeadAttention.class);
    private final int numHeads;

    public MultiHeadAttention() {
        this(8);
    }

    public MultiHeadAttention(int numHeads) {
        if (numHeads <= 0) {
            throw new IllegalArgumentException("Number of heads must be positive");
        }
        this.numHeads = numHeads;
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value, AttentionMask mask) {
        logger.debug("MultiHeadAttention forward pass with {} heads", numHeads);

        if (query == null || key == null || value == null) {
            throw new IllegalArgumentException("Query, key, and value tensors must not be null");
        }

        INDArray qData = query.getData();
        INDArray kData = key.getData();
        INDArray vData = value.getData();

        if (qData.rank() != 2 || kData.rank() != 2 || vData.rank() != 2) {
            throw new IllegalArgumentException("MultiHeadAttention expects 2D query/key/value tensors");
        }

        if (qData.size(1) != kData.size(1) || qData.size(1) != vData.size(1)) {
            throw new IllegalArgumentException("Query, key, and value must share the same feature dimension");
        }

        int dim = (int) qData.size(1);
        if (dim % numHeads != 0) {
            throw new IllegalArgumentException("Feature dimension must be divisible by numHeads");
        }

        int headDim = dim / numHeads;
        INDArray[] headOutputs = new INDArray[numHeads];

        for (int head = 0; head < numHeads; head++) {
            int start = head * headDim;
            int end = start + headDim;
            INDArray qHead = qData.get(NDArrayIndex.all(), NDArrayIndex.interval(start, end));
            INDArray kHead = kData.get(NDArrayIndex.all(), NDArrayIndex.interval(start, end));
            INDArray vHead = vData.get(NDArrayIndex.all(), NDArrayIndex.interval(start, end));

            INDArray scores = qHead.mmul(kHead.transpose());
            if (mask != null) {
                INDArray maskData = mask.getMask().getData();
                if (!Arrays.equals(maskData.shape(), scores.shape())) {
                    throw new IllegalArgumentException("Attention mask shape must match attention score shape");
                }
                scores = scores.add(maskData);
            }

            double scale = 1.0 / Math.sqrt(headDim);
            INDArray scaledScores = scores.mul(scale);
            Tensor attentionWeights = new Softmax().apply(new Tensor(scaledScores));
            headOutputs[head] = attentionWeights.getData().mmul(vHead);
        }

        INDArray combined = Nd4j.concat(1, headOutputs);
        return new Tensor(combined);
    }
}