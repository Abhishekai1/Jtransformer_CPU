package com.jtransformer.transformer.attention;

import com.jtransformer.core.math.Softmax;
import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sparse attention for efficiency on long sequences.
 */
public class SparseAttention {

    private static final Logger logger = LoggerFactory.getLogger(SparseAttention.class);
    private final int windowSize;

    public SparseAttention() {
        this(8);
    }

    public SparseAttention(int windowSize) {
        if (windowSize < 1) {
            throw new IllegalArgumentException("windowSize must be at least 1");
        }
        this.windowSize = windowSize;
    }

    public Tensor computeSparseAttention(Tensor q, Tensor k, Tensor v) {
        logger.info("Using sparse attention with windowSize={}", windowSize);

        if (q == null || k == null || v == null) {
            throw new IllegalArgumentException("Query, key, and value tensors must not be null");
        }

        INDArray qData = q.getData();
        INDArray kData = k.getData();
        INDArray vData = v.getData();

        if (qData.rank() != 2 || kData.rank() != 2 || vData.rank() != 2) {
            throw new IllegalArgumentException("SparseAttention requires 2D query/key/value tensors");
        }

        long[] qShape = qData.shape();
        long[] kShape = kData.shape();
        long[] vShape = vData.shape();

        if (qShape[0] != kShape[0] || kShape[0] != vShape[0] || qShape[1] != kShape[1] || qShape[1] != vShape[1]) {
            throw new IllegalArgumentException("Q, K, and V must share the same [seqLen, dim] shape");
        }

        int seqLen = (int) qShape[0];
        int dim = (int) qShape[1];
        double scale = 1.0 / Math.sqrt(dim);

        INDArray output = Nd4j.zeros(DataType.FLOAT, seqLen, dim);

        for (int i = 0; i < seqLen; i++) {
            int start = Math.max(0, i - windowSize);
            int end = Math.min(seqLen, i + windowSize + 1);
            INDArray qi = qData.getRow(i);
            INDArray scoreVector = Nd4j.create(DataType.FLOAT, 1, end - start);

            for (int j = start; j < end; j++) {
                INDArray kj = kData.getRow(j);
                double score = qi.mul(kj).sumNumber().doubleValue() * scale;
                scoreVector.putScalar(0, j - start, score);
            }

            Tensor attentionWeights = new Softmax().apply(new Tensor(scoreVector));
            INDArray weightData = attentionWeights.getData();
            INDArray rowOut = Nd4j.zeros(DataType.FLOAT, dim);

            for (int j = start; j < end; j++) {
                float weight = weightData.getFloat(0, j - start);
                INDArray vj = vData.getRow(j);
                rowOut.addi(vj.mul(weight));
            }

            output.putRow(i, rowOut);
        }

        return new Tensor(output);
    }
}