package com.jtransformer.transformer.attention;

import com.jtransformer.core.math.Softmax;
import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.core.tensor.TensorOps;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Single attention head.
 */
public class AttentionHead {

    private static final Logger logger = LoggerFactory.getLogger(AttentionHead.class);

    public Tensor compute(Tensor q, Tensor k, Tensor v) {
        if (q == null || k == null || v == null) {
            throw new IllegalArgumentException("Query, key, and value must not be null");
        }

        INDArray qData = q.getData();
        INDArray kData = k.getData();
        INDArray vData = v.getData();

        if (qData.rank() != 2 || kData.rank() != 2 || vData.rank() != 2) {
            throw new IllegalArgumentException("AttentionHead expects 2D query/key/value tensors");
        }

        int dK = (int) qData.size(1);
        INDArray scores = qData.mmul(kData.transpose());
        double scale = 1.0 / Math.sqrt(dK);
        INDArray scaledScores = scores.mul(scale);

        Tensor attentionWeights = new Softmax().apply(new Tensor(scaledScores));
        return TensorOps.matmul(attentionWeights, v);
    }
}