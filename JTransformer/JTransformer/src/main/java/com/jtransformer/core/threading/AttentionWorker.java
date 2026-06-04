package com.jtransformer.core.threading;

import com.jtransformer.core.math.Softmax;
import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.core.tensor.TensorOps;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Worker for parallel attention computation.
 */
public class AttentionWorker implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(AttentionWorker.class);
    private final Tensor query;
    private final Tensor key;
    private final Tensor value;
    private Tensor output;

    public AttentionWorker(Tensor q, Tensor k, Tensor v) {
        this.query = q;
        this.key = k;
        this.value = v;
    }

    @Override
    public void run() {
        logger.debug("AttentionWorker processing head");
        if (query == null || key == null || value == null) {
            throw new IllegalStateException("Query, key, and value tensors must not be null");
        }

        INDArray qData = query.getData();
        INDArray kData = key.getData();
        INDArray vData = value.getData();

        if (qData.rank() != 2 || kData.rank() != 2 || vData.rank() != 2) {
            throw new IllegalArgumentException("AttentionWorker expects 2D query/key/value tensors");
        }

        int dK = (int) qData.size(1);
        INDArray scores = qData.mmul(kData.transpose());
        double scale = 1.0 / Math.sqrt(dK);
        INDArray scaledScores = scores.mul(scale);

        Tensor attentionWeights = new Softmax().apply(new Tensor(scaledScores));
        output = TensorOps.matmul(attentionWeights, value);
    }

    public Tensor getOutput() {
        return output;
    }
}