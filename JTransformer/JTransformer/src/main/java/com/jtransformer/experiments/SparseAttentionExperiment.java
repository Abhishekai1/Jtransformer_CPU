package com.jtransformer.experiments;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.attention.SparseAttention;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Experiment for sparse attention efficiency.
 */
public class SparseAttentionExperiment {

    private static final Logger logger = LoggerFactory.getLogger(SparseAttentionExperiment.class);

    public void run() {
        logger.info("Running Sparse Attention Experiment");

        int seqLen = 256;
        int dim = 64;
        int windowSize = 8;

        INDArray query = Nd4j.rand(DataType.FLOAT, seqLen, dim).sub(0.5).muli(2.0);
        INDArray key = Nd4j.rand(DataType.FLOAT, seqLen, dim).sub(0.5).muli(2.0);
        INDArray value = Nd4j.rand(DataType.FLOAT, seqLen, dim).sub(0.5).muli(2.0);

        Tensor qTensor = new Tensor(query);
        Tensor kTensor = new Tensor(key);
        Tensor vTensor = new Tensor(value);

        long denseStart = System.nanoTime();
        Tensor denseOutput = computeDenseAttention(qTensor, kTensor, vTensor);
        long denseElapsed = System.nanoTime() - denseStart;

        long sparseStart = System.nanoTime();
        SparseAttention sparseAttention = new SparseAttention(windowSize);
        Tensor sparseOutput = sparseAttention.computeSparseAttention(qTensor, kTensor, vTensor);
        long sparseElapsed = System.nanoTime() - sparseStart;

        INDArray diff = denseOutput.getData().sub(sparseOutput.getData());
        double meanAbsDiff = Transforms.abs(diff).meanNumber().doubleValue();
        double maxAbsDiff = Transforms.abs(diff).maxNumber().doubleValue();

        double densityRatio = (2.0 * windowSize + 1.0) / seqLen;

        logger.info("Dense attention shape: {}", query.shape());
        logger.info("Sparse attention windowSize: {}", windowSize);
        logger.info("Sparse attention density ratio: {}", densityRatio);
        logger.info("Dense attention time (ms): {}", denseElapsed / 1_000_000.0);
        logger.info("Sparse attention time (ms): {}", sparseElapsed / 1_000_000.0);
        logger.info("Mean abs difference to dense: {}", meanAbsDiff);
        logger.info("Max abs difference to dense: {}", maxAbsDiff);
    }

    private Tensor computeDenseAttention(Tensor q, Tensor k, Tensor v) {
        INDArray qData = q.getData();
        INDArray kData = k.getData();
        INDArray vData = v.getData();

        int dim = (int) qData.size(1);
        double scale = 1.0 / Math.sqrt(dim);

        INDArray scores = qData.mmul(kData.transpose()).muli(scale);
        INDArray weights = Transforms.softmax(scores);
        INDArray output = weights.mmul(vData);
        return new Tensor(output);
    }
}