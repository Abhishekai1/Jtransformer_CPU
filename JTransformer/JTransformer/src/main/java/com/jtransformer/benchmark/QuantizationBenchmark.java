package com.jtransformer.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Quantization performance vs accuracy tradeoff.
 */
public class QuantizationBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(QuantizationBenchmark.class);

    public void compareInt8() {
        logger.info("Running quantization compareInt8");

        final int size = 512;
        final int warmupIters = 3;
        final int iters = 5;

        INDArray a = Nd4j.rand(DataType.FLOAT, size, size);
        INDArray b = Nd4j.rand(DataType.FLOAT, size, size);

        // Baseline float matmul (warmup)
        for (int i = 0; i < warmupIters; i++) {
            INDArray c = a.mmul(b);
            c.sumNumber();
        }

        long totalFloatNs = 0L;
        INDArray floatResult = null;
        for (int i = 0; i < iters; i++) {
            long t0 = System.nanoTime();
            INDArray c = a.mmul(b);
            c.sumNumber();
            long t1 = System.nanoTime();
            long dt = t1 - t0;
            totalFloatNs += dt;
            floatResult = c;
            logger.info("Float matmul iteration {}: {} ms", i + 1, dt / 1_000_000.0);
        }

        double avgFloatMs = (totalFloatNs / (double) iters) / 1_000_000.0;

        // Quantize to int8 (symmetric) and dequantize for matmul
        double maxA = Transforms.abs(a).maxNumber().doubleValue();
        double maxB = Transforms.abs(b).maxNumber().doubleValue();
        double max = Math.max(maxA, maxB);
        double scale = max / 127.0 + 1e-8;

        // quantize
        INDArray aScaled = a.div(scale);
        INDArray bScaled = b.div(scale);
        INDArray aRounded = Transforms.round(aScaled);
        INDArray bRounded = Transforms.round(bScaled);
        INDArray aQ = aRounded.castTo(DataType.BYTE);
        INDArray bQ = bRounded.castTo(DataType.BYTE);

        // dequantize
        INDArray aDq = aQ.castTo(DataType.FLOAT).mul(scale);
        INDArray bDq = bQ.castTo(DataType.FLOAT).mul(scale);

        // Warmup quantized path
        for (int i = 0; i < warmupIters; i++) {
            INDArray c = aDq.mmul(bDq);
            c.sumNumber();
        }

        long totalQuantNs = 0L;
        INDArray quantResult = null;
        for (int i = 0; i < iters; i++) {
            long t0 = System.nanoTime();
            INDArray c = aDq.mmul(bDq);
            c.sumNumber();
            long t1 = System.nanoTime();
            long dt = t1 - t0;
            totalQuantNs += dt;
            quantResult = c;
            logger.info("Quantized(deq) matmul iteration {}: {} ms", i + 1, dt / 1_000_000.0);
        }

        double avgQuantMs = (totalQuantNs / (double) iters) / 1_000_000.0;

        // Compute accuracy difference
        INDArray diff = Transforms.abs(floatResult.sub(quantResult));
        double maxAbsError = diff.maxNumber().doubleValue();

        logger.info("Average float matmul: {} ms", avgFloatMs);
        logger.info("Average quantized (dequant->float) matmul: {} ms", avgQuantMs);
        logger.info("Max absolute error between float and quantized result: {}", maxAbsError);
    }
}