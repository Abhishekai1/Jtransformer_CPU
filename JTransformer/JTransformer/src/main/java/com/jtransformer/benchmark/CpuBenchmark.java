package com.jtransformer.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * CPU performance benchmarks.
 */
public class CpuBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(CpuBenchmark.class);

    public void runMatmulBenchmark() {
        logger.info("Running CPU matrix multiplication benchmark");
        // Simple ND4J matrix multiplication timing
        final int size = 1024;
        final int warmupIters = 5;
        final int iters = 10;

        INDArray a = Nd4j.rand(size, size);
        INDArray b = Nd4j.rand(size, size);

        // Warm up the JVM and any native kernels
        for (int i = 0; i < warmupIters; i++) {
            INDArray c = a.mmul(b);
            c.sumNumber(); // materialize result
        }

        long totalNs = 0L;
        for (int i = 0; i < iters; i++) {
            long t0 = System.nanoTime();
            INDArray c = a.mmul(b);
            c.sumNumber();
            long t1 = System.nanoTime();
            long dt = t1 - t0;
            totalNs += dt;
            logger.info("Matmul iteration {}: {} ms", i + 1, dt / 1_000_000.0);
        }

        double avgMs = (totalNs / (double) iters) / 1_000_000.0;
        logger.info("Average ND4J matmul time ({}x{}): {} ms", size, size, avgMs);
        
    }
}