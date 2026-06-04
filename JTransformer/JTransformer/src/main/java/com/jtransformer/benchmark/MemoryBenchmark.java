package com.jtransformer.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Memory usage with quantization.
 */
public class MemoryBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(MemoryBenchmark.class);

    public void run() {
        logger.info("Memory benchmark started");

        final int rows = 2048;
        final int cols = 2048;

        try {
            System.gc();
            Thread.sleep(100);
        } catch (InterruptedException ignored) {}

        long beforeUsedMb = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);

        INDArray a = Nd4j.rand(DataType.FLOAT, rows, cols);
        INDArray b = Nd4j.rand(DataType.FLOAT, rows, cols);

        long afterAllocMb = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);

        long floatBytes = estimateBytes(a) + estimateBytes(b);

        // quantize to int8
        double maxA = Transforms.abs(a).maxNumber().doubleValue();
        double maxB = Transforms.abs(b).maxNumber().doubleValue();
        double max = Math.max(maxA, maxB);
        double scale = max / 127.0 + 1e-8;

        INDArray aQ = Transforms.round(a.div(scale)).castTo(DataType.BYTE);
        INDArray bQ = Transforms.round(b.div(scale)).castTo(DataType.BYTE);

        long quantBytes = estimateBytes(aQ) + estimateBytes(bQ);

        logger.info("Heap used before alloc: {} MB", beforeUsedMb);
        logger.info("Heap used after alloc: {} MB", afterAllocMb);
        logger.info("Estimated float arrays memory: {} MB", floatBytes / (1024.0 * 1024.0));
        logger.info("Estimated quantized arrays memory: {} MB", quantBytes / (1024.0 * 1024.0));

        double reduction = 100.0 * (1.0 - (quantBytes / (double) floatBytes));
        logger.info("Estimated memory reduction after int8 quantization: {} %", reduction);
    }

    private long estimateBytes(INDArray arr) {
        long elements = arr.length();
        int bytesPer = elementSizeBytes(arr.dataType());
        return elements * (long) bytesPer;
    }

    private int elementSizeBytes(DataType dt) {
        switch (dt) {
            case DOUBLE: return 8;
            case FLOAT: return 4;
            case HALF: return 2;
            case INT: return 4;
            case LONG: return 8;
            case BYTE: return 1;
            case SHORT: return 2;
            default: return 4;
        }
    }
}