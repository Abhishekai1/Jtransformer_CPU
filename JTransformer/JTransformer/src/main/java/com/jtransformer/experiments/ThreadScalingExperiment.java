package com.jtransformer.experiments;

import com.jtransformer.config.ThreadConfig;
import com.jtransformer.core.threading.VirtualThreadManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Virtual thread scaling experiment.
 */
public class ThreadScalingExperiment {

    private static final Logger logger = LoggerFactory.getLogger(ThreadScalingExperiment.class);

    public void run() {
        logger.info("Thread scaling experiment started");

        int matrixSize = 256;
        int tasksPerWorker = 2;
        int[] workerCounts = new int[]{1, 2, 4, 8, 16};

        ThreadConfig threadConfig = new ThreadConfig();
        VirtualThreadManager virtualThreadManager = new VirtualThreadManager(threadConfig);
        logger.info("Configured virtual threads: {}", threadConfig.getNumVirtualThreads());

        for (int workers : workerCounts) {
            int totalTasks = workers * tasksPerWorker;
            List<Callable<Double>> callables = new ArrayList<>();
            List<Runnable> virtualTasks = new ArrayList<>();

            for (int i = 0; i < totalTasks; i++) {
                callables.add(createMatrixTask(matrixSize));
                virtualTasks.add(createMatrixRunnable(matrixSize));
            }

            logger.info("Measuring with {} workers and {} tasks", workers, totalTasks);

            double platformWallMs = runPlatformThreadBenchmark(workers, callables);
            double virtualWallMs = runVirtualThreadBenchmark(virtualThreadManager, virtualTasks);

            logger.info("Worker count {}: platform wall-time={} ms, virtual wall-time={} ms", workers, platformWallMs, virtualWallMs);
        }
    }

    private Callable<Double> createMatrixTask(int matrixSize) {
        return () -> {
            long start = System.nanoTime();
            INDArray a = Nd4j.rand(DataType.FLOAT, matrixSize, matrixSize);
            INDArray b = Nd4j.rand(DataType.FLOAT, matrixSize, matrixSize);
            INDArray c = a.mmul(b);
            c.sumNumber();
            long end = System.nanoTime();
            return (end - start) / 1_000_000.0;
        };
    }

    private Runnable createMatrixRunnable(int matrixSize) {
        return () -> {
            INDArray a = Nd4j.rand(DataType.FLOAT, matrixSize, matrixSize);
            INDArray b = Nd4j.rand(DataType.FLOAT, matrixSize, matrixSize);
            INDArray c = a.mmul(b);
            c.sumNumber();
        };
    }

    private double runPlatformThreadBenchmark(int workers, List<Callable<Double>> tasks) {
        ExecutorService platform = Executors.newFixedThreadPool(workers);
        try {
            // Warmup
            platform.invokeAll(tasks);

            long start = System.nanoTime();
            List<Future<Double>> futures = platform.invokeAll(tasks);
            double totalTaskMs = 0.0;
            for (Future<Double> future : futures) {
                totalTaskMs += future.get();
            }
            long end = System.nanoTime();
            double wallMs = (end - start) / 1_000_000.0;
            logger.debug("Platform threads completed tasks in {} ms", wallMs);
            logger.debug("Platform total compute time {} ms", totalTaskMs);
            return wallMs;
        } catch (Exception e) {
            logger.error("Platform thread benchmark failed", e);
            return -1.0;
        } finally {
            platform.shutdown();
        }
    }

    private double runVirtualThreadBenchmark(VirtualThreadManager virtualThreadManager, List<Runnable> tasks) {
        try {
            long start = System.nanoTime();
            virtualThreadManager.executeAndJoin(tasks.toArray(new Runnable[0]));
            long end = System.nanoTime();
            double wallMs = (end - start) / 1_000_000.0;
            logger.debug("Virtual threads completed tasks in {} ms", wallMs);
            return wallMs;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Virtual thread benchmark was interrupted", e);
            return -1.0;
        }
    }
}