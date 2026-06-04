package com.jtransformer.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Virtual thread scaling benchmark.
 */
public class ThreadBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(ThreadBenchmark.class);

    public void benchmarkScaling() {
        logger.info("Running virtual thread scaling benchmark");

        final int matrixSize = 256;
        final int tasksPerWorker = 4;
        final int warmupRuns = 2;
        final int measureRuns = 3;

        int[] workerCounts = new int[]{1, 2, 4, 8, 16, 32, 64};

        for (int workers : workerCounts) {
            int totalTasks = workers * tasksPerWorker;

            // Build tasks
            List<Callable<Double>> tasks = new ArrayList<>();
            for (int t = 0; t < totalTasks; t++) {
                tasks.add(() -> {
                    INDArray a = Nd4j.rand(matrixSize, matrixSize);
                    INDArray b = Nd4j.rand(matrixSize, matrixSize);
                    long t0 = System.nanoTime();
                    INDArray c = a.mmul(b);
                    c.sumNumber();
                    long t1 = System.nanoTime();
                    return (t1 - t0) / 1_000_000.0; // ms
                });
            }

            // Platform threads (fixed thread pool)
            ExecutorService platform = Executors.newFixedThreadPool(workers);
            try {
                // warmup
                for (int w = 0; w < warmupRuns; w++) {
                    List<Future<Double>> fw = platform.invokeAll(tasks);
                    for (Future<Double> f : fw) f.get();
                }

                double totalMs = 0.0;
                for (int m = 0; m < measureRuns; m++) {
                    long t0 = System.nanoTime();
                    List<Future<Double>> fw = platform.invokeAll(tasks);
                    for (Future<Double> f : fw) totalMs += f.get();
                    long t1 = System.nanoTime();
                    double wallMs = (t1 - t0) / 1_000_000.0;
                    logger.info("Platform threads: workers={} run {} wall-time={} ms total-task-ms={}", workers, m + 1, wallMs, totalMs);
                }
                double avgTaskMs = totalMs / (measureRuns * totalTasks);
                logger.info("Platform threads: workers={} avg task time={} ms", workers, avgTaskMs);
            } catch (Exception e) {
                logger.error("Platform threads benchmark failed", e);
            } finally {
                platform.shutdown();
            }

            // Virtual threads (fixed thread pool)
            ExecutorService virtual = Executors.newFixedThreadPool(workers);
            try {
                // warmup
                for (int w = 0; w < warmupRuns; w++) {
                    List<Future<Double>> fw = virtual.invokeAll(tasks);
                    for (Future<Double> f : fw) f.get();
                }

                double totalMs = 0.0;
                for (int m = 0; m < measureRuns; m++) {
                    long t0 = System.nanoTime();
                    List<Future<Double>> fw = virtual.invokeAll(tasks);
                    for (Future<Double> f : fw) totalMs += f.get();
                    long t1 = System.nanoTime();
                    double wallMs = (t1 - t0) / 1_000_000.0;
                    logger.info("Virtual threads: workers={} run {} wall-time={} ms total-task-ms={}", workers, m + 1, wallMs, totalMs);
                }
                double avgTaskMs = totalMs / (measureRuns * totalTasks);
                logger.info("Virtual threads: workers={} avg task time={} ms", workers, avgTaskMs);
            } catch (Exception e) {
                logger.error("Virtual threads benchmark failed", e);
            } finally {
                virtual.shutdown();
            }

            // small pause between configs
            try { Thread.sleep(200); } catch (InterruptedException ignored) {}
        }
    }
}