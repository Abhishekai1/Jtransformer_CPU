package com.jtransformer.core.threading;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Parallel execution using structured concurrency and virtual threads.
 */
public class ParallelExecutor {

    private static final Logger logger = LoggerFactory.getLogger(ParallelExecutor.class);
    private static final long TERMINATION_TIMEOUT_SECONDS = 30;

    public void executeInParallel(Runnable... tasks) {
        if (tasks == null || tasks.length == 0) {
            logger.warn("No tasks supplied to executeInParallel");
            return;
        }

        List<Future<?>> futures = new ArrayList<>();
        ExecutorService executor = Executors.newFixedThreadPool(Math.min(tasks.length, Runtime.getRuntime().availableProcessors()));
        try {
            for (Runnable task : tasks) {
                if (task == null) {
                    logger.warn("Skipping null task");
                    continue;
                }
                futures.add(executor.submit(task));
            }

            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    logger.error("Task failed during parallel execution", e.getCause());
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    logger.warn("Parallel execution was interrupted", e);
                }
            }
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(TERMINATION_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        logger.info("Parallel execution completed");
    }
}