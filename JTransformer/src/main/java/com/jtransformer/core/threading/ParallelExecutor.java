package com.jtransformer.core.threading;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Parallel execution using structured concurrency and virtual threads.
 */
public class ParallelExecutor {

    private static final Logger logger = LoggerFactory.getLogger(ParallelExecutor.class);

    public void executeInParallel(Runnable... tasks) {
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (Runnable task : tasks) {
                executor.submit(task);
            }
        }
        logger.info("Parallel execution completed");
    }
}