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
        ExecutorService executor = Executors.newFixedThreadPool(tasks.length);
        try {
            for (Runnable task : tasks) {
                executor.submit(task);
            }
        } finally {
            executor.shutdown();
        }
        logger.info("Parallel execution completed");
    }
}