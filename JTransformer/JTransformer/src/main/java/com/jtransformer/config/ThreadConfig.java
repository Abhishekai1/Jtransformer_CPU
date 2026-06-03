package com.jtransformer.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configuration for virtual threads and parallel execution.
 */
public class ThreadConfig {

    private static final Logger logger = LoggerFactory.getLogger(ThreadConfig.class);

    private final int numVirtualThreads;
    private final int parallelism;

    public ThreadConfig() {
        this.numVirtualThreads = Runtime.getRuntime().availableProcessors() * 4;
        this.parallelism = Runtime.getRuntime().availableProcessors();
        logger.info("ThreadConfig initialized with {} virtual threads", numVirtualThreads);
    }

    public int getNumVirtualThreads() {
        return numVirtualThreads;
    }

    public int getParallelism() {
        return parallelism;
    }
}