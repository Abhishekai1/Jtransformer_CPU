package com.jtransformer.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configuration for threading and virtual threads.
 */
public class ThreadConfig {

    private static final Logger logger = LoggerFactory.getLogger(ThreadConfig.class);

    public ThreadConfig() {
        logger.debug("ThreadConfig initialized");
    }

    public int getNumThreads() {
        // TODO: Return optimal thread count based on CPU cores
        return Runtime.getRuntime().availableProcessors();
    }
}