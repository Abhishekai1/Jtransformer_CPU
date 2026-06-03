package com.jtransformer.core.threading;

import com.jtransformer.config.ThreadConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Manages Java 21 Virtual Threads for high concurrency.
 */
public class VirtualThreadManager {

    private static final Logger logger = LoggerFactory.getLogger(VirtualThreadManager.class);
    private final ThreadConfig config;

    public VirtualThreadManager(ThreadConfig config) {
        this.config = config;
    }

    public void execute(Runnable task) {
        Thread.ofVirtual().start(task);
        logger.debug("Virtual thread started");
    }
}