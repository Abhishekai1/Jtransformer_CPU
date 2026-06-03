package com.jtransformer.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Performance timing utilities.
 */
public class TimerUtils {

    private static final Logger logger = LoggerFactory.getLogger(TimerUtils.class);

    public static long measure(Runnable task) {
        long start = System.nanoTime();
        task.run();
        long duration = System.nanoTime() - start;
        logger.info("Task took {} ms", duration / 1_000_000);
        return duration;
    }
}