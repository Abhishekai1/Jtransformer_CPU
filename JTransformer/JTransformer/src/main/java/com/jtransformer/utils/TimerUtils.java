package com.jtransformer.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Performance timing utilities.
 */
public class TimerUtils {

    private static final Logger logger = LoggerFactory.getLogger(TimerUtils.class);
    private static final ThreadLocal<Long> timerStack = ThreadLocal.withInitial(() -> 0L);

    /**
     * Measure execution time of a runnable task in nanoseconds.
     *
     * @param task the task to measure
     * @return duration in nanoseconds
     */
    public static long measure(Runnable task) {
        if (task == null) {
            logger.warn("Cannot measure null task");
            return 0L;
        }
        long start = System.nanoTime();
        task.run();
        long duration = System.nanoTime() - start;
        logger.info("Task took {} ms", duration / 1_000_000);
        return duration;
    }

    /**
     * Measure execution time of a callable task and return its result.
     *
     * @param task the task to measure
     * @param <T>  the return type
     * @return the result of the task
     * @throws Exception if the task throws
     */
    public static <T> T measure(java.util.concurrent.Callable<T> task) throws Exception {
        if (task == null) {
            logger.warn("Cannot measure null task");
            return null;
        }
        long start = System.nanoTime();
        T result = task.call();
        long duration = System.nanoTime() - start;
        logger.info("Task took {} ms", duration / 1_000_000);
        return result;
    }

    /**
     * Start a timer and return label for later elapsed calculation.
     *
     * @param label the timer label
     * @return the current nanoTime for later elapsed calculation
     */
    public static long start(String label) {
        logger.debug("Timer started: {}", label);
        return System.nanoTime();
    }

    /**
     * Calculate elapsed time since start in milliseconds.
     *
     * @param startTime the nanoTime returned from start()
     * @return elapsed time in milliseconds
     */
    public static long elapsedMs(long startTime) {
        return (System.nanoTime() - startTime) / 1_000_000;
    }

    /**
     * Calculate elapsed time since start in seconds.
     *
     * @param startTime the nanoTime returned from start()
     * @return elapsed time in seconds
     */
    public static double elapsedSeconds(long startTime) {
        return (System.nanoTime() - startTime) / 1_000_000_000.0;
    }

    /**
     * Measure execution time in milliseconds.
     *
     * @param task the task to measure
     * @return duration in milliseconds
     */
    public static long measureMs(Runnable task) {
        return measure(task) / 1_000_000;
    }

    /**
     * Measure execution time in seconds.
     *
     * @param task the task to measure
     * @return duration in seconds
     */
    public static double measureSeconds(Runnable task) {
        return measure(task) / 1_000_000_000.0;
    }

    /**
     * Log elapsed time with label.
     *
     * @param label     the operation label
     * @param startTime the nanoTime from start()
     */
    public static void logElapsed(String label, long startTime) {
        if (label != null) {
            long elapsed = elapsedMs(startTime);
            logger.info("TIMER[{}]: {} ms", label, elapsed);
        }
    }

    /**
     * Get formatted timing string.
     *
     * @param label     the operation label
     * @param startTime the nanoTime from start()
     * @return formatted timing string
     */
    public static String formatElapsed(String label, long startTime) {
        long elapsedMs = elapsedMs(startTime);
        if (elapsedMs < 1000) {
            return label + ": " + elapsedMs + "ms";
        } else {
            return String.format("%s: %.2fs", label, elapsedMs / 1000.0);
        }
    }
}