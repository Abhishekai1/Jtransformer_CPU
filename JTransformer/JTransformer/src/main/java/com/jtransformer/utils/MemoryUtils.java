package com.jtransformer.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Memory management utilities.
 */
public class MemoryUtils {

    private static final Logger logger = LoggerFactory.getLogger(MemoryUtils.class);

    /**
     * Get current used memory in MB.
     *
     * @return used memory in MB
     */
    public static long getUsedMemoryMB() {
        Runtime rt = Runtime.getRuntime();
        return (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
    }

    /**
     * Get total allocated memory in MB.
     *
     * @return total memory in MB
     */
    public static long getTotalMemoryMB() {
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() / (1024 * 1024);
    }

    /**
     * Get free memory in MB.
     *
     * @return free memory in MB
     */
    public static long getFreeMemoryMB() {
        Runtime rt = Runtime.getRuntime();
        return rt.freeMemory() / (1024 * 1024);
    }

    /**
     * Get maximum available memory in MB.
     *
     * @return max memory in MB
     */
    public static long getMaxMemoryMB() {
        Runtime rt = Runtime.getRuntime();
        return rt.maxMemory() / (1024 * 1024);
    }

    /**
     * Get memory usage as percentage of max memory.
     *
     * @return percentage 0-100
     */
    public static double getMemoryUsagePercent() {
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        long max = rt.maxMemory();
        return (used * 100.0) / max;
    }

    /**
     * Trigger garbage collection explicitly.
     */
    public static void forceGarbageCollection() {
        System.gc();
        System.runFinalization();
        logger.debug("Garbage collection triggered");
    }

    /**
     * Log current memory statistics.
     */
    public static void logMemoryStats() {
        long used = getUsedMemoryMB();
        long total = getTotalMemoryMB();
        long max = getMaxMemoryMB();
        double percent = getMemoryUsagePercent();
        logger.info("Memory: Used={}MB, Total={}MB, Max={}MB, Usage={:.1f}%",
                used, total, max, percent);
    }

    /**
     * Get formatted memory info string.
     *
     * @return memory info string
     */
    public static String getMemoryInfo() {
        long used = getUsedMemoryMB();
        long total = getTotalMemoryMB();
        long max = getMaxMemoryMB();
        double percent = getMemoryUsagePercent();
        return String.format("Memory: Used=%dMB, Total=%dMB, Max=%dMB, Usage=%.1f%%",
                used, total, max, percent);
    }
}