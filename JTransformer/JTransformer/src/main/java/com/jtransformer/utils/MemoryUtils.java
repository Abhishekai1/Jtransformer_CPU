package com.jtransformer.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Memory management utilities.
 */
public class MemoryUtils {

    private static final Logger logger = LoggerFactory.getLogger(MemoryUtils.class);

    public static long getUsedMemoryMB() {
        Runtime rt = Runtime.getRuntime();
        return (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
    }
}