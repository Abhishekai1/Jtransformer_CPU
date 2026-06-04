package com.jtransformer.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Logging helpers for structured and performance logging.
 */
public class LoggerUtils {

    private static final Logger logger = LoggerFactory.getLogger(LoggerUtils.class);

    /**
     * Log a debug message.
     *
     * @param message the debug message
     */
    public static void logDebug(String message) {
        if (message != null) {
            logger.debug(message);
        }
    }

    /**
     * Log a debug message with format arguments.
     *
     * @param format the format string
     * @param args   the format arguments
     */
    public static void logDebug(String format, Object... args) {
        if (format != null && args != null) {
            logger.debug(format, args);
        }
    }

    /**
     * Log an info message.
     *
     * @param message the info message
     */
    public static void logInfo(String message) {
        if (message != null) {
            logger.info(message);
        }
    }

    /**
     * Log an info message with format arguments.
     *
     * @param format the format string
     * @param args   the format arguments
     */
    public static void logInfo(String format, Object... args) {
        if (format != null && args != null) {
            logger.info(format, args);
        }
    }

    /**
     * Log a warning message.
     *
     * @param message the warning message
     */
    public static void logWarn(String message) {
        if (message != null) {
            logger.warn(message);
        }
    }

    /**
     * Log a warning message with format arguments.
     *
     * @param format the format string
     * @param args   the format arguments
     */
    public static void logWarn(String format, Object... args) {
        if (format != null && args != null) {
            logger.warn(format, args);
        }
    }

    /**
     * Log an error message.
     *
     * @param message the error message
     */
    public static void logError(String message) {
        if (message != null) {
            logger.error(message);
        }
    }

    /**
     * Log an error message with exception.
     *
     * @param message   the error message
     * @param exception the exception
     */
    public static void logError(String message, Throwable exception) {
        if (message != null && exception != null) {
            logger.error(message, exception);
        }
    }

    /**
     * Log an error message with format arguments.
     *
     * @param format the format string
     * @param args   the format arguments
     */
    public static void logError(String format, Object... args) {
        if (format != null && args != null) {
            logger.error(format, args);
        }
    }

    /**
     * Log performance metrics (timing information).
     *
     * @param label    the operation label
     * @param duration the duration in milliseconds
     */
    public static void logPerformance(String label, long duration) {
        if (label != null) {
            logger.info("PERF: {} completed in {} ms", label, duration);
        }
    }

    /**
     * Log model-related information.
     *
     * @param modelName the model name
     * @param metric    the metric name
     * @param value     the metric value
     */
    public static void logModelMetric(String modelName, String metric, Object value) {
        if (modelName != null && metric != null && value != null) {
            logger.info("MODEL[{}]: {} = {}", modelName, metric, value);
        }
    }
}