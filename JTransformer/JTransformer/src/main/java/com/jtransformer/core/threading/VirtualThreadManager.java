package com.jtransformer.core.threading;

import com.jtransformer.config.ThreadConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Manages Java 21 Virtual Threads for high concurrency.
 */
public class VirtualThreadManager {

    private static final Logger logger = LoggerFactory.getLogger(VirtualThreadManager.class);
    private final ThreadConfig config;

    public VirtualThreadManager(ThreadConfig config) {
        this.config = config;
    }

    public Thread execute(Runnable task) {
        if (task == null) {
            throw new IllegalArgumentException("Task must not be null");
        }

        // Use standard threads (Java 11 compatible) instead of virtual threads
        Thread thread = new Thread(task);
        thread.start();
        logger.debug("Thread started: {}", thread.getName());
        return thread;
    }

    public List<Thread> executeAll(Runnable... tasks) {
        if (tasks == null || tasks.length == 0) {
            logger.warn("No tasks provided to executeAll");
            return Collections.emptyList();
        }

        int maxThreads = config != null ? config.getNumVirtualThreads() : -1;
        if (maxThreads > 0 && tasks.length > maxThreads) {
            logger.warn("Starting {} virtual threads, which exceeds configured numVirtualThreads={}", tasks.length, maxThreads);
        }

        List<Thread> threads = new ArrayList<>(tasks.length);
        for (Runnable task : tasks) {
            if (task == null) {
                logger.warn("Skipping null task");
                continue;
            }
            threads.add(execute(task));
        }
        return threads;
    }

    public void executeAndJoin(Runnable... tasks) throws InterruptedException {
        List<Thread> threads = executeAll(tasks);
        for (Thread thread : threads) {
            if (thread != null) {
                thread.join();
            }
        }
        logger.info("All virtual threads have completed");
    }
}