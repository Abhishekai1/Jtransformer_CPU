package com.jtransformer.experiments;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Virtual thread scaling experiment.
 */
public class ThreadScalingExperiment {

    private static final Logger logger = LoggerFactory.getLogger(ThreadScalingExperiment.class);

    public void run() {
        logger.info("Thread scaling experiment");
    }
}