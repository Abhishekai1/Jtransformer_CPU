package com.jtransformer.core.quantization;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility methods for quantization.
 */
public class QuantizationUtils {

    private static final Logger logger = LoggerFactory.getLogger(QuantizationUtils.class);

    public static float calculateScale(float maxVal) {
        if (maxVal <= 0.0f) {
            logger.warn("calculateScale called with non-positive maxVal={}", maxVal);
            return 1.0f;
        }

        // symmetric int8 range [-127,127]
        float scale = maxVal / 127.0f;
        if (scale == 0.0f) scale = 1e-8f;
        return scale;
    }
}