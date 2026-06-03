package com.jtransformer.dataset;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base dataset loader.
 */
public interface DatasetLoader {
    TextDataset loadDataset();
}