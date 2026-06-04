package com.jtransformer.dataset;

/**
 * Base dataset loader.
 */
public interface DatasetLoader {
    /**
     * Load a text dataset for training or evaluation.
     *
     * @return the loaded TextDataset
     */
    TextDataset loadDataset();
}