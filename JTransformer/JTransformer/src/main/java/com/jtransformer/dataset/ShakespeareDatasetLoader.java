package com.jtransformer.dataset;

import com.jtransformer.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads Shakespeare text for training.
 */
public class ShakespeareDatasetLoader implements DatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(ShakespeareDatasetLoader.class);

    @Override
    public TextDataset loadDataset() {
        logger.info("Loading Shakespeare dataset");
        // TODO: Load from resources
        return new TextDataset("To be or not to be...");
    }
}