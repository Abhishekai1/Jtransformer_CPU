package com.jtransformer.dataset;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * Loads Shakespeare text for training.
 */
public class ShakespeareDatasetLoader implements DatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(ShakespeareDatasetLoader.class);

    private static final String RESOURCE_NAME = "/shakespeare.txt";

    @Override
    public TextDataset loadDataset() {
        logger.info("Loading Shakespeare dataset");
        String text = loadFromResource();
        if (text == null || text.isEmpty()) {
            logger.warn("Shakespeare resource not found, using fallback sample text");
            text = "To be, or not to be, that is the question.";
        }
        return new TextDataset(text);
    }

    private String loadFromResource() {
        try (InputStream is = getClass().getResourceAsStream(RESOURCE_NAME)) {
            if (is == null) {
                logger.warn("Resource {} not found on classpath", RESOURCE_NAME);
                return null;
            }
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                return reader.lines().collect(Collectors.joining("\n"));
            }
        } catch (IOException e) {
            logger.error("Failed to load Shakespeare dataset from resource", e);
            return null;
        }
    }
}