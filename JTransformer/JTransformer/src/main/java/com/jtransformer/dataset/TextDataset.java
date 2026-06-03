package com.jtransformer.dataset;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * In-memory text dataset.
 */
public class TextDataset {

    private static final Logger logger = LoggerFactory.getLogger(TextDataset.class);
    private final String text;

    public TextDataset(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }
}