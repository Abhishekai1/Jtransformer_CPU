package com.jtransformer.dataset;

/**
 * In-memory text dataset.
 */
public class TextDataset {

    private final String text;

    public TextDataset(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }

    /**
     * Check if dataset is empty.
     *
     * @return true if text is null or empty
     */
    public boolean isEmpty() {
        return text == null || text.isEmpty();
    }

    /**
     * Get the size of the dataset (text length).
     *
     * @return text length
     */
    public int size() {
        return text == null ? 0 : text.length();
    }
}