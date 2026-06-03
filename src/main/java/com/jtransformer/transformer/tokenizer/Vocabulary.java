package com.jtransformer.transformer.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Vocabulary management.
 */
public class Vocabulary {

    private static final Logger logger = LoggerFactory.getLogger(Vocabulary.class);
    private final Map<String, Integer> tokenToId;

    public Vocabulary() {
        // TODO: Load vocab
        this.tokenToId = Map.of();
    }
}