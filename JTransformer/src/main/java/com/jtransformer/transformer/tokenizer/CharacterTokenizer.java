package com.jtransformer.transformer.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple character-level tokenizer for research.
 */
public class CharacterTokenizer implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(CharacterTokenizer.class);

    @Override
    public int[] encode(String text) {
        logger.debug("Encoding text of length {}", text.length());
        int[] tokens = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            tokens[i] = text.charAt(i);
        }
        return tokens;
    }

    @Override
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int t : tokens) sb.append((char) t);
        return sb.toString();
    }
}