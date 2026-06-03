package com.jtransformer.transformer.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base tokenizer interface.
 */
public interface Tokenizer {

    int[] encode(String text);
    String decode(int[] tokens);
}