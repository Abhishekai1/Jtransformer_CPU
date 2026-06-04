package com.jtransformer.transformer.tokenizer;

/**
 * Base tokenizer interface.
 */
public interface Tokenizer {

    /**
     * Encode a string into token ids.
     * @param text input text, must not be null
     * @return array of token ids
     */
    int[] encode(String text);

    /**
     * Decode token ids back to a string.
     * @param tokens token id array, must not be null
     * @return decoded string
     */
    String decode(int[] tokens);

    /**
     * Convenience wrapper that validates input before encoding.
     */
    default int[] encodeChecked(String text) {
        if (text == null) throw new IllegalArgumentException("text must not be null");
        return encode(text);
    }

    /**
     * Convenience wrapper that validates input before decoding.
     */
    default String decodeChecked(int[] tokens) {
        if (tokens == null) throw new IllegalArgumentException("tokens must not be null");
        return decode(tokens);
    }
}