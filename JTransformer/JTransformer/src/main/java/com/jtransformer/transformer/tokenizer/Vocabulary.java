package com.jtransformer.transformer.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Vocabulary management. By default builds a simple character->id mapping
 * covering the Unicode BMP (0..65535). Optionally can load a vocabulary
 * from a newline-separated token file where each line is a token.
 */
public class Vocabulary {

    private static final Logger logger = LoggerFactory.getLogger(Vocabulary.class);
    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;

    public Vocabulary() {
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        buildDefaultCharVocab();
        logger.debug("Built default character vocabulary with size={}", tokenToId.size());
    }

    public Vocabulary(Path vocabFile) throws IOException {
        if (vocabFile == null) throw new IllegalArgumentException("vocabFile must not be null");
        Map<String, Integer> tmp = new HashMap<>();
        Map<Integer, String> rev = new HashMap<>();
        try (BufferedReader r = Files.newBufferedReader(vocabFile, StandardCharsets.UTF_8)) {
            String line;
            int idx = 0;
            while ((line = r.readLine()) != null) {
                if (line.isBlank()) continue;
                tmp.put(line, idx);
                rev.put(idx, line);
                idx++;
            }
        }
        this.tokenToId = Collections.unmodifiableMap(tmp);
        this.idToToken = Collections.unmodifiableMap(rev);
        logger.info("Loaded vocabulary from {} size={}", vocabFile, tokenToId.size());
    }

    private void buildDefaultCharVocab() {
        // map single-character strings to their code point id (0..65535)
        for (int i = 0; i <= Character.MAX_VALUE; i++) {
            String s = String.valueOf((char) i);
            tokenToId.put(s, i);
            idToToken.put(i, s);
        }
    }

    public Integer getId(String token) {
        return tokenToId.get(token);
    }

    public String getToken(int id) {
        return idToToken.get(id);
    }

    public int size() {
        return tokenToId.size();
    }
}