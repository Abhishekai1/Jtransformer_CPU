package com.jtransformer.transformer.embedding;

import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Token embedding layer.
 */
public class TokenEmbedding {

    private static final Logger logger = LoggerFactory.getLogger(TokenEmbedding.class);
    private final int vocabSize;
    private final int dim;
    private final INDArray embeddingTable;

    public TokenEmbedding(int vocabSize, int dim) {
        if (vocabSize <= 0 || dim <= 0) {
            throw new IllegalArgumentException("Vocabulary size and embedding dimension must be positive");
        }
        this.vocabSize = vocabSize;
        this.dim = dim;
        this.embeddingTable = Nd4j.randn(DataType.FLOAT, vocabSize, dim).mul(0.02f);
        logger.debug("Created token embedding table vocabSize={} dim={}", vocabSize, dim);
    }

    public TokenEmbedding(int dim) {
        this(65536, dim);
    }

    public Tensor embed(int[] tokens) {
        logger.debug("Generating token embeddings for {} tokens", tokens == null ? 0 : tokens.length);
        if (tokens == null) {
            throw new IllegalArgumentException("Token array must not be null");
        }

        INDArray output = Nd4j.zeros(DataType.FLOAT, tokens.length, dim);
        for (int i = 0; i < tokens.length; i++) {
            int tokenId = tokens[i];
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException("Token id " + tokenId + " is out of bounds for vocab size " + vocabSize);
            }
            output.putRow(i, embeddingTable.getRow(tokenId));
        }

        return new Tensor(output);
    }
}