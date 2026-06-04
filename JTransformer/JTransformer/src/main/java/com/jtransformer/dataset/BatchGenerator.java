package com.jtransformer.dataset;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.tokenizer.CharacterTokenizer;
import com.jtransformer.transformer.tokenizer.Tokenizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Generates batches for training.
 */
public class BatchGenerator {

    private static final Logger logger = LoggerFactory.getLogger(BatchGenerator.class);
    private static final int DEFAULT_SEQUENCE_LENGTH = 64;

    public Tensor[] nextBatch(TextDataset dataset, int batchSize) {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset must not be null");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }

        String text = dataset.getText();
        if (text == null || text.isEmpty()) {
            logger.warn("Dataset text is empty");
            return new Tensor[0];
        }

        Tokenizer tokenizer = new CharacterTokenizer();
        int[] tokens = tokenizer.encode(text);
        int seqLength = Math.min(DEFAULT_SEQUENCE_LENGTH, tokens.length);
        List<Tensor> batch = new ArrayList<>(batchSize);

        if (tokens.length < seqLength) {
            INDArray arr = Nd4j.createFromArray(tokens).reshape(1, tokens.length);
            batch.add(new Tensor(arr));
            return batch.toArray(new Tensor[0]);
        }

        int maxBatches = tokens.length / seqLength;
        int actualBatchSize = Math.min(batchSize, maxBatches);
        for (int i = 0; i < actualBatchSize; i++) {
            int start = i * seqLength;
            int end = start + seqLength;
            int[] slice = Arrays.copyOfRange(tokens, start, end);
            INDArray arr = Nd4j.createFromArray(slice).reshape(1, seqLength);
            batch.add(new Tensor(arr));
        }

        logger.debug("Generated batch of {} sequences, sequence length={}", batch.size(), seqLength);
        return batch.toArray(new Tensor[0]);
    }
}