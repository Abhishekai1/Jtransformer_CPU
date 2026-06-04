package com.jtransformer.transformer.embedding;

import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sinusoidal positional encoding.
 */
public class PositionalEncoding {

    private static final Logger logger = LoggerFactory.getLogger(PositionalEncoding.class);

    public Tensor encode(int seqLength, int dim) {
        logger.debug("Computing positional encodings");

        if (seqLength <= 0 || dim <= 0) {
            throw new IllegalArgumentException("seqLength and dim must be positive");
        }

        INDArray positionalEncoding = Nd4j.zeros(DataType.FLOAT, seqLength, dim);

        for (int pos = 0; pos < seqLength; pos++) {
            for (int i = 0; i < dim; i += 2) {
                double angle = pos / Math.pow(10000.0, (double) i / dim);
                positionalEncoding.putScalar(pos, i, Math.sin(angle));
                if (i + 1 < dim) {
                    positionalEncoding.putScalar(pos, i + 1, Math.cos(angle));
                }
            }
        }

        return new Tensor(positionalEncoding);
    }
}