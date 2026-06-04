package com.jtransformer.core.quantization;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Binary (1-bit) quantization for extreme compression.
 */
public class BinaryQuantizer {

    private static final Logger logger = LoggerFactory.getLogger(BinaryQuantizer.class);

    public Tensor quantizeToBinary(Tensor tensor) {
        logger.info("Binary quantization applied");
        INDArray x = tensor.getData();

        // scale (alpha) = mean(abs(x))
        double alpha = Transforms.abs(x).meanNumber().doubleValue();

        // sign-based binarization: values > 0 -> 1, else -> -1
        INDArray signFloat = x.gt(0).castTo(DataType.FLOAT).muli(2.0).subi(1.0);

        // store as BYTE (-1 or 1) for compactness
        INDArray signByte = signFloat.castTo(DataType.BYTE);

        logger.info("Binary quantization scale alpha={}", alpha);
        return new Tensor(signByte);
    }
}