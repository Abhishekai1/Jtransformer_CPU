package com.jtransformer.experiments;

import com.jtransformer.core.quantization.BinaryQuantizer;
import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Binary weight transformer experiment.
 */
public class BinaryTransformerExperiment {

    private static final Logger logger = LoggerFactory.getLogger(BinaryTransformerExperiment.class);

    public void run() {
        logger.info("Binary Transformer Experiment started");

        INDArray weights = Nd4j.rand(DataType.FLOAT, 64, 64).sub(0.5).muli(2.0);
        Tensor weightTensor = new Tensor(weights);

        BinaryQuantizer quantizer = new BinaryQuantizer();
        Tensor binaryTensor = quantizer.quantizeToBinary(weightTensor);

        INDArray signOriginal = Transforms.sign(weights);
        INDArray signBinary = binaryTensor.getData().castTo(DataType.FLOAT);
        double mismatch = Transforms.abs(signOriginal.sub(signBinary)).sumNumber().doubleValue() / 2.0;
        double mismatchRatio = mismatch / weights.length();

        logger.info("Binary quantization completed: shape={}, dtype={}", weights.shape(), binaryTensor.getData().dataType());
        logger.info("Sign mismatch ratio between original and binary weights: {}", mismatchRatio);
    }
}