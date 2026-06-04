package com.jtransformer.experiments;

import com.jtransformer.core.quantization.Int8Quantizer;
import com.jtransformer.core.tensor.QuantizedTensor;
import com.jtransformer.core.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * INT8 quantization experiment.
 */
public class Int8Experiment {

    private static final Logger logger = LoggerFactory.getLogger(Int8Experiment.class);

    public void run() {
        logger.info("INT8 Experiment started");

        INDArray weights = Nd4j.rand(DataType.FLOAT, 128, 128).sub(0.5).muli(2.0);
        Tensor weightTensor = new Tensor(weights);

        Int8Quantizer quantizer = new Int8Quantizer();
        QuantizedTensor quantized = quantizer.quantize(weightTensor);
        Tensor dequantized = quantized.dequantize();

        if (dequantized == null) {
            logger.error("Dequantization failed");
            return;
        }

        INDArray diff = weights.sub(dequantized.getData());
        double mse = diff.mul(diff).meanNumber().doubleValue();
        double maxError = Transforms.abs(diff).maxNumber().doubleValue();

        logger.info("Quantized INT8 tensor shape: {}", weights.shape());
        logger.info("Quantization scale: {}", quantized.getScale());
        logger.info("Quantization zero point: {}", quantized.getZeroPoint());
        logger.info("Reconstruction MSE: {}", mse);
        logger.info("Max absolute error: {}", maxError);
    }
}