package com.jtransformer.core.tensor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Tensor with quantization support for memory optimization.
 */
public class QuantizedTensor {

    private static final Logger logger = LoggerFactory.getLogger(QuantizedTensor.class);
    private final byte[] quantizedData;
    private final float scale;
    private final float zeroPoint;
    private final long[] shape;

    public QuantizedTensor(byte[] quantizedData, float scale, float zeroPoint, long... shape) {
        this.quantizedData = quantizedData;
        this.scale = scale;
        this.zeroPoint = zeroPoint;
        this.shape = shape != null ? shape.clone() : new long[0];
    }

    public float getScale() {
        return scale;
    }

    public float getZeroPoint() {
        return zeroPoint;
    }

    public long[] getShape() {
        return shape.clone();
    }

    public Tensor dequantize() {
        logger.info("Dequantizing tensor");
        if (quantizedData == null || quantizedData.length == 0) {
            logger.warn("No quantized data to dequantize");
            return null;
        }

        INDArray qArr = Nd4j.createFromArray(quantizedData).castTo(DataType.FLOAT);
        if (shape.length > 0) {
            qArr = qArr.reshape(shape);
        }

        INDArray deq = qArr.sub(zeroPoint).muli(scale);
        return new Tensor(deq);
    }
}