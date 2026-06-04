package com.jtransformer.core.quantization;

import com.jtransformer.core.tensor.QuantizedTensor;
import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Int8 Quantizer for weights and activations.
 */
public class Int8Quantizer {

    private static final Logger logger = LoggerFactory.getLogger(Int8Quantizer.class);

    public QuantizedTensor quantize(Tensor tensor) {
        logger.info("Quantizing to INT8");
        INDArray x = tensor.getData();
        long len = x.length();

        double min = x.minNumber().doubleValue();
        double max = x.maxNumber().doubleValue();

        int qmin = -128;
        int qmax = 127;

        double scale = (max - min) / (double) (qmax - qmin);
        if (scale == 0.0) scale = 1e-8;

        double zeroPointReal = qmin - min / scale;
        int zeroPoint = (int) Math.round(zeroPointReal);
        if (zeroPoint < qmin) zeroPoint = qmin;
        if (zeroPoint > qmax) zeroPoint = qmax;

        byte[] out = new byte[(int) len];
        for (int i = 0; i < (int) len; i++) {
            double v = x.getDouble(i);
            int q = (int) Math.round(v / scale + zeroPoint);
            if (q < qmin) q = qmin;
            if (q > qmax) q = qmax;
            out[i] = (byte) q;
        }

        return new QuantizedTensor(out, (float) scale, (float) zeroPoint, x.shape());
    }
}