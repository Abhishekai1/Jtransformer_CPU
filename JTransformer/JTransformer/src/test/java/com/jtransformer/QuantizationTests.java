package com.jtransformer;

import com.jtransformer.core.quantization.Int8Quantizer;
import com.jtransformer.core.tensor.QuantizedTensor;
import com.jtransformer.core.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.*;

class QuantizationTests {

    @Test
    void testInt8Quantization() {
        Int8Quantizer quantizer = new Int8Quantizer();
        Tensor original = new Tensor(Nd4j.create(new double[][]{
                {0.1, -0.5, 1.2, 2.4},
                {-0.8, 0.0, 0.7, -1.3}
        }));

        QuantizedTensor quantized = quantizer.quantize(original);
        assertNotNull(quantized, "Quantized tensor should not be null");
        assertArrayEquals(original.shape(), quantized.getShape(), "Quantized tensor shape should preserve original dimensions");
        assertTrue(quantized.getScale() > 0.0f, "Quantization scale should be positive");

        Tensor dequantized = quantized.dequantize();
        assertNotNull(dequantized, "Dequantized tensor should not be null");
        assertArrayEquals(original.shape(), dequantized.shape(), "Dequantized tensor should preserve original shape");

        double maxError = Transforms.abs(original.getData().sub(dequantized.getData())).maxNumber().doubleValue();
        assertTrue(maxError <= quantized.getScale() * 1.5,
                "Max quantization error should remain within expected scale bounds");
    }
}