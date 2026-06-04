package com.jtransformer;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.transformer.attention.MultiHeadAttention;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class AttentionTests {

    @Test
    void testMultiHeadAttention() {
        MultiHeadAttention attention = new MultiHeadAttention(2);

        Tensor query = new Tensor(Nd4j.create(new double[][]{
                {1.0, 0.0, 1.0, 0.0},
                {0.5, 0.5, 0.5, 0.5}
        }));

        Tensor key = new Tensor(Nd4j.create(new double[][]{
                {1.0, 0.0, 1.0, 0.0},
                {0.5, 0.5, 0.5, 0.5}
        }));

        Tensor value = new Tensor(Nd4j.create(new double[][]{
                {0.1, 0.2, 0.3, 0.4},
                {1.0, 1.0, 1.0, 1.0}
        }));

        Tensor output = attention.forward(query, key, value, null);

        assertNotNull(output, "Attention output should not be null");
        assertEquals(2, output.rank(), "Output should be a 2D tensor");
        assertArrayEquals(new long[]{2, 4}, output.shape(), "Output shape should match input feature dimension");
        assertFalse(Double.isNaN(output.getData().getDouble(0)), "Output values should be numeric");
        assertFalse(Double.isInfinite(output.getData().getDouble(0)), "Output values should be finite");
        assertTrue(output.getData().sumNumber().doubleValue() > 0.0, "Output should contain positive values");
    }
}