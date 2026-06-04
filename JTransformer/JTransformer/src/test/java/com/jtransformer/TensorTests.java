package com.jtransformer;

import com.jtransformer.core.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class TensorTests {

    @Test
    void testTensorCreation() {
        Tensor tensor = new Tensor(Nd4j.create(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        }));

        assertNotNull(tensor, "Tensor should be created successfully");
        assertArrayEquals(new long[]{2, 2}, tensor.shape(), "Tensor shape should be [2, 2]");
        assertEquals(2, tensor.rank(), "Tensor should have rank 2");
        assertEquals(4, tensor.size(), "Tensor should contain 4 elements");
    }

    @Test
    void testTensorMathOperations() {
        Tensor left = new Tensor(Nd4j.create(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        }));
        Tensor right = new Tensor(Nd4j.create(new double[][]{
                {0.5, 1.0},
                {1.5, 2.0}
        }));

        Tensor sum = left.add(right);
        assertArrayEquals(new long[]{2, 2}, sum.shape(), "Sum result should preserve shape");
        assertEquals(1.5, sum.getData().getDouble(0, 0), 1e-6);
        assertEquals(3.0, sum.getData().getDouble(0, 1), 1e-6);

        Tensor product = left.multiply(2.0);
        assertArrayEquals(new long[]{2, 2}, product.shape(), "Scalar multiply should preserve shape");
        assertEquals(2.0, product.getData().getDouble(0, 0), 1e-6);
        assertEquals(8.0, product.getData().getDouble(1, 1), 1e-6);

        Tensor difference = left.subtract(right);
        assertEquals(0.5, difference.getData().getDouble(0, 0), 1e-6);
        assertEquals(2.0, difference.getData().getDouble(1, 1), 1e-6);

        assertEquals(2.5, left.mean(), 1e-6, "Mean should equal 2.5");
        assertEquals(10.0, left.sum(), 1e-6, "Sum should equal 10.0");
    }
}
