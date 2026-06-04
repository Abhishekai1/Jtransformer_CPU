package com.jtransformer.core.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Core Tensor abstraction using ND4J INDArray.
 */
public class Tensor {

    private static final Logger logger = LoggerFactory.getLogger(Tensor.class);
    private INDArray data;

    public Tensor(INDArray data) {
        this.data = data;
        logger.debug("Tensor created with shape: {}", data.shape());
    }

    public INDArray getData() {
        return data;
    }

    /**
     * Get shape of the tensor.
     *
     * @return shape array
     */
    public long[] shape() {
        return data.shape();
    }

    /**
     * Get number of dimensions.
     *
     * @return rank
     */
    public int rank() {
        return data.rank();
    }

    /**
     * Get total number of elements.
     *
     * @return size
     */
    public long size() {
        return data.length();
    }

    /**
     * Add another tensor element-wise.
     *
     * @param other the tensor to add
     * @return result tensor
     */
    public Tensor add(Tensor other) {
        if (other == null) {
            throw new IllegalArgumentException("Other tensor must not be null");
        }
        return new Tensor(this.data.add(other.getData()));
    }

    /**
     * Subtract another tensor element-wise.
     *
     * @param other the tensor to subtract
     * @return result tensor
     */
    public Tensor subtract(Tensor other) {
        if (other == null) {
            throw new IllegalArgumentException("Other tensor must not be null");
        }
        return new Tensor(this.data.sub(other.getData()));
    }

    /**
     * Multiply by a scalar.
     *
     * @param scalar the scalar value
     * @return result tensor
     */
    public Tensor multiply(double scalar) {
        return new Tensor(this.data.mul(scalar));
    }

    /**
     * Element-wise multiplication with another tensor.
     *
     * @param other the tensor to multiply
     * @return result tensor
     */
    public Tensor multiply(Tensor other) {
        if (other == null) {
            throw new IllegalArgumentException("Other tensor must not be null");
        }
        return new Tensor(this.data.mul(other.getData()));
    }

    /**
     * Divide by a scalar.
     *
     * @param scalar the scalar value
     * @return result tensor
     */
    public Tensor divide(double scalar) {
        if (scalar == 0) {
            throw new IllegalArgumentException("Cannot divide by zero");
        }
        return new Tensor(this.data.div(scalar));
    }

    /**
     * Element-wise division with another tensor.
     *
     * @param other the tensor to divide by
     * @return result tensor
     */
    public Tensor divide(Tensor other) {
        if (other == null) {
            throw new IllegalArgumentException("Other tensor must not be null");
        }
        return new Tensor(this.data.div(other.getData()));
    }

    /**
     * Transpose the tensor (swap last two dimensions for 2D).
     *
     * @return transposed tensor
     */
    public Tensor transpose() {
        return new Tensor(this.data.transpose());
    }

    /**
     * Reshape tensor to new shape.
     *
     * @param newShape the new shape
     * @return reshaped tensor
     */
    public Tensor reshape(long... newShape) {
        if (newShape == null || newShape.length == 0) {
            throw new IllegalArgumentException("New shape must not be null or empty");
        }
        return new Tensor(this.data.reshape(newShape));
    }

    /**
     * Flatten tensor to 1D.
     *
     * @return flattened tensor
     */
    public Tensor flatten() {
        return new Tensor(this.data.reshape(-1));
    }

    /**
     * Sum all elements.
     *
     * @return sum as double
     */
    public double sum() {
        return this.data.sumNumber().doubleValue();
    }

    /**
     * Get mean of all elements.
     *
     * @return mean as double
     */
    public double mean() {
        return this.data.meanNumber().doubleValue();
    }

    /**
     * Clone the tensor.
     *
     * @return cloned tensor
     */
    public Tensor clone() {
        return new Tensor(this.data.dup());
    }

    /**
     * Get string representation.
     *
     * @return string representation
     */
    @Override
    public String toString() {
        return "Tensor" + java.util.Arrays.toString(shape()) + ":" + data.toString();
    }
}