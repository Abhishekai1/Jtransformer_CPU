package com.jtransformer.transformer.model;

/**
 * Model hyperparameters.
 */
public class ModelConfig {

    private final int dim;
    private final int heads;
    private final int layers;
    private final int seqLength;

    public ModelConfig(int dim, int heads, int layers, int seqLength) {
        if (dim <= 0) {
            throw new IllegalArgumentException("Model dimension must be positive");
        }
        if (heads <= 0) {
            throw new IllegalArgumentException("Number of attention heads must be positive");
        }
        if (layers <= 0) {
            throw new IllegalArgumentException("Number of layers must be positive");
        }
        if (seqLength <= 0) {
            throw new IllegalArgumentException("Sequence length must be positive");
        }

        this.dim = dim;
        this.heads = heads;
        this.layers = layers;
        this.seqLength = seqLength;
    }

    public int getDim() { return dim; }
    public int getHeads() { return heads; }
    public int getLayers() { return layers; }
    public int getSeqLength() { return seqLength; }
}