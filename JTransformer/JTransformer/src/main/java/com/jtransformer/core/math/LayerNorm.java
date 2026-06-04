package com.jtransformer.core.math;

import com.jtransformer.core.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;

/**
 * Layer Normalization implementation.
 */
public class LayerNorm {

    private static final Logger logger = LoggerFactory.getLogger(LayerNorm.class);

    public Tensor normalize(Tensor input) {
        logger.debug("Layer normalization");
        INDArray x = input.getData();
        int rank = x.rank();
        int lastDim = rank - 1;

        // mean over last dimension
        INDArray mean = x.mean(lastDim);

        // variance over last dimension: E[(x-mean)^2]
        INDArray centered = x.sub(mean);
        INDArray var = centered.mul(centered).mean(lastDim);

        double eps = 1e-5;
        INDArray std = Transforms.sqrt(var.add(eps));

        // normalize: (x - mean) / std
        INDArray normalized = centered.div(std);

        return new Tensor(normalized);
    }
}