package com.jtransformer.core.math;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.core.tensor.TensorOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Optimized matrix multiplication with quantization awareness.
 */
public class MatrixMultiply {

    private static final Logger logger = LoggerFactory.getLogger(MatrixMultiply.class);

    public Tensor multiply(Tensor a, Tensor b) {
        logger.debug("Matrix multiplication called");
        INDArray da = a.getData();
        INDArray db = b.getData();

        // If tensors are quantized (byte), cast to float for matmul
        if (da.dataType() == DataType.BYTE || db.dataType() == DataType.BYTE) {
            logger.debug("Detected quantized input, casting to FLOAT for matmul");
            INDArray fa = da.castTo(DataType.FLOAT);
            INDArray fb = db.castTo(DataType.FLOAT);
            return TensorOps.matmul(new Tensor(fa), new Tensor(fb));
        }

        return TensorOps.matmul(a, b);
    }
}