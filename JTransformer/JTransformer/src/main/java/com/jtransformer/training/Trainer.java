package com.jtransformer.training;

import com.jtransformer.core.tensor.Tensor;
import com.jtransformer.dataset.BatchGenerator;
import com.jtransformer.dataset.TextDataset;
import com.jtransformer.transformer.model.TinyTransformer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main training orchestrator.
 */
public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);
    private final TinyTransformer model;
    private final Optimizer optimizer;
    private final LossFunction lossFunction;
    private final BackPropagation backPropagation;
    private final GradientAccumulator accumulator;
    private final BatchGenerator batchGenerator;

    public Trainer(TinyTransformer model, com.jtransformer.config.AppConfig config) {
        this.model = model;
        this.optimizer = new AdamOptimizer();
        this.lossFunction = new LossFunction();
        this.backPropagation = new BackPropagation();
        this.accumulator = new GradientAccumulator();
        this.batchGenerator = new BatchGenerator();
    }

    public void train(TextDataset dataset, int epochs) {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset must not be null");
        }

        logger.info("Starting training for {} epochs", epochs);
        int batchSize = 4;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Tensor[] batch = batchGenerator.nextBatch(dataset, batchSize);
            if (batch.length == 0) {
                logger.warn("No training batches generated; aborting training");
                break;
            }

            double epochLoss = 0.0;
            int batchCount = 0;

            for (Tensor input : batch) {
                Tensor predictions = model.forward(input);
                Tensor targets = input;
                Tensor loss = lossFunction.compute(predictions, targets);
                backPropagation.backward(loss);
                Tensor gradients = backPropagation.getGradients();
                if (gradients != null) {
                    accumulator.accumulate(gradients);
                }
                epochLoss += loss.getData().getDouble(0);
                batchCount++;
            }

            Tensor averageGradient = accumulator.getAverageGradient();
            if (averageGradient != null) {
                optimizer.step(averageGradient);
            }

            accumulator.reset();
            double averageEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
            logger.info("Epoch {} completed - average loss={}", epoch, averageEpochLoss);
        }
    }
}