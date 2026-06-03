# JTransformer

A Java-based CPU-optimized mini transformer framework using Java 21, Maven, ND4J, Virtual Threads, Quantization and Sparse Attention.

## Features
- Tiny Transformer implementation optimized for CPU
- ND4J tensor operations
- Multithreading with Virtual Threads
- Int8 and Binary Quantization
- Sparse Attention mechanisms
- Training pipeline

## Quick Start
```bash
mvn clean compile
mvn exec:java -Dexec.mainClass="com.jtransformer.Main"
```

## Structure
See docs/architecture.md for details.