#!/bin/bash
echo "Starting training..."
mvn clean compile exec:java -Dexec.mainClass="com.jtransformer.Main"