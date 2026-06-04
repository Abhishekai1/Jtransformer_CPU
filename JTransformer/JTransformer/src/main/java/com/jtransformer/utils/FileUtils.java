package com.jtransformer.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * File utilities.
 */
public class FileUtils {

    private static final Logger logger = LoggerFactory.getLogger(FileUtils.class);

    public static String readFile(String path) throws IOException {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("File path must not be null or blank");
        }
        logger.debug("Reading file: {}", path);
        return Files.readString(Paths.get(path), StandardCharsets.UTF_8);
    }

    public static void writeFile(String path, String content) throws IOException {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("File path must not be null or blank");
        }
        if (content == null) {
            throw new IllegalArgumentException("Content must not be null");
        }
        logger.debug("Writing file: {}", path);
        Path p = Paths.get(path);
        Files.createDirectories(p.getParent());
        Files.writeString(p, content, StandardCharsets.UTF_8);
    }

    public static List<String> readLines(String path) throws IOException {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("File path must not be null or blank");
        }
        logger.debug("Reading lines from file: {}", path);
        return Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
    }

    public static boolean fileExists(String path) {
        if (path == null || path.isBlank()) {
            return false;
        }
        return Files.exists(Paths.get(path));
    }
}