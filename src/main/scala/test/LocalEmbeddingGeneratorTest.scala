package test

import mapperReducer.EmbeddingsGenerator
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

object LocalEmbeddingGeneratorTest {
  def main(args: Array[String]): Unit = {
    // Set up Hadoop configuration
    val conf = new Configuration()
    conf.set("mapreduce.framework.name", "local")
    conf.set("fs.defaultFS", "file:///")

    // Define input and output paths
    val inputPath = new Path("/Users/akhilnair/Desktop/CS441_Fall2024_Assignment/src/main/resources/tokens_shards/")
    val outputPath = new Path("EmbeddingsOutput")

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true)
    }

    // Run the job
    EmbeddingsGenerator.runJob(conf, inputPath, outputPath)

    println("Embedding generation test completed.")
  }
}
