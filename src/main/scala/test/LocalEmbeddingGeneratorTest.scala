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
    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))
    val outputFilePath = new Path(outputPath, "part-r-00000") // This is the generated output file
    val renamedFilePath = new Path(outputPath, "embeddings.csv")   // This is the target CSV file name

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true)
    }

    // Run the job
    EmbeddingsGenerator.runJob(conf, inputPath, outputPath)

    println("Embedding generation test completed.")

    if (fs.exists(outputFilePath)) {
      val success = fs.rename(outputFilePath, renamedFilePath)
      if (success) {
        println(s"Renamed part-r-00000 to embeddings.csv successfully.")
      } else {
        println(s"Failed to rename part-r-00000 to output.csv.")
      }
    } else {
      println(s"Output file part-r-00000 does not exist.")
    }
  }
}
