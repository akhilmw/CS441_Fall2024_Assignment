package test

import mapperReducer.SemanticRepresentation
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

object LocalSemanticRepresentationJob {
  def main(args: Array[String]): Unit = {
    // Create a new Configuration
    val conf = new Configuration()
    conf.set("mapreduce.framework.name", "local")
    conf.set("fs.defaultFS", "file:///")

    // Set up paths for the input embeddings and output
    val inputPath = new Path(args(0)) // Ensure this file is present in the current directory
    val outputPath = new Path(args(1))
    val outputFilePath = new Path(outputPath, "part-r-00000") // This is the generated output file
    val renamedFilePath = new Path(outputPath, "semantics.yaml")   // This is the target CSV file name

    // Set the topN value for finding similar words (e.g., top 5 similar words)
    val topN = 5
    conf.setInt("topN", topN)

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true)
    }

    // Run the SemanticRepresentation job
    SemanticRepresentation.runJob(conf, inputPath, outputPath)

    println(s"Semantic representation job completed. Results are stored in $outputPath.")

    if (fs.exists(outputFilePath)) {
      val success = fs.rename(outputFilePath, renamedFilePath)
      if (success) {
        println(s"Renamed part-r-00000 to semantics.yaml successfully.")
      } else {
        println(s"Failed to rename part-r-00000 to semantics.yaml.")
      }
    } else {
      println(s"Output file part-r-00000 does not exist.")
    }
  }
}
