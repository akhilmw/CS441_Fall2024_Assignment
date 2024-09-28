package test

import mapperReducer.WordCountBPEJob
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

object LocalWordCountBPEJob {
  def main(args: Array[String]): Unit = {
    // Create a new Configuration
    val conf = new Configuration()
    conf.set("mapreduce.framework.name", "local")
    conf.set("fs.defaultFS", "file:///")

    // Set up paths
    val inputPath = new Path("/Users/akhilnair/Desktop/CS441_Fall2024_Assignment/src/main/resources/shards/")
    val outputPath = new Path("output")

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true)
    }

    // Set the configuration in WordCountBPEJob
    WordCountBPEJob.runJob(conf, inputPath, outputPath)
  }
}
