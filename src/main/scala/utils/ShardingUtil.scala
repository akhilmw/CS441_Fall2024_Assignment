package utils

import java.io.{File, IOException, PrintWriter}
import scala.collection.mutable.ListBuffer
import scala.io.Source
import com.typesafe.config.ConfigFactory


class ShardingUtil {

  private val config = ConfigFactory.load()
  private val datasetPath: String = config.getString("filePaths.datasetPath")
  private val shardPath: String = config.getString("filePaths.shardPath")
  private val orderedTokensPath: String = config.getString("filePaths.orderedTokensPath")
  private val orderedCsvPath: String = config.getString("filePaths.orderedCsvPath")
  val embeddingsPath: String = config.getString("filePaths.embeddingsPath")
  val embeddingsCsvPath: String = config.getString("filePaths.embeddingsCsvPath")
  val orderedTokenInputPath: String = config.getString("filePaths.orderedTokenInputPath")
  val ordered_tokens_shards: String = config.getString("filePaths.ordered_tokens_shards")



  /**
   * Deletes existing shard files in the specified output directory.
   *
   * @param outputDir The directory where shard files are stored.
   */
  private def deleteExistingShards(outputDir: String): Unit = {
    val directory = new File(outputDir)
    if (directory.exists() && directory.isDirectory) {
      // Delete files that start with "shard"
      directory.listFiles()
        .filter(file => file.getName.startsWith("shard"))
        .foreach(_.delete())
    }
  }

  /**
   * Cleans a line of text by removing unwanted characters, keeping only words,
   * numeric, alphanumeric, and apostrophes, and trims the result.
   *
   * @param line The input line of text.
   * @return A cleaned version of the line.
   */
  private def cleanText(line: String): String = {
    line.replaceAll("[^\\w\\s]", "").trim // Removing non-word characters and extra spaces
  }

  /**
   * Shards both text and CSV content based on the specified file type.
   *
   * @param inputFilePath The path of the input file to be sharded.
   * @param outputDir     The directory where the shard files will be stored.
   * @param shardSize     The number of lines each shard should contain.
   * @param isCSV         Boolean indicating if the file is a CSV.
   */
  private def shardFile(inputFilePath: String, outputDir: String, shardSize: Int, isCSV: Boolean): Unit = {
    // Clean up any existing shards before creating new ones
    deleteExistingShards(outputDir)

    val source = Source.fromFile(inputFilePath)
    val lines = source.getLines()

    var shardIndex = 0
    var currentShardWriter = createShardWriter(outputDir, shardIndex, isCSV)
    var lineCounter = 0

    // Process each line from the input file
    lines.foreach { line =>
      val cleanedLine = if (isCSV) line else cleanText(line)

      if (lineCounter >= shardSize) {
        // Close the current shard and start a new one
        currentShardWriter.close()
        shardIndex += 1
        currentShardWriter = createShardWriter(outputDir, shardIndex, isCSV)
        lineCounter = 0
      }

      currentShardWriter.println(cleanedLine)
      lineCounter += 1
    }

    // Close any open resources
    currentShardWriter.close()
    source.close()
  }

  /**
   * Creates a new PrintWriter for a shard file with the specified index and type.
   *
   * @param outputDir The directory where the shard files will be stored.
   * @param index     The index of the shard.
   * @param isCSV     Boolean indicating if the file is a CSV.
   * @return A PrintWriter for the shard file.
   */
  private def createShardWriter(outputDir: String, index: Int, isCSV: Boolean): PrintWriter = {
    val dir = new File(outputDir)

    // Check if the directory exists, if not, create it
    if (!dir.exists()) {
      val created = dir.mkdirs() // Create the directory and any necessary parent directories
      if (!created) {
        throw new IOException(s"Failed to create directory: $outputDir")
      }
    }
    val extension = if (isCSV) "csv" else "txt"
    new PrintWriter(new File(s"$outputDir/shard_$index.$extension"))
  }

  /**
   * Public method to shard either a text dataset or a CSV embeddings file.
   *
   * @param isDataset Boolean indicating whether the input file is a text dataset.
   */
  def shardTextOrCSV(isDataset: Boolean, isTokens: Boolean, shardSize: Int): Unit = {
    val (inputFilePath, outputDir, isCSV) = if (isDataset && isTokens) {
      (
        orderedTokensPath,orderedCsvPath, true
      )
    }
    else if (isDataset && !isTokens) {
      (datasetPath, shardPath, false
      )
    } else if (!isDataset && isTokens) {
      (orderedTokenInputPath, ordered_tokens_shards, true)
    }
    else {
      (
        embeddingsPath,
        embeddingsCsvPath,
        true
      )
    }
      shardFile(inputFilePath, outputDir, shardSize, isCSV)
//    shardWithLineNumbers(inputFilePath, outputDir, shardSize)
  }

  def shardWithLineNumbers(inputFilePath: String, outputDir: String, numShards: Int): Unit = {
    // Clean up any existing shards before creating new ones
    deleteExistingShards(outputDir)

    val source = Source.fromFile(inputFilePath)
    val lines = source.getLines().toArray
    val linesPerShard = Math.ceil(lines.length.toDouble / numShards).toInt

    var shardIndex = 0
    var currentShardWriter = createShardWriter(outputDir, shardIndex, isCSV = false)
    var lineCounter = 0

    lines.zipWithIndex.foreach { case (line, lineNumber) =>
      if (lineCounter >= linesPerShard) {
        // Close the current shard and start a new one
        currentShardWriter.close()
        shardIndex += 1
        currentShardWriter = createShardWriter(outputDir, shardIndex, isCSV = false)
        lineCounter = 0
      }

      // Write the line number and the line content as a tab-separated value
      currentShardWriter.println(s"$lineNumber\t$line")
      lineCounter += 1
    }

    // Close any open resources
    currentShardWriter.close()
    source.close()
  }

  def splitCSV(line: String): Array[String] = {
    val buffer = ListBuffer[String]()
    val current = new StringBuilder
    var inQuotes = false

    line.foreach {
      case '"' => inQuotes = !inQuotes // Toggle quotes
      case ',' if !inQuotes =>
        buffer += current.toString().trim // Add value to buffer
        current.clear() // Clear for the next value
      case char => current.append(char)
    }

    buffer += current.toString().trim // Add the last value
    buffer.toArray
  }

}
