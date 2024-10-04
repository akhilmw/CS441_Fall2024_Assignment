package mapperReducer

import utils.SimilarityUtil

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import scala.jdk.CollectionConverters._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import scala.collection.mutable.ArrayBuffer
import org.slf4j.LoggerFactory

object SemanticRepresentation {

  private val logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      logger.error("Usage: FindSimilarWordsJob <input embeddings path> <output path> <topN>")
      System.exit(-1)
    }

    val conf = new Configuration()
    conf.setInt("topN", args(2).toInt)
    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))

    runJob(conf, inputPath, outputPath)
  }

  def runJob(conf: Configuration, inputPath: Path, outputPath: Path): Unit = {
    val job = Job.getInstance(conf, "Find Similar Words")
    job.setJarByClass(this.getClass)

    job.setMapperClass(classOf[EmbeddingMapper])
    job.setReducerClass(classOf[SimilarityReducer])

    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[Text])

    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job, inputPath)
    FileOutputFormat.setOutputPath(job, outputPath)

    if (job.waitForCompletion(true)) {
      logger.info("Job completed successfully.")
    } else {
      logger.error("Job failed.")
    }
  }

  // Mapper class to read the embeddings
  class EmbeddingMapper extends Mapper[Object, Text, Text, Text] {

    private val logger = LoggerFactory.getLogger(this.getClass)
    // Track the first row in each mapper
    var isFirstLine = true

    override def map(key: Object, value: Text, context: Mapper[Object, Text, Text, Text]#Context): Unit = {
      val line = value.toString.trim

      // Skip the header line, checking within each mapper
      if (isFirstLine && line.startsWith("TokenID,Word,Embeddings")) {
        isFirstLine = false
        logger.debug("Skipping header line.")
        return
      }

      isFirstLine = false // Mark first line as processed

      // Splitting the CSV line while handling potential commas within quoted fields
      val parts = splitCSV(line)

      // Check if the line contains at least a word and one embedding value
      if (parts.length > 2) {
        val word = parts(1).trim // The third column is the word
        val embedding = parts.drop(2).mkString(",") // Join all embedding values into a single string
        context.write(new Text(word), new Text(embedding))
        logger.debug(s"Emitted word: $word with embedding.")
      }else {
        logger.warn(s"Invalid line format: $line")
      }
    }

    // Helper method to split CSV lines while handling quoted commas
    private def splitCSV(line: String): Array[String] = {
      val buffer = ArrayBuffer[String]()
      val current = new StringBuilder
      var inQuotes = false

      line.foreach {
        case '"' => inQuotes = !inQuotes // Toggle inQuotes flag
        case ',' if !inQuotes =>
          buffer += current.toString().trim // Add the value
          current.clear() // Clear for the next value
        case char => current.append(char)
      }

      buffer += current.toString().trim // Add the last value
      buffer.toArray
    }
  }

  // Reducer class to compute cosine similarities
  class SimilarityReducer extends Reducer[Text, Text, Text, Text] {

    private val logger = LoggerFactory.getLogger(this.getClass)
    // Store all word embeddings
    private val wordEmbeddings = scala.collection.mutable.Map[String, INDArray]()

    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      // Collect the current word's embeddings
      val embeddingArray = values.asScala.toSeq.flatMap(value => value.toString.split(",").map(_.toDouble))

      if (embeddingArray.nonEmpty) {
        val embeddingVector = Nd4j.create(embeddingArray.toArray)
        wordEmbeddings.put(key.toString, embeddingVector)
        logger.debug(s"Stored embedding for word: ${key.toString}")
      }else {
        logger.warn(s"Empty embedding array for word: ${key.toString}")
      }
    }

    // Cleanup method runs after all reduce calls are done
    override def cleanup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val topN = context.getConfiguration.getInt("topN", 5)

      // Now we have all word embeddings, perform pairwise similarity calculations
      wordEmbeddings.foreach { case (word1, vector1) =>
        // Compute cosine similarity with all other embeddings
        val similarities = wordEmbeddings.collect {
          case (word2, vector2) if word1 != word2 =>
            val similarity = SimilarityUtil.cosineSimilarity(vector1.toDoubleVector, vector2.toDoubleVector)
            (word2, similarity)
        }

        // Get the top N most similar words
        val topSimilarWords = similarities.toSeq.sortBy(-_._2).take(topN)

        val yamlLines = topSimilarWords.map { case (word, score) =>
          s"  - word: $word\n    similarity: $score"
        }.mkString("\n")

        // Write the YAML entry for each word
        val yamlEntry = s"$word1:\n$yamlLines"
        context.write(null, new Text(yamlEntry))
        logger.debug(s"Written YAML entry for word: $word1")
      }
    }
  }

}
