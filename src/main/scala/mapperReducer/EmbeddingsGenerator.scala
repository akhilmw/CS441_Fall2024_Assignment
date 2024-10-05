package mapperReducer

import utils.{EmbeddingGenerator, ShardingUtil, BytePairUtils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import com.typesafe.config.ConfigFactory

import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters._


object EmbeddingsGenerator {

  private val config = ConfigFactory.load()
  private val reducersNum = config.getInt("embedding-generator.reducer-count")
  private val windowSize = config.getInt("embedding-generator.window-size")
  private val stride = config.getInt("embedding-generator.stride-size")

  private val logger = LoggerFactory.getLogger(this.getClass)
  private val shardingUtil = new ShardingUtil()

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      logger.error("Usage: EmbeddingTrainingJob <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration()
    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))

    logger.info(s"Starting job with input: $inputPath and output: $outputPath")
    runJob(conf, inputPath, outputPath)
  }

  def runJob(conf: Configuration, inputPath: Path, outputPath: Path): Unit = {
    val job = Job.getInstance(conf, "Embedding Training Job")
    job.setJarByClass(this.getClass)

    job.setMapperClass(classOf[EmbeddingMapper])
    job.setReducerClass(classOf[EmbeddingReducer])

    // Since mapper outputs Text and Text
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[Text])

    // The reducer's final output is also Text and Text
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    job.setNumReduceTasks(reducersNum)

    FileInputFormat.addInputPath(job, inputPath)
    FileOutputFormat.setOutputPath(job, outputPath)

    if (job.waitForCompletion(true)) {
      logger.info("Embedding Training Job completed successfully.")
    } else {
      logger.error("Embedding Training Job failed.")
    }
  }


  class EmbeddingMapper extends Mapper[LongWritable, Text, Text, Text] {

    private val logger = LoggerFactory.getLogger(this.getClass)
    // Accumulate tokens from the shard
    private val collectedTokens = ListBuffer[Int]()

    // The map method now only collects tokens
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      logger.debug(s"Processing line: ${value.toString}")
      // Skip the header lines
      if (value.toString.contains("Word,EncodedTokens,Frequency")) {
        return
      }

      // Read and parse the CSV line
      val line = value.toString.trim
      val columns = shardingUtil.splitCSV(line)

      // Extract the EncodedTokens column, expected to be in the second column
      if (columns.length > 1) {
        val encodedTokensStr = columns(1).trim

        // Remove brackets and split by space to extract individual token IDs
        val tokens = encodedTokensStr.stripPrefix("[").stripSuffix("]").split("\\s+").map(_.toInt)
        collectedTokens ++= tokens // Collect tokens for embedding generation
        logger.debug(s"Collected tokens: ${tokens.mkString("Array(", ", ", ")")}")
      }
    }

    // The cleanup method is called once at the end of processing the shard
    override def cleanup(context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      if (collectedTokens.nonEmpty) {
        logger.info(s"Generating embeddings for ${collectedTokens.size} tokens.")
        // Generate embeddings using the EmbeddingGenerator for the entire shard's tokens
        val embeddings: Map[Int, INDArray] = EmbeddingGenerator.generateEmbeddingsForTokens(collectedTokens.toSeq, windowSize, stride)

        // Emit each token and its corresponding embedding
        embeddings.foreach { case (token, embeddingVector) =>
          val embeddingStr = embeddingVector.toDoubleVector.mkString(",")
          context.write(new Text(token.toString), new Text(embeddingStr))
          logger.debug(s"Emitted embedding for token: $token")
        }
      }
    }
  }


  class EmbeddingReducer extends Reducer[Text, Text, Text, Text] {

    private val logger = LoggerFactory.getLogger(this.getClass)

    override def setup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      logger.info("Reducer setup initialized.")
      // Write the CSV header to the context once at the beginning
      val header = "TokenID,Word,Embeddings"
      context.write(null, new Text(header))
    }

    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      logger.debug(s"Reducing for key: ${key.toString}")
      val embeddingVectors = values.asScala.map { value =>
        val vectorArray = value.toString.split(",").map(_.toDouble)
        Nd4j.create(vectorArray)
      }.toList

      // Compute the average of these vectors
      val sumVector = embeddingVectors.reduce(_ add _)
      val averageVector = sumVector.div(embeddingVectors.size)

      // Convert the average vector to a string representation
      val tokenID = key.toString.toInt
      val tokenWord = BytePairUtils.decode(Seq(tokenID))
      val embeddingStr = averageVector.toDoubleVector.mkString(",")

      val csvRow = s"$tokenID,$tokenWord,$embeddingStr"

      // Write the CSV row to the context
      context.write(null, new Text(csvRow))
      logger.debug(s"Emitted CSV row: $csvRow")
    }
  }


}

