package mapperReducer

import utils.BytePairUtils

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import scala.jdk.CollectionConverters._
import org.slf4j.LoggerFactory
import com.typesafe.config.ConfigFactory



object WordCountBPEJob {

  private val config = ConfigFactory.load()
  private val reducersNum = config.getInt("word-count-bpe.reducer-count")

  // Initialize the logger
  private val logger = LoggerFactory.getLogger(this.getClass)


  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      logger.error("Usage: WordCountBPEJob <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration()
    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))

    logger.info(s"Running WordCountBPEJob with input path: $inputPath and output path: $outputPath")
    runJob(conf, inputPath, outputPath)
  }

  def runJob(conf: Configuration, inputPath: Path, outputPath: Path): Unit = {
    val job = Job.getInstance(conf, "Word Count with BPE")
    job.setJarByClass(this.getClass)

    job.setMapperClass(classOf[TokenizerMapper])
    job.setReducerClass(classOf[BPEReducer])

    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[IntWritable])

    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job, inputPath)
    FileOutputFormat.setOutputPath(job, outputPath)

    job.setNumReduceTasks(reducersNum)

    if (job.waitForCompletion(true)) {
      logger.info("Job completed successfully.")
    } else {
      logger.error("Job failed.")
    }
  }

  // Mapper class
  class TokenizerMapper extends Mapper[LongWritable, Text, Text, IntWritable] {
    val one = new IntWritable(1)
    val wordText = new Text()
    private val logger = LoggerFactory.getLogger(this.getClass)

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit = {
      // Tokenize the line into words
      val line = value.toString.toLowerCase
      logger.debug(s"Processing line: $line")
      val words = line.split("\\W+").filter(_.nonEmpty)


      words.foreach { word =>
        wordText.set(word)
        context.write(wordText, one)
        logger.debug(s"Emitting word: $word with count 1")
      }
    }
  }

  // Reducer class
  class BPEReducer extends Reducer[Text, IntWritable, Text, Text] {

    private val logger = LoggerFactory.getLogger(this.getClass)

    private val collectedTokens = scala.collection.mutable.ListBuffer[Int]()
    private var isHeaderWritten = false
    override def setup(context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
      // Check if this is the first call to the reducer and write the CSV header
      if (!isHeaderWritten) {
        val header = "Word,EncodedTokens,Frequency"
        context.write(new Text(header), null) // Writing header once
        isHeaderWritten = true
      }
    }

    override def reduce(key: Text, values: java.lang.Iterable[IntWritable], context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
      // Sum the counts
      val sum = values.asScala.foldLeft(0)(_ + _.get())
      logger.debug(s"Reducing word: ${key.toString} with count: $sum")

      // Apply Byte Pair Encoding
      val word = key.toString
      val tokens = BytePairUtils.encodeText(word)

      // Collect tokens for embedding generation
      collectedTokens ++= tokens

      // Prepare output value
      val tokensStr = tokens.mkString("[", " ", "]") // Represent the tokens in the desired CSV format
      val outputValue = s"$word,\"$tokensStr\",$sum" // CSV format for each row

      context.write(new Text(outputValue), null) // Write to context as CSV row

      logger.debug(s"Emitting reduced word: $word with encoded tokens: $tokensStr and count: $sum")
    }

  }



}
