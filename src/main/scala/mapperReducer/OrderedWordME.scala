package mapperReducer

import utils.BytePairUtils
import org.apache.hadoop.conf.{Configuration, Configurable}
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Partitioner, Reducer}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.slf4j.{Logger, LoggerFactory}
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._
import scala.io.Source

object OrderedWordME {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass)
  private val config = ConfigFactory.load()
  private val reducersNum = config.getInt("ordered-word-mr.reducer-count")


  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      logger.error("Usage: OrderedWordME <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))
    val outputFilePath = new Path(outputPath, "part-r-00000") // This is the generated output file
    val renamedFilePath = new Path(outputPath, "output.csv")   // This is the target CSV file name

    // Preprocess to find the maximum position
    val maxPosition = getMaxPosition(fs, inputPath)
    conf.setLong("max.position", maxPosition)
    conf.setInt("mapreduce.job.reduces", reducersNum)

    runJob(conf, inputPath, outputPath)

    if (fs.exists(outputFilePath)) {
      val success = fs.rename(outputFilePath, renamedFilePath)
      if (success) {
        println(s"Renamed part-r-00000 to output.csv successfully.")
      } else {
        println(s"Failed to rename part-r-00000 to output.csv.")
      }
    } else {
      println(s"Output file part-r-00000 does not exist.")
    }

  }

  def runJob(conf: Configuration, inputPath: Path, outputPath: Path): Unit = {
    val job = Job.getInstance(conf, "Word Tokenization with Positions")
    job.setJarByClass(this.getClass)

    // Set Mapper and Reducer classes
    job.setMapperClass(classOf[TokenizerMapper])
    job.setReducerClass(classOf[TokenizerReducer])
    job.setPartitionerClass(classOf[PositionPartitioner])

    // Set output key and value types
    job.setMapOutputKeyClass(classOf[LongWritable])
    job.setMapOutputValueClass(classOf[Text])
    job.setOutputKeyClass(classOf[NullWritable])
    job.setOutputValueClass(classOf[Text])

    // Set input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[NullWritable, Text]])

    // Set input and output paths
    FileInputFormat.addInputPath(job, inputPath)
    FileOutputFormat.setOutputPath(job, outputPath)

    job.setNumReduceTasks(reducersNum)

    val success = job.waitForCompletion(true)
    if (success) {
      logger.info("Job completed successfully.")
    } else {
      logger.error("Job failed.")
      System.exit(1)
    }
  }

  def getMaxPosition(fs: FileSystem, inputPath: Path): Long = {
    var maxPosition = 0L
    val statusList = fs.listStatus(inputPath)
    val files = getAllFiles(statusList)

    for (file <- files) {
      val stream = fs.open(file)
      val reader = Source.fromInputStream(stream)
      for (line <- reader.getLines()) {
        val splitIndex = line.indexOf('_')
        if (splitIndex > 0 && splitIndex < line.length - 1) {
          val positionStr = line.substring(0, splitIndex)
          try {
            val position = positionStr.toLong
            if (position > maxPosition) {
              maxPosition = position
            }
          } catch {
            case _: NumberFormatException =>
          }
        }
      }
      reader.close()
    }
    maxPosition
  }

  def getAllFiles(statusList: Array[FileStatus]): Array[Path] = {
    statusList.flatMap { status =>
      if (status.isDirectory) {
        val fs = status.getPath.getFileSystem(new Configuration())
        getAllFiles(fs.listStatus(status.getPath))
      } else {
        Array(status.getPath)
      }
    }
  }

  // Mapper Class - Uses word as key and tokenized version of the word as the value
  class TokenizerMapper extends Mapper[LongWritable, Text, LongWritable, Text] {
    private val logger: Logger = LoggerFactory.getLogger(this.getClass)

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, LongWritable, Text]#Context): Unit = {
      val line = value.toString.trim
      val splitIndex = line.indexOf('_')
      if (splitIndex > 0 && splitIndex < line.length - 1) {
        val positionStr = line.substring(0, splitIndex)
        val word = line.substring(splitIndex + 1)
        try {
          val position = positionStr.toLong
          context.write(new LongWritable(position), new Text(word))
        } catch {
          case _: NumberFormatException =>
            logger.warn(s"Invalid position format: $positionStr")
        }
      }
    }
  }

  // Reducer Class - Tokenizes the words and outputs them with their original positions
  // Reducer Class - Tokenizes the words and outputs them with their original positions
  class TokenizerReducer extends Reducer[LongWritable, Text, NullWritable, Text] {
    private val logger: Logger = LoggerFactory.getLogger(this.getClass)

    private var headerWritten = false // Ensure the CSV header is written only once

    override def reduce(key: LongWritable, values: java.lang.Iterable[Text], context: Reducer[LongWritable, Text, NullWritable, Text]#Context): Unit = {
      val position = key.get()

      if (!headerWritten) {
        // Write the CSV header once at the beginning
        context.write(NullWritable.get(), new Text("word_pos,Word,EncodedTokens"))
        headerWritten = true
      }

      values.asScala.foreach { value =>
        var word = value.toString.trim

        // Clean up the word: Remove non-alphabetic characters (like commas, punctuation, etc.)
        word = word.replaceAll("[^a-zA-Z]", "")

        if (word.nonEmpty) {
          val tokenArray = BytePairUtils.encodeText(word).toArray
          if (tokenArray.nonEmpty) {
            val tokenIds = tokenArray.mkString("[", " ", "]")
            val outputLine = s"$position,$word,$tokenIds"
            context.write(NullWritable.get(), new Text(outputLine))
            logger.info(s"Processed word: $word with tokens: $tokenIds")
          } else {
            logger.warn(s"Skipping empty token array for word: $word")
          }
        } else {
          logger.warn(s"Skipping invalid word at position: $position")
        }
      }
    }
  }


  // Custom Partitioner Class - Ensures all words go to the same partition for ordering
  class PositionPartitioner extends Partitioner[LongWritable, Text] with Configurable {
    private var conf: Configuration = _
    private var positionsPerReducer: Long = _
    private var numReducers: Int = _
    private var maxPosition: Long = _

    override def setConf(conf: Configuration): Unit = {
      this.conf = conf
      numReducers = conf.getInt("mapreduce.job.reduces", 1)
      maxPosition = conf.getLong("max.position", Long.MaxValue)
      positionsPerReducer = (maxPosition.toDouble / numReducers).ceil.toLong
      logger.info(s"Max Position: $maxPosition, Num Reducers: $numReducers, Positions per reducer: $positionsPerReducer")
    }

    override def getConf: Configuration = conf

    override def getPartition(key: LongWritable, value: Text, numPartitions: Int): Int = {
      val position = key.get()
      val partition = (position / positionsPerReducer).toInt
      logger.info(s"Key: $position goes to partition: $partition (out of $numPartitions)")

      if (partition >= numPartitions) numPartitions - 1 else partition
    }
  }

}
