package mapperReducer

import utils.BytePairUtils
import utils.EmbeddingGenerator

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import scala.jdk.CollectionConverters._



object WordCountBPEJob {


  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: WordCountBPEJob <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration()
    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))

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

    if (job.waitForCompletion(true)) {
      println("Job completed successfully.")
    } else {
      println("Job failed.")
    }
  }

  // Mapper class
  class TokenizerMapper extends Mapper[LongWritable, Text, Text, IntWritable] {
    val one = new IntWritable(1)
    val wordText = new Text()

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit = {
      // Tokenize the line into words
      val line = value.toString.toLowerCase
      val words = line.split("\\W+").filter(_.nonEmpty)

      words.foreach { word =>
        wordText.set(word)
        context.write(wordText, one)
      }
    }
  }

  // Reducer class
  class BPEReducer extends Reducer[Text, IntWritable, Text, Text] {

    private val collectedTokens = scala.collection.mutable.ListBuffer[Int]()
    private val embeddingOutputFile = "/Users/akhilnair/Desktop/CS441_Fall2024_Assignment/EmbeddingsOutput/embeddings.csv" // Path where embeddings will be saved


    override def reduce(key: Text, values: java.lang.Iterable[IntWritable], context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
      // Sum the counts
      // Sum the counts
      val sum = values.asScala.foldLeft(0)(_ + _.get())

      // Apply Byte Pair Encoding
      val word = key.toString
      val tokens = BytePairUtils.encodeText(word)

      // Collect tokens for embedding generation
      collectedTokens ++= tokens

      // Prepare output value
      val tokensStr = tokens.mkString("[", " ", "]")
      val outputValue = s"$tokensStr,$sum"

      context.write(key, new Text(outputValue))
    }

    override def cleanup(context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
      // Train the embedding model using the collected tokens
      if (collectedTokens.nonEmpty) {
        println("Writing collected tokens to HDFS for the next MapReduce job...")
        val tokensOutputPath = new Path("/Users/akhilnair/Desktop/CS441_Fall2024_Assignment/TokensOutput/tokens.txt")
        val fs = tokensOutputPath.getFileSystem(context.getConfiguration)
        val outputStream = fs.create(tokensOutputPath, true)

        collectedTokens.distinct.foreach(token => {
          outputStream.writeBytes(token.toString + "\n")
        })

        outputStream.close()
//        println("Training embeddings using the collected tokens...")
//        val uniqueTokens = collectedTokens.distinct.toSeq
//        EmbeddingGenerator.trainAndSaveEmbeddings(uniqueTokens, windowSize = 3, stride = 1, outputFileName = embeddingOutputFile)
//        println(s"Embeddings saved to $embeddingOutputFile")
      }
    }
  }



}
