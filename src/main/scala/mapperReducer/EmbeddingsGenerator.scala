package mapperReducer

import utils.EmbeddingGenerator
import utils.BytePairUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters._




object EmbeddingsGenerator {

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: EmbeddingTrainingJob <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration()
    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))

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

    FileInputFormat.addInputPath(job, inputPath)
    FileOutputFormat.setOutputPath(job, outputPath)

    if (job.waitForCompletion(true)) {
      println("Embedding Training Job completed successfully.")
    } else {
      println("Embedding Training Job failed.")
    }
  }


  class EmbeddingMapper extends Mapper[LongWritable, Text, Text, Text] {
    // Accumulate tokens from the shard
    private val collectedTokens = ListBuffer[Int]()

    // The map method now only collects tokens
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val line = value.toString.trim
      val tokens = line.split("\\s+").map(_.toInt) // Convert tokens to integers
      collectedTokens ++= tokens // Collect all tokens from this shard
    }

    // The cleanup method is called once at the end of processing the shard
    override def cleanup(context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      if (collectedTokens.nonEmpty) {
        // Generate embeddings using the EmbeddingGenerator for the entire shard's tokens
        val embeddings: Map[Int, INDArray] = EmbeddingGenerator.generateEmbeddingsForTokens(collectedTokens.toSeq, windowSize = 3, stride = 1)

        // Emit each token and its corresponding embedding
        embeddings.foreach { case (token, embeddingVector) =>
          val embeddingStr = embeddingVector.toDoubleVector.mkString(",")
          context.write(new Text(token.toString), new Text(embeddingStr))
        }
      }
    }
  }



  class EmbeddingReducer extends Reducer[Text, Text, Text, Text] {

    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
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

      // Write the final averaged embedding
      EmbeddingGenerator.saveEmbeddingToCSV(tokenID, tokenWord, embeddingStr)

      context.write(key, new Text(averageVector.toDoubleVector.mkString(",")))
    }
  }

}

