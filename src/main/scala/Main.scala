
import utils.ShardingUtil
import test.{LocalWordCountBPEJob, LocalSemanticRepresentationJob, LocalEmbeddingGeneratorTest}
import com.typesafe.config.ConfigFactory


object Main {

  private val config = ConfigFactory.load()
  private val datasetShardSize = config.getInt("main.dataset-shard-size")
  private val tokensShardSize = config.getInt("main.tokens-shard-size")
  private val embeddingsShardSize = config.getInt("main.embeddings-shard-size")

  def main(args: Array[String]): Unit = {

    // Step1 : Create shards of the dataset
    val shardingUtil = new ShardingUtil()
    shardingUtil.shardTextOrCSV(isDataset = false, isTokens = false, embeddingsShardSize)

    // Step 2: Run the first mapper reducer (WordCountBPEJob)
//    LocalWordCountBPEJob.main(Array.empty)

    // Step 3: Shard the tokens output
//    shardingUtil.shardTextOrCSV(isDataset = true, isTokens = true)
//
//    // Step 4: Run the second MapReduce job (EmbeddingsGenerator)
//    LocalEmbeddingGeneratorTest.main(Array.empty)
//
//    // Step 5: Shard the embeddings.csv
//    shardingUtil.shardTextOrCSV(isDataset = false, isTokens = false)
//
//    // Step 6: Run the third MapReduce job (SemanticRepresentation)
//    println("Running the SemanticRepresentation job...")
//    LocalSemanticRepresentationJob.main(Array.empty)


  }



}