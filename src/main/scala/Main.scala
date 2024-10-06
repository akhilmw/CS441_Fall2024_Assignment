
import utils.{ShardingUtil, OrderedShardingUtil}
import test.{LocalWordCountBPEJob, LocalSemanticRepresentationJob, LocalEmbeddingGeneratorTest}
import com.typesafe.config.ConfigFactory
import mapperReducer.OrderedWordME


object Main {

  private val config = ConfigFactory.load()
  private val datasetShardSize = config.getInt("main.dataset-shard-size")
  private val tokensShardSize = config.getInt("main.tokens-shard-size")
  private val embeddingsShardSize = config.getInt("main.embeddings-shard-size")
  private val shardPath: String = config.getString("filePaths.shardPath")
  private val wordCountPath: String = config.getString("filePaths.wordCountPath")
  private val datasetPath: String = config.getString("filePaths.datasetPath")
  private val orderedShardPath: String = config.getString("filePaths.orderedShardPath")
  private val orderedWordCountTokensPath: String = config.getString("filePaths.orderedWordCountTokensPath")
  private val ordered_tokens_shards: String = config.getString("filePaths.ordered_tokens_shards")
  private val embeddings_output: String = config.getString("filePaths.embeddings_output")
  val embeddingShardsPath: String = config.getString("filePaths.embeddingsCsvPath")
  val semanticsOutputPath: String = config.getString("filePaths.semantics_output")
  private val lines : Int = config.getInt("embedding-generator.lines")

  def main(args: Array[String]): Unit = {

    // Step 1: Create shards of the dataset.
    // - `isDataset = true` indicates that we are sharding the original dataset (e.g., text or CSV).
    // - `isTokens = false` indicates that the data being processed is not tokenized yet.
    // - `datasetShardSize` controls the size of each shard.
    val shardingUtil = new ShardingUtil()
    shardingUtil.shardTextOrCSV(isDataset = true, isTokens = false, datasetShardSize)

    // Step 2: Run the first mapper reducer (WordCountBPEJob).
    // - This step runs the WordCountBPEJob to count the words and generate tokens.
    // - `shardPath` is the path to the dataset shards created in Step 1.
    // - `wordCountPath` is the output directory where the word count and token output will be saved.
    LocalWordCountBPEJob.main(Array(shardPath, wordCountPath))

    // Step 3: Create ordered shards of the tokenized dataset.
    // - `datasetPath` is the path to the original dataset.
    // - `orderedShardPath` is the path where the ordered tokenized shards will be stored.
    // - `lines` is the number of lines to be included in each shard.
    OrderedShardingUtil.shardTextFile(datasetPath, orderedShardPath, lines)

    // Step 4: Run the second mapper reducer (OrderedWordME).
    // - This step maintains the order of word frequency output.
    // - `orderedShardPath` is the path to the ordered shards created in Step 3.
    // - `orderedWordCountTokensPath` is the output path where the word count and tokens will be stored.
    OrderedWordME.main(Array(orderedShardPath, orderedWordCountTokensPath))

    // Step 5: Shard the tokens for embedding generation.
    // - `isDataset = false` indicates that we are now sharding tokens, not the original dataset.
    // - `isTokens = true` indicates that the input data consists of tokenized data.
    // - `tokensShardSize` controls the size of each token shard.
    shardingUtil.shardTextOrCSV(isDataset = false, isTokens = true, tokensShardSize)

    // Step 6: Run the embedding generator.
    // - This step runs the embedding generation process using the sharded tokens.
    // - `ordered_tokens_shards` is the path to the token shards created in Step 5.
    // - `embeddings_output` is the output path where the embeddings will be saved.
    LocalEmbeddingGeneratorTest.main(Array(ordered_tokens_shards, embeddings_output))

    // Step 7: Shard the embeddings for semantic representation calculation.
    // - `isDataset = false` indicates that we are sharding embeddings, not the original dataset.
    // - `isTokens = false` indicates that we are not processing tokens here but rather embeddings.
    // - `embeddingsShardSize` controls the size of each embedding shard.
    shardingUtil.shardTextOrCSV(isDataset = false, isTokens = false, embeddingsShardSize)

    // Step 8: Run the semantic representation job.
    // - This step calculates semantic similarities using the embeddings generated in Step 6.
    // - `embeddingShardsPath` is the path to the sharded embeddings created in Step 7.
    // - `semanticsOutputPath` is the output path where the semantic representation output will be saved.
    LocalSemanticRepresentationJob.main(Array(embeddingShardsPath, semanticsOutputPath))
  }




}