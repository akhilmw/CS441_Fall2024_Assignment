package utils

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.buffer.DataType
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import com.typesafe.config.ConfigFactory


object EmbeddingGenerator {

  // Load properties from the configuration file
  private val config = ConfigFactory.load()
  private val numEpochs = config.getInt("embedding-generator.num-epochs")
  private val embeddingDim = config.getInt("embedding-generator.embedding-dim")


  /**
   * Generate embeddings for the provided tokens.
   *
   * @param encodedTokens Sequence of tokens (integers) to generate embeddings for.
   * @param windowSize    Size of the sliding window used to create input-output pairs.
   * @param stride        Number of tokens to shift the sliding window each time.
   * @return A map of token IDs to their corresponding embedding vectors (INDArray).
   */

  def generateEmbeddingsForTokens(encodedTokens: Seq[Int], windowSize: Int, stride: Int): Map[Int, INDArray] = {
    // Remap tokens to create unique token IDs
    val (remappedDecoded, tokenToIndex) = remapTokens(encodedTokens)
    // Create input-output pairs from the sequence of tokens
    val inputOutputPairs = BytePairUtils.createInputOutputPairs(remappedDecoded, windowSize, stride)
    // Convert the input-output pairs into INDArrays (used by Deeplearning4j)
    val (inputFeatures, outputLabels) = convertToIndArrays(inputOutputPairs)

    val vocabSize = tokenToIndex.size

    // Define the neural network configuration
    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new EmbeddingSequenceLayer.Builder().nIn(vocabSize).nOut(embeddingDim).build())
      .inputPreProcessor(1, new CustomReshapePreprocessor(windowSize, embeddingDim))
      .layer(1, new OutputLayer.Builder(LossFunction.SPARSE_MCXENT).nIn(embeddingDim * windowSize).nOut(vocabSize).activation(Activation.SOFTMAX).build())
      .build()

    // Initialize the neural network model
    val model = new MultiLayerNetwork(config)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    // Train the model for the specified number of epochs
    (1 to numEpochs).foreach { _ =>
      model.fit(inputFeatures, outputLabels)
    }

    // Extract the learned embeddings from the first layer (EmbeddingSequenceLayer)
    val embeddings: INDArray = model.getLayer(0).getParam("W")
    // Reverse the map to get index -> token
    val indexToToken = tokenToIndex.map(_.swap)
    val embeddingsMap: Map[Int, INDArray] = (0 until embeddings.rows()).map { rowIndex =>
      val tokenId = indexToToken(rowIndex)
      tokenId -> embeddings.getRow(rowIndex).dup()
    }.toMap

    // Return the map containing each token ID mapped to its embedding vector
    embeddingsMap
  }

  /**
   * Remap tokens to unique indices.
   *
   * @param encodedTokens Sequence of original tokens (integers).
   * @return A tuple containing the remapped tokens and a map from original token to its new index.
   */
  private def remapTokens(encodedTokens: Seq[Int]): (Seq[Int], Map[Int, Int]) = {
    val uniqueTokens = encodedTokens.distinct.sorted
    val tokenToIndex = uniqueTokens.zipWithIndex.toMap
    val remappedTokens = encodedTokens.map(tokenToIndex)
    (remappedTokens, tokenToIndex)
  }


  /**
   * Convert input-output token pairs to INDArrays.
   *
   * @param inputOutputPairs Sequence of (input sequence, target token) pairs.
   * @return A tuple containing the input features and output labels as INDArrays.
   */
  private def convertToIndArrays(inputOutputPairs: Seq[(Array[Int], Int)]): (INDArray, INDArray) = {
    val inputSequences: Array[Array[Double]] = inputOutputPairs.map { case (inputArray, _) =>
      inputArray.map(_.toDouble)
    }.toArray

    // Convert target tokens to Array[Double]
    val targetTokens: Array[Double] = inputOutputPairs.map { case (_, target) =>
      target.toDouble
    }.toArray

    // Create INDArrays from the arrays
    val inputFeatures: INDArray = Nd4j.create(inputSequences).castTo(DataType.INT32)
    val outputLabels: INDArray = Nd4j.create(targetTokens).reshape(-1, 1).castTo(DataType.INT32)

    (inputFeatures, outputLabels)

  }


}
