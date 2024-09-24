import org.deeplearning4j.nn.conf.{InputPreProcessor, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import utils.{BytePairUtils, CustomReshapePreprocessor}
import utils.EmbeddingGenerator.convertToIndArrays

import java.io.PrintWriter
import scala.collection.mutable
import java.io.PrintWriter


object Main {

  def main(args: Array[String]): Unit = {
    // Your input text
    val text = "This is a sample text message"

    val decoded = BytePairUtils.encodeText(text)

    val windowSize = 3 // Adjust as needed
    val stride = 2 // Move the window by 2 tokens each time

    def remapTokens(decodedTokens: Seq[Int]): (Seq[Int], Map[Int, Int]) = {
      // Create a mapping from original tokens to a new range [0, vocabSize-1]
      val uniqueTokens = decodedTokens.distinct
      val tokenToIndex = uniqueTokens.zipWithIndex.toMap

      // Map the original tokens to the new indices
      val remappedTokens = decodedTokens.map(tokenToIndex)

      (remappedTokens, tokenToIndex)
    }

    val (remappedDecoded, tokenToIndex) = remapTokens(decoded)

    val inputOutputPairs = BytePairUtils.createInputOutputPairs(remappedDecoded, windowSize, stride)

    val convertedND4jPairs = convertToIndArrays(inputOutputPairs)
    val inputFeatures = convertedND4jPairs._1
    val outputLabels = convertedND4jPairs._2

    print("tokensss", decoded)


    print("remappedDecoded", remappedDecoded)

    // Get the vocab size based on the remapped tokens
    val vocabSize = tokenToIndex.size
    println("vocabSize", vocabSize)

    val embeddingDim = 50
    // Build the network configuration
    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .list()
      // EmbeddingSequenceLayer for sequence input
      .layer(0, new EmbeddingSequenceLayer.Builder()
        .nIn(vocabSize)
        .nOut(embeddingDim)
        .build())
      // Use the custom preprocessor to reshape the data
      .inputPreProcessor(1, new CustomReshapePreprocessor(windowSize, embeddingDim))
      .layer(1, new OutputLayer.Builder(LossFunction.SPARSE_MCXENT)
        .nIn(embeddingDim * windowSize)
        .nOut(vocabSize)
        .activation(Activation.SOFTMAX)
        .build())
      .build()

    val model = new MultiLayerNetwork(config)
    model.init()
    model.setListeners(new ScoreIterationListener(100)) // Print score every 100 iterations


    val numEpochs = 100
    for (epoch <- 1 to numEpochs) {
      model.fit(inputFeatures, outputLabels)
      println(s"Completed epoch $epoch")
    }

    val embeddings: INDArray = model.getLayer(0).getParam("W")
    saveEmbeddings("embeddings.csv", embeddings, tokenToIndex)
    println("Embeddings saved to embeddings.csv")

    def saveEmbeddings(filename: String, embeddings: INDArray, tokenToIndex: Map[Int, Int]): Unit = {
      val pw = new PrintWriter(filename)
      try {
        // Create a reverse mapping from remapped index back to original token
        val indexToToken = tokenToIndex.map(_.swap)

        // Get the actual words from the original text
        val indexToWord = indexToToken.map { case (index, originalToken) =>
          val word = BytePairUtils.decode(Seq(originalToken)) // Decode the original token back to the word
          (index, word)
        }

        // Write the header
        pw.println("Index,Original Token,Word,Embeddings")

        val rows = embeddings.rows()
        for (i <- 0 until rows) {
          // Get the original token and word for this embedding index
          val originalToken = indexToToken.getOrElse(i, -1)
          val word = indexToWord.getOrElse(i, "Unknown")

          // Get the embedding vector
          val vector = embeddings.getRow(i).toDoubleVector
          val vectorStr = vector.mkString(",")

          // Write the row to CSV
          pw.println(s"$i,$originalToken,$word,$vectorStr")
        }
      } finally {
        pw.close()
      }
    }

    print("vocab sie : ", vocabSize)

    //     Step 1: Tokenize text into words and count frequencies
    val words = tokenizeTextToWords(text)
    val wordFrequencies = countWordFrequencies(words)

    // Step 2: Encode each unique word using BytePairUtils
    val uniqueWords = wordFrequencies.keySet
    val wordToTokens = encodeWords(uniqueWords)

    // Step 3: Create a data structure to hold the data
    val wordDataList = uniqueWords.map { word =>
      val tokens = wordToTokens(word)
      val frequency = wordFrequencies(word)
      WordData(word, tokens, frequency)
    }.toList.sortBy(-_.frequency) // Sort by frequency descending

    // Step 4: Write the data to a CSV file
    writeDataToCSV("word_token_frequency.csv", wordDataList)

    println("Data has been written to word_token_frequency.csv")
  }

  // Function to tokenize text into words
  def tokenizeTextToWords(text: String): Seq[String] = {
    text.toLowerCase.split("\\W+").filter(_.nonEmpty)
  }

  // Function to count word frequencies
  def countWordFrequencies(words: Seq[String]): Map[String, Int] = {
    words.groupBy(identity).view.mapValues(_.size).toMap
  }

  // Function to encode words to tokens
  def encodeWords(words: Set[String]): Map[String, Seq[Int]] = {
    words.map { word =>
      val tokens = BytePairUtils.encodeText(word)
      word -> tokens
    }.toMap
  }


  // Case class to hold word data
  case class WordData(word: String, tokens: Seq[Int], frequency: Int)

  //     Function to write data to a CSV file
  def writeDataToCSV(filename: String, data: List[WordData]): Unit = {
    val pw = new PrintWriter(filename)
    try {
      // Write header
      pw.println("word,tokens,frequency")
      // Write data
      data.foreach { wd =>
        val tokensStr = wd.tokens.mkString("[", " ", "]")
        pw.println(s"${wd.word},\"$tokensStr\",${wd.frequency}")
      }
    } finally {
      pw.close()
    }
  }


}