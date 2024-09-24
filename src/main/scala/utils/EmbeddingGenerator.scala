package utils

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.buffer.DataType
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

import java.io.PrintWriter


object EmbeddingGenerator {

    def trainAndSaveEmbeddings(decodedTokens: Seq[Int], windowSize: Int, stride: Int, outputFileName: String): Unit = {
    val (remappedDecoded, tokenToIndex) = remapTokens(decodedTokens)
    val inputOutputPairs = BytePairUtils.createInputOutputPairs(remappedDecoded, windowSize, stride)
    val (inputFeatures, outputLabels) = convertToIndArrays(inputOutputPairs)

    val vocabSize = tokenToIndex.size
    val embeddingDim = 50

    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new EmbeddingSequenceLayer.Builder().nIn(vocabSize).nOut(embeddingDim).build())
      .inputPreProcessor(1, new CustomReshapePreprocessor(windowSize, embeddingDim))
      .layer(1, new OutputLayer.Builder(LossFunction.SPARSE_MCXENT).nIn(embeddingDim * windowSize).nOut(vocabSize).activation(Activation.SOFTMAX).build())
      .build()

    val model = new MultiLayerNetwork(config)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    val numEpochs = 100
    for (epoch <- 1 to numEpochs) {
      model.fit(inputFeatures, outputLabels)
      println(s"Completed epoch $epoch")
    }

    val embeddings: INDArray = model.getLayer(0).getParam("W")
    saveEmbeddings(outputFileName, embeddings, tokenToIndex)
    println(s"Embeddings saved to $outputFileName")
  }

  def remapTokens(decodedTokens: Seq[Int]): (Seq[Int], Map[Int, Int]) = {
    val uniqueTokens = decodedTokens.distinct.sorted
    val tokenToIndex = uniqueTokens.zipWithIndex.toMap
    val remappedTokens = decodedTokens.map(tokenToIndex)
    (remappedTokens, tokenToIndex)
  }

  def saveEmbeddings(filename: String, embeddings: INDArray, tokenToIndex: Map[Int, Int]): Unit = {
    val pw = new PrintWriter(filename)
    try {
      val indexToToken = tokenToIndex.map(_.swap)
      val indexToWord = indexToToken.map { case (index, originalToken) =>
        val word = BytePairUtils.decode(Seq(originalToken))
        (index, word)
      }
      pw.println("Index,Original Token,Word,Embeddings")
      val rows = embeddings.rows()
      for (i <- 0 until rows) {
        val originalToken = indexToToken.getOrElse(i, -1)
        val word = indexToWord.getOrElse(i, "Unknown")
        val vector = embeddings.getRow(i).toDoubleVector.mkString(",")
        pw.println(s"$i,$originalToken,$word,$vector")
      }
    } finally {
      pw.close()
    }
  }

  def convertToIndArrays(inputOutputPairs : Seq[(Array[Int], Int)]): (INDArray, INDArray) = {
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
