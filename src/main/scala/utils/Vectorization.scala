package utils

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.buffer.DataType

object Vectorization {

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
