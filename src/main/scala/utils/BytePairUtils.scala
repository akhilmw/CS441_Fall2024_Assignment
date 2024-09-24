package utils

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, ModelType, IntArrayList}
import scala.jdk.CollectionConverters._

object BytePairUtils {

  // Create a new encoding registry
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()

  // Get the encoder for a specific model (e.g., GPT-4)
  val encoder: Encoding = registry.getEncodingForModel(ModelType.GPT_4)

  // Method to encode a string into tokens using BPE
  def encodeText(text: String): Seq[Int] = {
    val encoded: IntArrayList = encoder.encode(text)
    // Convert IntArrayList to Scala List[Int]
    val tokens = (0 until encoded.size()).map(i => encoded.get(i)).toList
    tokens
  }


  // Method to decode tokens back to the original string
  def decode(tokens: Seq[Int]): String = {
    // Convert Scala Seq[Int] to IntArrayList
    val intArrayList = new IntArrayList()
    tokens.foreach(intArrayList.add)
    // Decode using IntArrayList
    encoder.decode(intArrayList)
  }

  // Helper method to print tokens
  def printTokens(tokens: Seq[Int]): Unit = {
    println(s"Tokens: ${tokens.mkString(", ")}")
  }

  def createInputOutputPairs(tokens: Seq[Int], windowSize: Int, stride: Int): Seq[(Array[Int], Int)] = {
    tokens.sliding(windowSize + 1, stride).map { window =>
      val inputSeq = window.take(windowSize).toArray
      val targetToken = window.last
      (inputSeq, targetToken)
    }.toSeq
  }

  def getVocabSize(decodedTokens: Seq[Int]): Int = {
    decodedTokens.distinct.length
  }

}
