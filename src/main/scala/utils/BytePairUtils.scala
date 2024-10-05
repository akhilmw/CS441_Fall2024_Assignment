package utils

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, ModelType, IntArrayList}

object BytePairUtils {

  // Create a new encoding registry
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()

  // Get the encoder for a specific model (here I've used GPT-4)
  private val encoder: Encoding = registry.getEncodingForModel(ModelType.GPT_4)

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

  def createInputOutputPairs(tokens: Seq[Int], windowSize: Int, stride: Int): Seq[(Array[Int], Int)] = {
    // Slide over the tokens sequence with a window of size `windowSize + 1` (to include the target token)
    // and a step (stride) determining how much to move the window at each step
    tokens.sliding(windowSize + 1, stride).map { window =>
      // Take the first `windowSize` tokens as input
      val inputSeq = window.take(windowSize).toArray
      val targetToken = window.last
      // Return the pair (input sequence, target token)
      (inputSeq, targetToken)
    }.toSeq
  }


}
