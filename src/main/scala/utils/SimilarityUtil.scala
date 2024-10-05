package utils

object SimilarityUtil {

  /**
   * Computes the cosine similarity between two vectors.
   * Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space.
   *
   * @param vec1 The first vector, represented as an array of doubles.
   * @param vec2 The second vector, represented as an array of doubles.
   * @return A value between -1 and 1 representing the cosine similarity. 1 means the vectors are identical in direction,
   *         0 means they are orthogonal, and -1 means they are opposite in direction.
   */
  def cosineSimilarity(vec1: Array[Double], vec2: Array[Double]): Double = {
    val dotProduct = vec1.zip(vec2).map { case (a, b) => a * b }.sum
    val magnitude1 = math.sqrt(vec1.map(x => x * x).sum)
    val magnitude2 = math.sqrt(vec2.map(x => x * x).sum)
    if (magnitude1 == 0.0 || magnitude2 == 0.0) 0.0
    else dotProduct / (magnitude1 * magnitude2)
  }

}
