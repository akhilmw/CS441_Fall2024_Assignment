import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import utils.SimilarityUtil

class SimilarityUtilSpec extends AnyFlatSpec with Matchers {

  "SimilarityUtil.cosineSimilarity" should "return 1.0 for identical vectors" in {
    val vec1 = Array(1.0, 2.0, 3.0)
    val vec2 = Array(1.0, 2.0, 3.0)
    SimilarityUtil.cosineSimilarity(vec1, vec2) shouldBe 1.0 +- 1e-9
  }

  it should "return -1.0 for opposite vectors" in {
    val vec1 = Array(1.0, 2.0, 3.0)
    val vec2 = Array(-1.0, -2.0, -3.0)
    SimilarityUtil.cosineSimilarity(vec1, vec2) shouldBe -1.0 +- 1e-9
  }

  it should "return 0.0 for orthogonal vectors" in {
    val vec1 = Array(1.0, 0.0, 0.0)
    val vec2 = Array(0.0, 1.0, 0.0)
    SimilarityUtil.cosineSimilarity(vec1, vec2) shouldBe 0.0 +- 1e-9
  }

  it should "handle zero vectors correctly" in {
    val zeroVec = Array(0.0, 0.0, 0.0)
    val nonZeroVec = Array(1.0, 2.0, 3.0)
    SimilarityUtil.cosineSimilarity(zeroVec, nonZeroVec) shouldBe 0.0
    SimilarityUtil.cosineSimilarity(nonZeroVec, zeroVec) shouldBe 0.0
    SimilarityUtil.cosineSimilarity(zeroVec, zeroVec) shouldBe 0.0
  }

  it should "calculate similarity correctly for arbitrary vectors" in {
    val vec1 = Array(1.0, 2.0, 3.0)
    val vec2 = Array(4.0, 5.0, 6.0)
    val expectedSimilarity = 0.974631846
    SimilarityUtil.cosineSimilarity(vec1, vec2) shouldBe expectedSimilarity +- 1e-9
  }

  it should "handle single-element vectors" in {
    val vec1 = Array(5.0)
    val vec2 = Array(10.0)
    SimilarityUtil.cosineSimilarity(vec1, vec2) shouldBe 1.0 +- 1e-9
  }

}