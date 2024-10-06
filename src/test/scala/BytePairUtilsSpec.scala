import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import utils.BytePairUtils

class BytePairUtilsSpec extends AnyFlatSpec with Matchers {

  "BytePairUtils" should "encode and decode text correctly" in {
    val originalText = "Hello, world!"
    val encoded = BytePairUtils.encodeText(originalText)
    val decoded = BytePairUtils.decode(encoded)

    decoded shouldBe originalText
  }

  it should "create correct input-output pairs" in {
    val tokens = Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val windowSize = 3
    val stride = 2

    val pairs = BytePairUtils.createInputOutputPairs(tokens, windowSize, stride)

    pairs should have length 4

    // Compare each pair individually
    pairs.zipWithIndex.foreach { case ((inputArray, outputToken), index) =>
      withClue(s"Pair at index $index:") {
        index match {
          case 0 =>
            inputArray.sameElements(Array(1, 2, 3)) shouldBe true
            outputToken shouldBe 4
          case 1 =>
            inputArray.sameElements(Array(3, 4, 5)) shouldBe true
            outputToken shouldBe 6
          case 2 =>
            inputArray.sameElements(Array(5, 6, 7)) shouldBe true
            outputToken shouldBe 8
          case 3 =>
            inputArray.sameElements(Array(7, 8, 9)) shouldBe true
            outputToken shouldBe 10
        }
      }
    }
  }

  it should "handle empty input for createInputOutputPairs" in {
    val emptyTokens = Seq.empty[Int]
    val pairs = BytePairUtils.createInputOutputPairs(emptyTokens, 3, 2)

    pairs should be (empty)
  }


  it should "encode and decode special characters correctly" in {
    val specialText = "!@#$%^&*()_+{}|:\"<>?`~"
    val encoded = BytePairUtils.encodeText(specialText)
    val decoded = BytePairUtils.decode(encoded)

    decoded shouldBe specialText
  }

  it should "encode and decode Unicode characters correctly" in {
    val unicodeText = "akhil"
    val encoded = BytePairUtils.encodeText(unicodeText)
    val decoded = BytePairUtils.decode(encoded)

    decoded shouldBe unicodeText
  }
}