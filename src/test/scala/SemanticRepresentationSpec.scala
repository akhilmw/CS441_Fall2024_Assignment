import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mrunit.mapreduce.{MapDriver, ReduceDriver, MapReduceDriver}
import mapperReducer.SemanticRepresentation.{EmbeddingMapper, SimilarityReducer}
import org.apache.hadoop.conf.Configuration
import org.scalatest.BeforeAndAfterEach
import scala.jdk.CollectionConverters._

class SemanticRepresentationSpec extends AnyFlatSpec with Matchers with BeforeAndAfterEach {

  var mapDriver: MapDriver[Object, Text, Text, Text] = _
  var reduceDriver: ReduceDriver[Text, Text, Text, Text] = _
  var mapReduceDriver: MapReduceDriver[Object, Text, Text, Text, Text, Text] = _

  override def beforeEach() {
    val mapper = new EmbeddingMapper
    val reducer = new SimilarityReducer

    mapDriver = MapDriver.newMapDriver(mapper)
    reduceDriver = ReduceDriver.newReduceDriver(reducer)
    mapReduceDriver = MapReduceDriver.newMapReduceDriver(mapper, reducer)

    // Set the configuration for topN
    val conf = new Configuration()
    conf.setInt("topN", 3)
    reduceDriver.setConfiguration(conf)
    mapReduceDriver.setConfiguration(conf)
  }

  "EmbeddingMapper" should "correctly map input to word-embedding pairs" in {
    val inputKey = new LongWritable(0)
    val inputValue = new Text("1,word1,0.1,0.2,0.3")

    mapDriver
      .withInput(inputKey, inputValue)
      .withOutput(new Text("word1"), new Text("0.1,0.2,0.3"))
      .runTest()
  }

  it should "skip header line" in {
    val inputKey = new LongWritable(0)
    val inputValue = new Text("TokenID,Word,Embeddings")

    mapDriver
      .withInput(inputKey, inputValue)
      .runTest()
  }

  it should "handle commas within quoted fields" in {
    val inputKey = new LongWritable(0)
    val inputValue = new Text("1,\"word, with, commas\",0.1,0.2,0.3")

    mapDriver
      .withInput(inputKey, inputValue)
      .withOutput(new Text("word, with, commas"), new Text("0.1,0.2,0.3"))
      .runTest()
  }

}