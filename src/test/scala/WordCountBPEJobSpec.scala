import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mrunit.mapreduce.{MapDriver, ReduceDriver, MapReduceDriver}
import mapperReducer.WordCountBPEJob.{TokenizerMapper, BPEReducer}
import utils.BytePairUtils
import org.scalatest.BeforeAndAfterEach
import scala.jdk.CollectionConverters._

class WordCountBPEJobSpec extends AnyFlatSpec with Matchers with BeforeAndAfterEach {

  var mapDriver: MapDriver[LongWritable, Text, Text, IntWritable] = _
  var reduceDriver: ReduceDriver[Text, IntWritable, Text, Text] = _
  var mapReduceDriver: MapReduceDriver[LongWritable, Text, Text, IntWritable, Text, Text] = _

  override def beforeEach() {
    val mapper = new TokenizerMapper
    val reducer = new BPEReducer

    mapDriver = MapDriver.newMapDriver(mapper)
    reduceDriver = ReduceDriver.newReduceDriver(reducer)
    mapReduceDriver = MapReduceDriver.newMapReduceDriver(mapper, reducer)
  }

  "TokenizerMapper" should "tokenize input text correctly" in {
    val inputKey = new LongWritable(0)
    val inputValue = new Text("Hello world hello")

    mapDriver
      .withInput(inputKey, inputValue)
      .withOutput(new Text("hello"), new IntWritable(1))
      .withOutput(new Text("world"), new IntWritable(1))
      .withOutput(new Text("hello"), new IntWritable(1))
      .runTest()
  }

  it should "handle empty input" in {
    val inputKey = new LongWritable(0)
    val inputValue = new Text("")

    mapDriver
      .withInput(inputKey, inputValue)
      .runTest()
  }

}