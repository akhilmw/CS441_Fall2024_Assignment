//import com.knuddels.jtokkit.Encodings
//import com.knuddels.jtokkit.api.ModelType
//import org.tensorflow.{Graph, Session, Tensor}
//import org.tensorflow.op.Ops
//import org.tensorflow.types.TFloat32
//import org.tensorflow.types.TInt32
//import org.tensorflow.ndarray.Shape
//import org.tensorflow.ndarray.buffer.DataBuffers
//import org.tensorflow.Operand
//import org.tensorflow.op.core.Gradients
//import org.tensorflow.op.core.Placeholder
//import org.tensorflow.op.train.ApplyGradientDescent
//
//import java.util
//import scala.util.Random
//
//object SlidingWindowTokenizationAndEmbeddingLearning {
//
//  def main(args: Array[String]): Unit = {
//    // Sample input text (in a real scenario, this would be a large corpus)
//    val inputText = """
//                      |The quick brown fox jumps over the lazy dog.
//                      |Machine learning models process vast amounts of data.
//                      |Natural language processing has made significant advances.
//                      |Transformers have revolutionized many NLP tasks.
//      """.stripMargin
//
//    // Tokenization and sliding window parameters
//    val windowSize = 5
//    val stride = 1
//
//    // Step 1: Tokenization with JTokenkit and sliding window
//    val (tokenIds, windows, vocabSize) = tokenizeWithSlidingWindow(inputText, windowSize, stride)
//    println(s"Total tokens: ${tokenIds.length}")
//    println(s"Number of windows: ${windows.length}")
//
//    // Step 2: Embedding learning with TensorFlow
//    learnEmbeddings(windows, vocabSize)
//  }
//
//  def tokenizeWithSlidingWindow(input: String, windowSize: Int, stride: Int): (Seq[Int], Seq[Seq[Int]], Int) = {
//    val registry = Encodings.newDefaultEncodingRegistry()
//    val encoding = registry.getEncodingForModel(ModelType.GPT_4)
//
//    val tokenIds = encoding.encode(input).toArray.toSeq
//    val windows = tokenIds.sliding(windowSize, stride).toSeq
//    val actualVocabSize = tokenIds.max + 1
//
//    (tokenIds, windows, actualVocabSize)
//  }
//
//  def learnEmbeddings(windows: Seq[Seq[Int]], vocabSize: Int): Unit = {
//    val embeddingDim = 50 // Arbitrary embedding size, adjust as needed
//
//    val graph = new Graph()
//    val session = new Session(graph)
//
//    try {
//      val tf = Ops.create(graph)
//
//      // Initialize random embeddings
//      val randomEmbeddings = tf.variable(tf.random.randomUniform(
//        tf.constant(Array(vocabSize.toLong, embeddingDim.toLong)),
//        classOf[TFloat32]
//      ))
//
//      // Create placeholders for input data
//      val centerWordsPh = tf.placeholder(classOf[TInt32], Placeholder.shape(Shape.scalar()))
//      val contextWordsPh = tf.placeholder(classOf[TInt32], Placeholder.shape(Shape.scalar()))
//      val negativeWordsPh = tf.placeholder(classOf[TInt32], Placeholder.shape(Shape.scalar()))
//
//      // Define model and loss function
//      def skipGramModel(centerWord: Operand[TInt32], contextWord: Operand[TInt32]): Operand[TFloat32] = {
//        val centerEmbedding = tf.gather(randomEmbeddings, centerWord, tf.constant(0))
//        val contextEmbedding = tf.gather(randomEmbeddings, contextWord, tf.constant(0))
//        tf.reduceSum(tf.math.mul(centerEmbedding, contextEmbedding), tf.constant(0))
//      }
//
//      def negativeSamplingLoss(positivePair: Operand[TFloat32], negativePairs: Operand[TFloat32]): Operand[TFloat32] = {
//        val positiveLoss = tf.math.log(tf.math.sigmoid(positivePair))
//        val negativeLoss = tf.math.log(tf.math.sigmoid(tf.math.neg(negativePairs)))
//        tf.math.neg(tf.math.add(positiveLoss, negativeLoss))
//      }
//
//      val positivePair = skipGramModel(centerWordsPh, contextWordsPh)
//      val negativePair = skipGramModel(centerWordsPh, negativeWordsPh)
//      val loss = negativeSamplingLoss(positivePair, negativePair)
//
//      // Compute gradients
//      val gradients = tf.gradients(loss, util.Arrays.asList(randomEmbeddings))
//
//      // Set up training
//      val learningRate = tf.constant(0.01f)
//      val trainOp = tf.train.applyGradientDescent(randomEmbeddings, learningRate, gradients.dy(0).asInstanceOf[Operand[TFloat32]])
//
//      // Training loop
//      val numEpochs = 10000
//      val (centerWords, contextWords, negativeWords) = createTrainingData(windows, vocabSize)
//
//      for (epoch <- 1 to numEpochs) {
//        // Run the training operation
//        session.runner()
//          .feed(centerWordsPh.asOutput(), TInt32.scalarOf(centerWords(epoch % centerWords.length)))
//          .feed(contextWordsPh.asOutput(), TInt32.scalarOf(contextWords(epoch % contextWords.length)))
//          .feed(negativeWordsPh.asOutput(), TInt32.scalarOf(negativeWords(epoch % negativeWords.length)))
//          .addTarget(trainOp)
//          .run()
//
//        if (epoch % 1000 == 0) {
//          val lossValue = session.runner()
//            .feed(centerWordsPh.asOutput(), TInt32.scalarOf(centerWords(epoch % centerWords.length)))
//            .feed(contextWordsPh.asOutput(), TInt32.scalarOf(contextWords(epoch % contextWords.length)))
//            .feed(negativeWordsPh.asOutput(), TInt32.scalarOf(negativeWords(epoch % negativeWords.length)))
//            .fetch(loss)
//            .run()
//            .get(0)
//            .asInstanceOf[TFloat32]
//          println(s"Epoch $epoch, Loss: ${lossValue.getFloat()}")
//        }
//      }
//
//      // Retrieve learned embeddings
//      val learnedEmbeddings = session.runner().fetch(randomEmbeddings).run().get(0).asInstanceOf[TFloat32]
//
//      // Print a few learned embeddings
//      println("Learned embeddings for first 3 tokens:")
//      for (i <- 0 until math.min(3, vocabSize)) {
//        val embedding = for (j <- 0 until math.min(5, embeddingDim)) yield {
//          learnedEmbeddings.getFloat(i.toLong, j.toLong)
//        }
//        println(s"Token $i: ${embedding.mkString("[", ", ", ", ...")}")
//      }
//
//    } finally {
//      session.close()
//      graph.close()
//    }
//  }
//
//  def createTrainingData(windows: Seq[Seq[Int]], vocabSize: Int): (Seq[Int], Seq[Int], Seq[Int]) = {
//    val centerWords = windows.flatMap(window => window.init)
//    val contextWords = windows.flatMap(window => window.tail)
//    val negativeWords = centerWords.map(_ => Random.nextInt(vocabSize))
//    (centerWords, contextWords, negativeWords)
//  }
//}