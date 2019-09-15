/*************************************************************************************************
 *  Test for implementation of plain SGD
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/SGDTest.scala", $intp)
 ************************************************************************************************/


import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV, SparseVector => SV}
import breeze.stats.distributions.Rand
import distfom.{ArrayBatcher, Example, Models, DistributedStackedDenseVectors => DSV, FomDistStackedDenseVec => FDSV, StochasticGradientDescent => SGD}
import distfom.DistributedDiffFunction.{DDF, FV, linCombF, regularizationL2}
// logger
import org.apache.log4j.{Logger,Level}

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem,Path}

object SGDSigmoidPairedRankingLoss {
  /* the SparkSession is created with a boilerplate; in reality
 * this is just supplied to prevent the IDE from flagging errors, and
 * the tests can be run in cluster/client mode (on YARN or MESOS)
 */
  val spark = SparkSession.
    builder().
    master("local").
    appName("DistributedDenseVector").
    getOrCreate()

  val sc = spark.sparkContext

  sc.setLogLevel("FATAL")

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/SGDTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/SGDTest")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  import distfom.SigmoidPairedRanking.{PositiveFeedBack, NegativeFeedBack,
    FeedBackType}

  // the linked label type we will be using
  import distfom.LinkedExLabel
  type LEL = LinkedExLabel[String, String, Double]

  // for serialization
  object serMap extends Serializable {
    // mapping examples ids to Long
    val indexerFun = (a:LEL) => a.exId.toLong
  }

  /* Step1: The feature space */
  // feature space size & features partitions
  val denseSize = 50
  // feature separating items & context
  val itemContextBoundary = 19
  val feaParts = 5

  // # of examples & partitions
  val posExaNum=5000
  val exaPart = 2
  // number of negative to sample
  val negsToSampleInSampler = 15
  val itersInSampler = 4
  val negExaNum = posExaNum * negsToSampleInSampler * itersInSampler

  // latent space dimension
  val rowDim = 25

  // computes the factorization machine score
  def fmLocalScore(m: DM[Double], v: DV[Double]) = {
    var target = 0.0

    for {
      alpha <- 0 until m.cols
      beta <- (alpha+1) until m.cols
      add_ = m(::, alpha).dot(m(::, beta))
      mul_ = add_ * v(alpha) * v(beta)
      _ = (target += mul_)
    } yield ()

    target
  }

  // initialize model matrix
  val dmatDouble = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)
  // the starting point for optimization
  val dmatStart = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)

  val numTopItems = 5
  // create the positive examples

  val exRnk = for {
    j <- 0 to 3 //posExaNum-1

    // first contruct the contextual features
    contextFeaVals = DV.rand(size = denseSize - itemContextBoundary - 1, rand = Rand.uniform)
    contextFeas = contextFeaVals.data.zipWithIndex
      .map(x => ((x._2 + itemContextBoundary + 1).toLong,
        x._1))

    // score different positive items against the context
    scoredItems = for {
      ij <- 0 to itemContextBoundary

      positiveItem = math.ceil(DV.rand(size = 1, rand = Rand.uniform)
        .data(0) * (itemContextBoundary)).toLong
      exaFeatures = (positiveItem.toLong, 1.0) +: contextFeas
      svFeatures = new SV[Double](index = exaFeatures.map(_._1.toInt), data =
        exaFeatures.map(_._2), length = denseSize)

      score = fmLocalScore(dmatDouble, svFeatures.toDenseVector)

    } yield (ij, score)

    topItems = scoredItems.sortBy(-_._2).map(_._1).take(numTopItems)

    // produce the top numTopItems examples
    item <- topItems
    exaFeatures = (item.toLong, 1.0) +: contextFeas

    weight = DV.rand(size=1, rand=Rand.uniform).data(0)

    exUid = j.toString
    exLink = j.toString
    exid = LinkedExLabel(exUid, exLink, 1.0)
    target = PositiveFeedBack

  } yield new Example[LEL, FeedBackType, Double](exid=exid, target = target,
    weight=weight, features=exaFeatures)

  // the negative sampler
  // for multinomial sampling
  import breeze.stats.distributions.{Multinomial, RandBasis}

  val sampler = (label: LinkedExLabel[String, String, Double],
                 target: FeedBackType, weight: Double, features: Array[(Long, Double)]) => {

    // sample from items different from the current one
    val positiveItemId = features(0)._1
    val samplingWeights = Array.fill(n=itemContextBoundary + 1)(1.0 / itemContextBoundary)
      .zipWithIndex.map(x => if(x._2 != positiveItemId) x._1 else 0.0)

    val labMultinomial = new Multinomial(new DV(samplingWeights))

    val out = for {
      i <- 0 until negsToSampleInSampler
      // to ensure that the two different samples are really different
      _ = RandBasis.withSeed(i)
      LinkedExLabel(exUid, exLink, posRelWeight) = label
      newExUid = ((exUid.toInt * negsToSampleInSampler) + i).toString
      target = NegativeFeedBack
      newRelWeight = posRelWeight / (negsToSampleInSampler * itersInSampler)
      newExId = LinkedExLabel(newExUid, exLink, newRelWeight)

      // replace the first feature with the new item
      newItem = labMultinomial.draw()
      newFeatures = (newItem.toLong, 1.0) +: features.slice(1, features.size)
      outExa = Example[LEL, FeedBackType, Double](exid=newExId, target = target,
        weight=weight, features=newFeatures)
    } yield outExa

    out.toArray
  }

  // parallelize examples
  val exRnkRDD = sc.parallelize(exRnk, numSlices=4)

  // instantiate a Batcher
  val posBatcher = new ArrayBatcher[LEL, FeedBackType, Double](
    indexer=serMap.indexerFun, maxExampleIdx=negExaNum,
    maxFeaIdx=denseSize, suggPartExamples=exaPart, suggPartFeas=feaParts,
    batches = Array(exRnkRDD))

  // sample the negatives
  import distfom.NegativeIterativeSampler

  // instantiate the negative sampler
  val negSampler = new NegativeIterativeSampler(sampler=sampler, iterations=itersInSampler)

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  val ddmatStart = DSV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dmatStart)

  val fvecDoubleStart = new FDSV[Double](ddmatStart, 2)

  import org.apache.spark.HashPartitioner

  // implicit conversions for differentiable functions
  import FDSV.implicits._

  val sigLoss = Models.SigmoidPairedRankingLoss(posBatcher, negSampler, indexer=serMap.indexerFun,
    linkPartitioner=new HashPartitioner(256), batcherDepth=2)

  val regularization: DDF[FV[FDSV[Double], Double], Double] = regularizationL2[FDSV[Double], Double]

  val regSigLoss = linCombF(1.0, sigLoss, 1e-3, regularization)
  // optimizer
  val SGDMin = new SGD[FDSV[Double], Double](defaultStepSize = .1,
  powDec = 1 / 30.0, decayFactor = 1.0, maxIter = 50)

  val finalWeight = SGDMin.minimize(f = regSigLoss, init = fvecDoubleStart)

  hfs.delete(chkPath)

}

