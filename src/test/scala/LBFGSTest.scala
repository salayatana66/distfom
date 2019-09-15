/*************************************************************************************************
 *  Test for implementation of LBFGS
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/LBFGSTest.scala", $intp)
 ***********************************************************************************************/

import distfom.{DistributedDenseVector => DDV,
  Example, ArrayBatcher, Models, 
  DistributedStackedDenseVectors => DSV,
  FomDistDenseVec => FDDV, LBFGS, FomDistVec => FBase,
  FomDistStackedDenseVec => FDSV,
  MultiLabelSoftMaxLogLoss => MLogLoss,
DistributedDiffFunction}

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{SparseVector => SV, norm, DenseVector => DV,
DenseMatrix => DM}

import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger, Level}

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem, Path}

object LBFGSRegressionTest {

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
  val hdconf = sc.hadoopConfiguration
  val hfs = FileSystem.get(hdconf)

  val chkPath = new Path("/tmp/LBFGSTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/LBFGSTest")

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /* This test is fitting a linear regression on L2Loss
   */

  /* Prepare features for Linear Regression */
  // feature space size & features partitions
  val denseSize = 1000
  val feaParts = 10

  // # of examples & partitions
  val exaNum = 50
  val exaPart = 5

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // initialize random vectors
  val dvecDoubleReg = DV.rand(size = denseSize, rand = Rand.uniform)
  // we initialize weights randomly
  val dvecDoubleStart = DV.rand(size = denseSize, rand = Rand.uniform)

  val exReg = for {
    j <- 0 to exaNum-1

    // features
    exaFDbl = DV.rand(size = denseSize, rand = Rand.uniform)
    weight = DV.rand(size=1, rand=Rand.uniform).data(0)

    exaF = exaFDbl.data.zipWithIndex.map(x => (x._2.toLong, x._1))

    exid = j.toString
    target = dvecDoubleReg.dot(new DV[Double](exaF.map(_._2),0,1,
      denseSize))
    
  } yield new Example[String, Double, Double](exid=exid, target = target,
    weight=weight, features=exaF)

  // parallelize examples
  val exRegRDD = sc.parallelize(exReg,
    numSlices=4)

  // for serialization
  object serMap extends Serializable {
    // mapping examples ids to Long
    val serFun = (a:String) => a.toLong
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, Double, Double](
    indexer=serMap.serFun, maxExampleIdx=exaNum,
    maxFeaIdx=denseSize, suggPartExamples=exaPart, suggPartFeas=feaParts,
    batches = Array(exRegRDD))

  // distribute
  val ddvecDoubleStart = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecDoubleStart)
  val fvecDoubleStart = new FDDV[Double](ddvecDoubleStart, 2)

  val l2loss = Models.linearL2Loss(batcher)

  // optimizer
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m = 5,
    maxIter=100, tolerance=1e-4, interruptLinSteps = 1)

  // implicit conversions
  import FDDV.implicits._

  val finalWeight = LBFGSMin.minimize(f = l2loss, init = fvecDoubleStart)

  hfs.delete(chkPath)
}

object LBFGSQuantileTest {

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
  val hdconf = sc.hadoopConfiguration
  val hfs = FileSystem.get(hdconf)

  val chkPath = new Path("/tmp/LBFGSTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/LBFGSTest")

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /* This test is fitting a linear regression on Quantile Loss
   */

  /* Prepare features for Linear Regression */
  // feature space size & features partitions
  val denseSize = 1000
  val feaParts = 10

  // # of examples & partitions
  val exaNum = 50
  val exaPart = 5

  val quantile = .67

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // initialize random vectors
  val dvecDoubleReg = DV.rand(size = denseSize, rand = Rand.uniform)
  // we initialize weights randomly
  val dvecDoubleStart = DV.rand(size = denseSize, rand = Rand.uniform)

  val exReg = for {
    j <- 0 to exaNum-1

    // features
    exaFDbl = DV.rand(size = denseSize, rand = Rand.uniform)
    weight = DV.rand(size=1, rand=Rand.uniform).data(0)

    exaF = exaFDbl.data.zipWithIndex.map(x => (x._2.toLong, x._1))

    exid = j.toString
    target = dvecDoubleReg.dot(new DV[Double](exaF.map(_._2),0,1,
      denseSize))

  } yield new Example[String, Double, Double](exid=exid, target = target,
    weight=weight, features=exaF)

  // parallelize examples
  val exRegRDD = sc.parallelize(exReg,
    numSlices=4)

  // for serialization
  object serMap extends Serializable {
    // mapping examples ids to Long
    val serFun = (a:String) => a.toLong
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, Double, Double](
    indexer=serMap.serFun, maxExampleIdx=exaNum,
    maxFeaIdx=denseSize, suggPartExamples=exaPart, suggPartFeas=feaParts,
    batches = Array(exRegRDD))

  val quantileLoss = Models.linearQuantileLoss(batcher, quantile)

  // distribute
  val ddvecDoubleStart = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecDoubleStart)
  val fvecDoubleStart = new FDDV[Double](ddvecDoubleStart, 2)

  // optimizer
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m = 5,
    maxIter=100, tolerance=1e-4, interruptLinSteps = 1)

  // implicit conversions
  import FDDV.implicits._

  val finalWeight = LBFGSMin.minimize(quantileLoss, fvecDoubleStart)

  hfs.delete(chkPath)

}

object LBFGSBinLogLossTest {

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
  val hdconf = sc.hadoopConfiguration
  val hfs = FileSystem.get(hdconf)

  val chkPath = new Path("/tmp/LBFGSTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/LBFGSTest")

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /* This test is fitting a linear regression on Quantile Loss
   */

  /* Prepare features for Linear Regression */
  // feature space size & features partitions
  val denseSize = 1000
  val feaParts = 10

  // # of examples & partitions
  val exaNum = 50
  val exaPart = 5

  val quantile = .67

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // initialize random vectors
  val dvecDoubleReg = DV.rand(size = denseSize, rand = Rand.uniform) - 0.5
  // we initialize weights randomly
  val dvecDoubleStart = DV.rand(size = denseSize, rand = Rand.uniform)- 0.5

  val exReg = for {
    j <- 0 to exaNum-1

    // features
    exaFDbl = DV.rand(size = denseSize, rand = Rand.uniform)
    weight = DV.rand(size=1, rand=Rand.uniform).data(0)

    exaF = exaFDbl.data.zipWithIndex.map(x => (x._2.toLong, x._1))

    exid = j.toString
    raw_score = dvecDoubleReg.dot(new DV[Double](exaF.map(_._2),0,1,
      denseSize))
    prob = 1.0 / (1.0 + scala.math.exp(-raw_score))
    target = prob > 0.5

  } yield new Example[String, Boolean, Double](exid=exid, target = target,
    weight=weight, features=exaF)

  // parallelize examples
  val exRegRDD = sc.parallelize(exReg,
    numSlices=4)

  // for serialization
  object serMap extends Serializable {
    // mapping examples ids to Long
    val serFun = (a:String) => a.toLong
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, Boolean, Double](
    indexer=serMap.serFun, maxExampleIdx=exaNum,
    maxFeaIdx=denseSize, suggPartExamples=exaPart, suggPartFeas=feaParts,
    batches = Array(exRegRDD))

  val binLogLoss = Models.linearBinLogLoss(batcher)

  // distribute
  val ddvecDoubleStart = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecDoubleStart)
  val fvecDoubleStart = new FDDV[Double](ddvecDoubleStart, 2)

  // optimizer
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m = 5,
    maxIter=100, tolerance=1e-4, interruptLinSteps = 1)

  // implicit conversions
  import FDDV.implicits._

  val finalWeight = LBFGSMin.minimize(binLogLoss, fvecDoubleStart)

  hfs.delete(chkPath)

}

object LBFGSMultiLogLossTest {

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
  val hdconf = sc.hadoopConfiguration
  val hfs = FileSystem.get(hdconf)

  val chkPath = new Path("/tmp/LBFGSTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/LBFGSTest")

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /* Prepare features for Linear Regression */
  // feature space size & features partitions
  val denseSize = 1000
  val feaParts = 10
  // number of rows
  val rowDim = 7

  // # of examples & partitions
  val exaNum = 50
  val exaPart = 5

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // initialize random matrices
  val dmatModel = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)
  val dmatStart = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)

  import MLogLoss.MultiLabelTarget

  // breeze imports
  import breeze.linalg.{max => bMax, sum => bSum}
  import breeze.numerics.{exp => bExp, log => bLog}
  // for multinomial sampling
  import breeze.stats.distributions.Multinomial

  // We sample 1, 2 or 3 labels with equal probability
  val labMultinomial = new Multinomial(params=DV(.5, .25, .25))

  // construct examples
  val exReg = for {
    j <- 0 to exaNum-1

    // features
    exaFDbl = DV.rand(size = denseSize, rand = Rand.uniform)
    weight = DV.rand(size=1, rand=Rand.uniform).data(0)
    // arrange features in shape for distributing them
    exaF = exaFDbl.data.zipWithIndex.map(x => (x._2.toLong, x._1))

    // example id
    exid = j.toString

    // we score against dmatModel
    fea_vec = new DV[Double](exaF.map(_._2),0,1,
      denseSize)
    raw_scores = dmatModel * fea_vec
    // stable softmax
    norm_raw_scores = raw_scores - bMax(raw_scores)
    exp_scores = bExp(norm_raw_scores)
    smax_scores = exp_scores / bSum(exp_scores)

    // Instance probability to sample the scores
    distro = new Multinomial(params=smax_scores)

    num_targets = labMultinomial.draw()
    tWeight = 1.0 / (num_targets.toDouble + 1)
    //tWeight = 1.0/5
    target_labels = for {
      j <- 0 to num_targets
      target_label = distro.draw()
    } yield target_label

    // target_labels = Array(0,1,2,3,4)
    target = MultiLabelTarget(target_labels.toArray,
      target_labels.map(x => tWeight).toArray)

  } yield new Example[String, MultiLabelTarget[Double], Double](exid=exid,
    target = target, weight=weight, features=exaF)

  // parallelize examples
  val exRegRDD = sc.parallelize(exReg, numSlices=4)

  // for serialization
  object serMap extends Serializable {
    // mapping examples ids to Long
    val serFun = (a:String) => a.toLong
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, MultiLabelTarget[Double], Double](
    indexer=serMap.serFun, maxExampleIdx=exaNum,
    maxFeaIdx=denseSize, suggPartExamples=exaPart, suggPartFeas=feaParts,
    batches = Array(exRegRDD))

  // distribute
  val ddmatStart = DSV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dmatStart)
  val fvecDoubleStart = new FDSV[Double](ddmatStart, 2)

  val logloss = Models.linearLogLoss(batcher=batcher, rowDim=rowDim, depth=2)

  // optimizer
  val LBFGSMin = new LBFGS[FDSV[Double], Double](m = 5,
    maxIter=100, tolerance=1e-4, interruptLinSteps = 1)

  // implicit conversions
  import FDSV.implicits._

  val finalWeight = LBFGSMin.minimize(f = logloss, init = fvecDoubleStart)

  hfs.delete(chkPath)

}

object LBFGSSigmoidPairedRankingLoss {
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

  @transient val chkPath = new Path("/tmp/LBFGSTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/LBFGSTest")

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
    val cols = m.cols
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
  val sigLoss = Models.SigmoidPairedRankingLoss(posBatcher, negSampler, indexer=serMap.indexerFun,
    linkPartitioner=new HashPartitioner(256), batcherDepth=2)

  // optimizer
  val LBFGSMin = new LBFGS[FDSV[Double], Double](m = 5,
    maxIter=20, tolerance=1e-4, interruptLinSteps = 1)

  // implicit conversions for differentiable functions
  import FDSV.implicits._

  val finalWeight = LBFGSMin.minimize(f = sigLoss, init = fvecDoubleStart)

  hfs.delete(chkPath)

}

object LBFGSOptDualSinkhornTest {
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

  import distfom.{DualSpace, DistributedDenseVector => DDV,
    FomDistDenseVec => FDV}

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/LBFGSTest")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/LBFGSTest")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  // X space dimension
  val Ex = 1024
  // Y space dimension
  val Ey = 2048
  // elements per block
  val eb = 128
  // dimension space of X & Y
  val dimS = 2
  // normalization factor to check in cost
  //val normFactor = 1.233e4

  // construct a pairing context
  val ctx = DualSpace.PairingContext(Ex, Ey, eb)

  // sample on X
  val xPoints = for {
    i <- Range(0, Ex)
    vecX = DV.rand(size=dimS, rand=Rand.uniform)
  } yield (i.toLong, vecX)

  // sample on Y
  val yPoints = for {
    j <- Range(0, Ey)
    vecY = DV.rand(size=dimS, rand=Rand.uniform)
  } yield (j.toLong, vecY)

  // cost Function
  val costFun = (a: DV[Double], b: DV[Double]) => 0.5 * ((a-b).dot(a-b))

  // sinkhorn regularization
  val sinkReg = .1

  // parallelize
  val rXPoints = sc.parallelize(xPoints, 3)
  val rYPoints = sc.parallelize(yPoints, 3)

  val costs = DualSpace.computePairwiseCosts(rXPoints, rYPoints, ctx, costFun)
  costs.setName("costs")
  costs.cache.count

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // initialize weight
  val initVec = DV.zeros[Double](Ex + Ey)
  val dInitVec = DDV.fromBreezeDense(initVec, eb, sc)
  val fInitVec = new FDDV[Double](dInitVec, 2)

  val transportLoss = DualSpace.potentialLossWSinkhornRegularization(ctx, sinkReg, costs, 2)

  // optimizer
  val initFval = transportLoss.computeValue(fInitVec)
  testLogger.info(s"Initial value: $initFval")

  // Backtrack line search
  import distfom.BackTrackingLineSearch
  val lsearch = new BackTrackingLineSearch[Double](
    enforceStrongWolfeConditions = false, enforceWolfeConditions = false)
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m = 5,
    maxIter=50, tolerance=1e-8, interruptLinSteps = 1,
    suppliedLineSearch = Some(lsearch))

  // implicit conversions for differentiable functions
  //import FDDV.implicits._

  val finalWeight = LBFGSMin.minimize(f = transportLoss, init = fInitVec)

  hfs.delete(chkPath)

}