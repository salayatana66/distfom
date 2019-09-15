/*************************************************************************************************
 *  Test for implementation of various Losses
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/LossTest.scala", $intp)
 ************************************************************************************************/

import distfom.{DistributedDenseVector => DDV,
  Example, ArrayBatcher, Models, 
  DistributedStackedDenseVectors => DSV,
  FomDistDenseVec => FDDV,
  FomDistStackedDenseVec => FDSV,
  MultiLabelSoftMaxLogLoss => MLogLoss,
  RepSDDV, ProductPartitioner}

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{SparseVector => SV, norm, DenseVector => DV,
DenseMatrix => DM}

import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

object L2LossTest {


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

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

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
  // this is the point we check gradients at
  val dvecDoubleRegCheck = DV.rand(size = denseSize, rand = Rand.uniform)

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
  val ddvecRegCheck = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecDoubleRegCheck)
  val fvecRegCheck = new FDDV[Double](ddvecRegCheck, 2)

  /* In this test we check agreement of l2loss and its gradient
   */
  // compute l2loss on the driver
  def l2fun(x : DV[Double]) = {
    var loss = 0.0
    var wsum = 0.0
    for {
      ex <- exReg
      weight = ex.weight
      target = ex.target
      features = ex.features.map(x => (x._1.toInt, x._2))
      idxFeas = features.map(_._1)
      valFeas = features.map(_._2)
      svFeas = new SV[Double](idxFeas, valFeas, denseSize)
      score = x.dot(svFeas)
    } yield ({loss += weight * (score-target) * (score-target)
      wsum += weight
    })
    loss/wsum
  }

  val l2lossValDriver = l2fun(dvecDoubleRegCheck)

  // compute gradient on the driver
  val l2lossGradDriver = new DV[Double](
   { for {
      j <- 0 until denseSize
      perturb =  new SV[Double](Array(j), Array(1e-3), denseSize)
      w_+ = dvecDoubleRegCheck + perturb
      w_- = dvecDoubleRegCheck - perturb
      l2_+ = l2fun(w_+.toDenseVector)
      l2_- = l2fun(w_-.toDenseVector)
    } yield (l2_+ - l2_-)/2e-3 }.toArray,0,1, denseSize)


  val l2loss = Models.linearL2Loss(batcher)

  val (l2lossVal, l2lossGrad) = l2loss.compute(fvecRegCheck)

  val bl2lossGrad = DDV.toBreeze(l2lossGrad.ddvec)

  testLogger.info(s"Difference in final L2Loss: ${l2lossVal - l2lossValDriver}")
  testLogger.info(s"Difference in final L2Loss(Gradient): ${norm(bl2lossGrad- l2lossGradDriver)}")

}

object QuantileLossTest {
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

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

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
  // this is the point we check gradients at
  val dvecDoubleRegCheck = DV.rand(size = denseSize, rand = Rand.uniform)

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
  val ddvecRegCheck = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecDoubleRegCheck)
  val fvecRegCheck = new FDDV[Double](ddvecRegCheck, 2)

  /* In this test we check agreement of quantile loss and its gradient
   */
  // compute quantile on the driver
  val quantile = .3
  def quantileFun(x : DV[Double]) = {
    var loss = 0.0
    var wsum = 0.0
    for {
      ex <- exReg
      weight = ex.weight
      target = ex.target
      features = ex.features.map(x => (x._1.toInt, x._2))
      idxFeas = features.map(_._1)
      valFeas = features.map(_._2)
      svFeas = new SV[Double](idxFeas, valFeas, denseSize)
      score = x.dot(svFeas)
      residual = score - target
      example_loss = if(residual >= 0) quantile * residual
      else (quantile - 1) * residual
    } yield ({loss += weight * example_loss
      wsum += weight
    })
    loss/wsum
  }

  val quantileLossValDriver = quantileFun(dvecDoubleRegCheck)

  // compute gradient on the driver
  val quantileLossGradDriver = new DV[Double](
    { for {
      j <- 0 until denseSize
      perturb =  new SV[Double](Array(j), Array(1e-3), denseSize)
      w_+ = dvecDoubleRegCheck + perturb
      w_- = dvecDoubleRegCheck - perturb
      l2_+ = quantileFun(w_+.toDenseVector)
      l2_- = quantileFun(w_-.toDenseVector)
    } yield (l2_+ - l2_-)/2e-3 }.toArray,0,1, denseSize)

  val quantileLoss = Models.linearQuantileLoss(batcher, quantile=quantile)

  val (quantileLossVal, quantileLossGrad) = quantileLoss.
    compute(fvecRegCheck)

 val bquantileLossGrad = DDV.toBreeze(quantileLossGrad.ddvec)

  testLogger.info(s"Difference in final QuantileLoss: ${quantileLossVal - quantileLossValDriver}")
 testLogger.info(s"Difference in final QuantileLoss(Gradient): ${norm(bquantileLossGrad- quantileLossGradDriver)}")

}

object BinLogLossTest {
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

  val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

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
  // this is the point we check gradients at
  val dvecDoubleRegCheck = DV.rand(size = denseSize, rand = Rand.uniform)

  val exReg = for {
    j <- 0 to exaNum - 1

    // features
    exaFDbl = DV.rand(size = denseSize, rand = Rand.uniform) - 0.5
    weight = DV.rand(size = 1, rand = Rand.uniform).data(0)

    exaF = exaFDbl.data.zipWithIndex.map(x => (x._2.toLong, x._1))

    exid = j.toString
    raw_target = dvecDoubleReg.dot(new DV[Double](exaF.map(_._2), 0, 1,
      denseSize))

    prob = 1.0 / (1.0 + scala.math.exp(-raw_target))
    target = prob > 0.65

  } yield new Example[String, Boolean, Double](exid = exid, target = target,
    weight = weight, features = exaF)

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

  // distribute
  val ddvecRegCheck = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecDoubleRegCheck)
  val fvecRegCheck = new FDDV[Double](ddvecRegCheck, 2)

  /* In this test we check agreement of the bineary log loss and its gradient
   */
  def binaryFun(x : DV[Double]) = {
    var loss = 0.0
    var wsum = 0.0
    for {
      ex <- exReg
      weight = ex.weight
      target = ex.target
      features = ex.features.map(x => (x._1.toInt, x._2))
      idxFeas = features.map(_._1)
      valFeas = features.map(_._2)
      svFeas = new SV[Double](idxFeas, valFeas, denseSize)
      score = x.dot(svFeas)
      prob = 1.0 / (1.0 + scala.math.exp(-score))
      example_loss = if(target) -scala.math.log(prob + 1e-24)
      else -scala.math.log(1.0 - prob + 1e-24)
    } yield ({loss += weight * example_loss
      wsum += weight
    })
    loss/wsum
  }

  val binLogLossValDriver = binaryFun(dvecDoubleRegCheck)

  // compute gradient on the driver
  val binLogLossGradDriver = new DV[Double](
    { for {
      j <- 0 until denseSize
      perturb =  new SV[Double](Array(j), Array(1e-3), denseSize)
      w_+ = dvecDoubleRegCheck + perturb
      w_- = dvecDoubleRegCheck - perturb
      l2_+ = binaryFun(w_+.toDenseVector)
      l2_- = binaryFun(w_-.toDenseVector)
    } yield (l2_+ - l2_-)/2e-3 }.toArray,0,1, denseSize)

  val binLogLoss = Models.linearBinLogLoss(batcher)

  val (binLogLossVal, binLogLossGrad) = binLogLoss.
    compute(fvecRegCheck)

  val bbinLogLossGrad = DDV.toBreeze(binLogLossGrad.ddvec)

  testLogger.info(s"Difference in final Binary LogLoss: ${binLogLossVal - binLogLossValDriver}")
  testLogger.info(s"Difference in final Binary LogLoss(Gradient): ${norm(bbinLogLossGrad- binLogLossGradDriver)}")

}
  object MultiLabelSoftMaxLogLossTest {

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
  val dmatCheck = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)

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
  val ddmatCheck = DSV.fromBreezeDense(
    suggestedNumPartitions=feaParts, sc=sc)(dmatCheck)
  val fmatCheck = new FDSV(ddmatCheck, rowDim)

  /* In this test we check agreement of logloss and its gradient
   */
  // compute logloss on the driver
  def loglossfun(x : DM[Double]) = {
    var loss = 0.0
    var wsum = 0.0

    for {
      ex <- exReg

      // extract examples data
      weight = ex.weight
      target = ex.target
      features = ex.features.map(x => (x._1.toInt, x._2))
      idxFeas = features.map(_._1)
      valFeas = features.map(_._2)
      svFeas = new SV[Double](idxFeas, valFeas, denseSize)

      // update total weight
      _ = (wsum += weight)
      // score & softmax
      raw_scores = x * (svFeas)
      // stable softmax
      norm_raw_scores = raw_scores - bMax(raw_scores)
      exp_scores = bExp(norm_raw_scores)
      smax_scores = exp_scores / bSum(exp_scores)

      // iterate on targets
      (tgtId, tgtWgt) <- target.targetIds.
      zip(target.targetWeights)
      prob = smax_scores(tgtId)

    } yield ( loss -= weight * tgtWgt * scala.math.log(prob + 1e-24))

    loss/wsum
  }

  val loglossDriver = loglossfun(dmatCheck)

  // compute gradient on the driver
  val loglossGradDriver = {
    val out = DM.zeros[Double](rows=rowDim, cols=denseSize)

    for {
      j <- 0 until denseSize
      i <- 0 until rowDim
      perturb = DM.zeros[Double](rows=rowDim, cols=denseSize)
      _ = perturb.update(i, j, 1e-3)
      w_+ = dmatCheck + perturb
      w_- = dmatCheck - perturb
      logloss_+ = loglossfun(w_+)
      logloss_- = loglossfun(w_-)
    } yield {out.update(i, j, (logloss_+ - logloss_-)/2e-3)}
    
    out
  }

  val logloss = Models.linearLogLoss(batcher, rowDim=rowDim)

  val (loglossVal, loglossGrad) = logloss.compute(fmatCheck)

  val bloglossGrad = DSV.toBreeze(loglossGrad.dsvec)

  testLogger.info(s"Difference in final LogLoss: ${loglossVal - loglossDriver}")
  testLogger.info(s"Difference in final LogLoss(Gradient): ${norm(bloglossGrad.flatten() - loglossGradDriver.flatten())}")

}

object FactorizationMachineLoss {

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
  sc.setCheckpointDir("/user/aschioppa/locTest")

  import scala.collection.mutable.{Map => MutMap}
  import org.apache.spark.HashPartitioner
  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // For paired ranking
  import distfom.SigmoidPairedRanking
  import SigmoidPairedRanking.{FeedBackType, PositiveFeedBack, NegativeFeedBack}
  import distfom.LinkedExLabel

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)


  // the linked label type we will be using
  type LEL = LinkedExLabel[String, String, Double]

  // for serialization
  object serMap extends Serializable {
    // mapping examples ids to Long
    val indexerFun = (a:LEL) => a.exId.toLong
  }

  /* Step1: The feature space */
  // feature space size & features partitions
  val denseSize = 200
  // feature separating items & context
  val itemContextBoundary = 99
  val feaParts = 10

  // # of examples & partitions
  val negExaNum = 1500
  val posExaNum=50
  val exaPart = 5
  // number of negative to sample
  val negsToSampleInSampler = 15
  val itersInSampler = 3

  // latent space dimension
  val rowDim = 25

  // initialize random matrices
  val dmatDouble = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)
  val ddmatDouble = DSV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dmatDouble)
  val fdmatDouble = new FDSV(ddmatDouble, rowDim)

  /* Step2: Data creation */

  // create the positive examples
  val exRnk = for {
    j <- 0 to posExaNum-1

    // features
    positiveItem = math.ceil(DV.rand(size = 1, rand = Rand.uniform)
      .data(0) * (itemContextBoundary)).toLong
    contextFeaVals = DV.rand(size = denseSize - itemContextBoundary - 1, rand = Rand.uniform)
    contextFeas = contextFeaVals.data.zipWithIndex
    .map(x => ((x._2 + itemContextBoundary + 1).toLong,
      x._1))
    exaF = (positiveItem.toLong, 1.0) +: contextFeas

    weight = DV.rand(size=1, rand=Rand.uniform).data(0)

    exUid = j.toString
    exLink = j.toString
    exid = LinkedExLabel(exUid, exLink, 1.0)
    target = PositiveFeedBack
    
  } yield new Example[LEL, FeedBackType, Double](exid=exid, target = target,
    weight=weight, features=exaF)

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

  val negSampler = new NegativeIterativeSampler(sampler=sampler, iterations=itersInSampler)
  val negsRDD = negSampler.sample(exRnkRDD)

  /* Caching to ensure deterministic behavior
   *  otherwise distributed results will be different
   */
  negsRDD.map(_.cache.count)

  /* Step3: local socring */
  val posLocal = exRnkRDD.collect()
  //  collect the negatives in each negative batch
  val negsLocalArray = negsRDD.map(_.collect)

  // constructs the factorization machine score
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

  // converts features to a dense vector for scoring
  def convertFeasToDV(feas : Array[(Long, Double)]) = {
    val out = DV.zeros[Double](denseSize)
    for {
      (key, value) <- feas

    } yield (out.update(key.toInt, value))

    out
  }

  def sigmoid(x: Double) = 1.0 / (1.0 + math.exp(-x))

  // to help in the loss computation
  case class LwBuffer(var loss: Double, var weight: Double)

  def localLoss(m: DM[Double]) = {

    // score positive and negative examples on the driver
    val posScoredLocal = for {
      exa <- posLocal
      features = convertFeasToDV(exa.features)

    } yield( exa, fmLocalScore(m, features))


    val negsScoredLocal = negsLocalArray.map(
      negsBatch => for {
        exa <- negsBatch
        features = convertFeasToDV(exa.features)

      } yield( exa, fmLocalScore(m, features))
    )


    val posMap = posScoredLocal.map(x => (x._1.exid.linkId, (x._1.weight,
      x._2))).toMap

    // compute total loss & weights in each negative batch
    val batchLossesAndWeights = negsScoredLocal.map(
      negsScored => {
        val lossesAndWeights = MutMap[String, LwBuffer]()
        for {
          ex <- posMap
        } yield (lossesAndWeights(ex._1) = LwBuffer(0.0,0.0))

        for {
          (negEx, negScore) <- negsScored
          LinkedExLabel(_, linkId, relWeight) = negEx.exid
          (weight, posScore) = posMap(linkId)
          diff = posScore - negScore
          sigDiff = sigmoid(diff)
        } yield ({
          lossesAndWeights(linkId).loss -= relWeight * scala.math.log(sigDiff + 1e-24) * weight
          lossesAndWeights(linkId).weight += relWeight * weight

        })

        val batchLoss = lossesAndWeights.toList.map(_._2.loss).foldLeft(0.0)(_+_)
        val batchWeight = lossesAndWeights.toList.map(_._2.weight).foldLeft(0.0)(_+_)

        (batchLoss, batchWeight)
      })

    val totalLoss = batchLossesAndWeights.map(_._1).foldLeft(0.0)(_ + _)
    val totalWeight = batchLossesAndWeights.map(_._2).foldLeft(0.0)(_ + _)

    totalLoss/totalWeight
  }

  def localGrad(m: DM[Double], i: Int, j: Int) = {
    // up & down
    val m_up = m.copy
    m_up.update(i, j, m(i, j) + 1e-3)

    val m_down = m.copy
    m_down.update(i, j, m(i, j) - 1e-3)

    val f_up = localLoss(m_up)
    val f_down = localLoss(m_down)

    (f_up - f_down) / 2e-3

  }

  println("Loading Positive Batch")
  val (posTotalWeight, posDistFeas, posDistTgts) = posBatcher.next(depth=2)

  
  // hash partitioner for aggregating between positive & negative examples
  val hp = new HashPartitioner(256)

  val (distLoss, totalGrad) = SigmoidPairedRanking.computeOnNegBatches(
    posDistFeas, posDistTgts, negsRDD,
    fdmatDouble.dsvec,
    posBatcher,
    indexer=serMap.indexerFun,
    linkPartitioner=hp,
    rowDim=rowDim)

  val locLoss = localLoss(dmatDouble)
  testLogger.info(s"Discrepancy in losses: ${locLoss - distLoss}")
  val btotalGrad = DSV.toBreeze(totalGrad.get)
  val locGrad = DM.zeros[Double](rows=btotalGrad.rows, cols=btotalGrad.cols)

  // test in sample points as explicit differentiation is super slow
  for {
    i <- 5 until 7
    j <- 10 until 20
    _ = (println(s"$i, $j"))
    _ = (locGrad.update(i, j, localGrad(dmatDouble, i, j)))
    _ = testLogger.info(s"Discrepancy in gradients @($i, $j): ${locGrad(i, j) - btotalGrad(i,j)}")
  }yield ()

}
