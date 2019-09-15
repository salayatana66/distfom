/*************************************************************************************************
 *  Test for implementation of Scorers for linear models
 *  and factorization machines
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/ScorerTest.scala", $intp)
 ************************************************************************************************/

import distfom.{DistributedDenseVector => DDV,
  Example, ArrayBatcher, DistributedFeatures => DFea,
  RepDDV, LinearScorer, FactorizationMachine,
  QuotientPartitioner, ProductPartitioner,
RichTarget, DistributedStackedDenseVectors => DSV, RepSDDV}

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{SparseVector => SV, norm, DenseVector => DV,
DenseMatrix => DM}

import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

object ScorerTest {

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

  // feature space size & features partitions
  val denseSize = 1000
  val feaParts = 10

  // # of examples & partitions
  val exaNum = 50
  val exaPart = 5


  // Import the field implicits
  import distfom.CommonImplicitGenericFields._

  // check scalar scores
  def checkScalarFloat(s: Map[Long, Float],
    t: Map[Long,RichTarget[String,Float,Float]]) = {
    val pairs = s.zip(t.mapValues(x => x.target))

    pairs.map(x => math.abs(x._1._2 - x._2._2)).
      foldLeft(0f)(_ + _)
  }

  // check scalar scores
  def checkVecFloat(s: Map[Long, DV[Float]],
    t: Map[Long, DV[Float]]) = {

    val pairs = s.zip(t)

    pairs.map(x => norm(x._1._2 - x._2._2).toFloat).
      foldLeft(0f)(_ + _)
  }

  import FactorizationMachine.FMScore

  def checkFMFloat(s: Map[Long, (Float, DV[Float])],
    t: Map[Long, FMScore[Float]]) = {

      val pairs = s.zip(t)

      val scDiff = pairs.map(x => math.abs(x._1._2._1
        - x._2._2.score)).foldLeft(0f)(_ + _)

      scDiff

  }


  // initialize random vectors
  val dvecDouble = DV.rand(size = denseSize, rand = Rand.uniform)
  val dvecFloat = new DV[Float](dvecDouble.data.map(_.toFloat),
    0, 1, dvecDouble.size)

  // distribute
  val ddvecFloat = DDV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dvecFloat)

  val exSq = for {
    j <- 0 to exaNum-1

    // features
    exaFDbl = DV.rand(size = denseSize, rand = Rand.uniform)
    exaF = exaFDbl.data.map(_.toFloat).zipWithIndex.
    map(x => (x._2.toLong, x._1))

    exid = j.toString
    target = dvecFloat.dot(new DV[Float](exaF.map(_._2),0,1,
      denseSize))
    
    } yield new Example[String, Float, Float](exid=exid, target = target,
      features=exaF)

  val exRDD = sc.parallelize(exSq, numSlices=10)

  // for serialization
  object serMap extends Serializable {

    val serFun = (a:String) => a.toLong
  }

  // get a batch of features
  val batcher = new ArrayBatcher[String, Float, Float](
    indexer=serMap.serFun, maxExampleIdx=exaNum.toLong,
    maxFeaIdx=denseSize.toLong, suggPartExamples=exaPart, suggPartFeas=feaParts,
    batches = Array(exRDD))

  val (_, feas1, tgts1) = batcher.next()
  feas1.self.setName("feas1")
  feas1.self.persist.count

  // get linear weights
  val linWg = RepDDV.apply(ddvecFloat, feas1.self.partitioner.get.
    asInstanceOf[ProductPartitioner].leftPartitioner)
  linWg.self.setName("Weights")

  val scores1 = LinearScorer.apply(feas1, linWg)

  // pull results to driver
  val sMp1 = scores1.self.collect.toMap
  val tMp1 = tgts1.self.collect.toMap

 testLogger.info(s"scoring scalar consistency: ${checkScalarFloat(sMp1, tMp1)}")

  /* this test checks the scorer which takes
   * stacked dense vectors (e.g. for a multinomial classifier)
   */

  def convertMatToFloat(a: DM[Double]): DM[Float] = {
    new DM[Float](rows=a.rows, cols=a.cols,
    data=a.data.map(_.toFloat), offset=0,
    majorStride=a.majorStride, isTranspose=a.isTranspose)   
  }

  // number of rows
  val rowDim = 5
  // initialize random matrices
  val dmatDouble = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)
  val dmatFloat = convertMatToFloat(dmatDouble)

  // distribute
  val ddmatFloat = DSV.fromBreezeDense(suggestedNumPartitions=feaParts,
    sc: SparkContext)(dmatFloat)

  // score sq using dmatDouble
  val multiScores =  for {
    ex <- exSq
    feas = ex.features
    sfeas = new SV[Float](feas.map(_._1.toInt),
      feas.map(_._2), denseSize)
    
    id = ex.exid.toLong
    target = dmatFloat * sfeas
  } yield (id, target)

  val msMp2 = multiScores.toMap

  // distribute weight matrix 
  val linMt = RepSDDV.apply(ddmatFloat, feas1.self.partitioner.get.
    asInstanceOf[ProductPartitioner].leftPartitioner)
  linMt.self.setName("MatrixWeights")

  val scores2 = LinearScorer.apply(feas1, linMt)
  val tsMp2 = scores2.self.collect.toMap

  testLogger.info(s"scoring vector consistency: ${checkVecFloat(msMp2, tsMp2)}")

  // this test checks the correctness of the factorization machine scoring
  // score sq using dmatDouble

  /* score the factorization machine on one example
   *  the stupid way
   */
  def fmLocalScore(m: DM[Float], v: DV[Float]) = {
    val cols = m.cols
    var target = 0f

    for {
      alpha <- 0 until m.cols
      beta <- (alpha+1) until m.cols
      add_ = m(::, alpha).dot(m(::, beta))
      mul_ = add_ * v(alpha) * v(beta)
      _ = (target += mul_)
    } yield ()

    var target_vec = DV.zeros[Float](m.rows)

    for {
      alpha <- 0 until m.cols
      _ = (target_vec+= m(::, alpha) * v(alpha))
    } yield ()

    (target, target_vec)
  }

  // score the examples
  val fmScores =  for {
    ex <- exSq
    feas = ex.features
    sfeas = new SV[Float](feas.map(_._1.toInt),
      feas.map(_._2), denseSize).toDenseVector
    
    id = ex.exid.toLong
    fmScores = fmLocalScore(dmatFloat, sfeas)

  } yield (id, fmScores)

  val dfmScores = FactorizationMachine.apply(feas1, linMt)
  val tsMp3 = dfmScores.self.collect.toMap

  val scDiff = checkFMFloat(fmScores.toMap, tsMp3)

  /* the score discrepancy we take into account rounding errors
   * divide by average score
   */
  val avScDf = scDiff/(fmScores.map(_._2._1).foldLeft(0f)(_ + _)
    / fmScores.size)

  testLogger.info(s"factorization machine consistency: ${avScDf}")
}
