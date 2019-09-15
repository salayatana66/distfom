/**********************************************************************************************
 *  Test for implementation of FomDistStackedDenseVec
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/FomDistStackedDenseVecTest.scala", $intp)
 **********************************************************************************************/

import distfom.{DistributedStackedDenseVectors => DSV,
  FomDistStackedDenseVec => FSV}

import FSV.implicits._

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseVector => DV, DenseMatrix => DM, norm}

import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

import scala.util.Random

object FomDistStackedDenseVecTest {

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

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  // Import the smooth implicits
  import distfom.CommonImplicitGenericSmoothFields._

  /* consistency check: equality up to padding
   * a must be have at least as many columns as b
   */

  def consCheckDouble(a: DM[Double], b: FSV[Double]) = {
    val aFlat = a.flatten()
    val bFlat = DSV.toBreeze(b.dsvec).flatten()

    norm(bFlat.slice(0, aFlat.size) - aFlat) +
    norm(bFlat.slice(aFlat.size, bFlat.size))

  }

  def consCheckFloat(a: DM[Float], b: FSV[Float]) = {
    val aFlat = a.flatten()
    val bFlat = DSV.toBreeze(b.dsvec).flatten()

    norm(bFlat.slice(0, aFlat.size) - aFlat) +
    norm(bFlat.slice(aFlat.size, bFlat.size))

  }


  def convertMatToFloat(a: DM[Double]): DM[Float] = {
    new DM[Float](rows=a.rows, cols=a.cols,
    data=a.data.map(_.toFloat), offset=0,
    majorStride=a.majorStride, isTranspose=a.isTranspose)   
  }

  // vector size & number of elements per block
  val denseSize = 10000
  val eb = 217
  val trDim = 8

  // create 3 matrices
  val amD = DM.rand(rows=trDim, cols=denseSize, rand = Rand.uniform)
  val amF = convertMatToFloat(amD)
  val bmD = DM.rand(rows=trDim, cols=denseSize, rand = Rand.uniform)
  val bmF = convertMatToFloat(bmD)
  val cmD = DM.rand(rows=trDim, cols=denseSize, rand = Rand.uniform)
  val cmF = convertMatToFloat(cmD)

  // distribute them
  val afmD = new FSV(
    DSV.fromBreezeDense(amD, elementsPerBlock = eb, sc),
    trDim)

  val afmF = new FSV(
    DSV.fromBreezeDense(amF, elementsPerBlock = eb, sc),
    trDim)

  val bfmD = new FSV(
    DSV.fromBreezeDense(bmD, elementsPerBlock = eb, sc),
    trDim)

  val bfmF = new FSV(
    DSV.fromBreezeDense(bmF, elementsPerBlock = eb, sc),
    trDim)

  val cfmD = new FSV(
    DSV.fromBreezeDense(cmD, elementsPerBlock = eb, sc),
    trDim)

  val cfmF = new FSV(
    DSV.fromBreezeDense(cmF, elementsPerBlock = eb, sc),
    trDim)

    // set names and persists
  afmD.setName("afmD")
  afmF.setName("afmF")
  afmD.persist.count
  afmF.persist.count
  bfmD.setName("bfmD")
  bfmF.setName("bfmF")
  bfmD.persist.count
  bfmF.persist.count
  cfmD.setName("cfmD")
  cfmF.setName("cfmF")
  cfmD.persist.count
  cfmF.persist.count

  // vector operation's test
  val opmD = ((amD *:* bmD) - ((cmD *:* 0.312 + amD) + bmD) *:* cmD)
  val opmF = ((amF *:* bmF) - ((cmF *:* 0.312f + amF) + bmF) *:* cmF)
  val opfmD = ((afmD * bfmD) - ((cfmD * 0.312 + afmD) + bfmD) * cfmD)
  val opfmF = ((afmF * bfmF) - ((cfmF * 0.312f + afmF) + bfmF) * cfmF)
  
  testLogger.info(s"""Consistency on vector operation (Double):
                  ${consCheckDouble(opmD, opfmD)}
                  """)

  testLogger.info(s"""Consistency on vector operation (Float):
                  ${consCheckFloat(opmF, opfmF)}
                  """)

  // dot operation's test
  val dotvD = amD.flatten().dot(bmD.flatten())
  val dotvF = amF.flatten().dot(bmF.flatten())
  val dotmD = afmD.dot(bfmD)
  val dotmF = afmF.dot(bfmF)

  testLogger.info(s"Consistency on dot (Double): ${dotvD-dotmD}")
  testLogger.info(s"Consistency on dot (Float): ${dotvF-dotmF}")

    // test on the l2 norm
  val n2vD = norm(cmD.flatten())
  val n2vF = norm(cmF.flatten())
  val n2mD = cfmD.norm()
  val n2mF = cfmF.norm()

  testLogger.info(s"Consistency on l2 norm (Double): ${n2vD-n2mD}")
  testLogger.info(s"Consistency on l2 norm (Float): ${n2vD-n2mF}")

  // test on the l1 norm
  val n1vD = cmD.data.map(math.abs(_)).foldLeft(0.0)(_ + _)
  val n1vF = cmF.data.map(math.abs(_)).foldLeft(0.0)(_ + _)
  val n1mD = cfmD.norm(1.0)
  val n1mF = cfmF.norm(1.0)

  testLogger.info(s"Consistency on l1 norm (Double): ${n1vD-n1mD}")
  testLogger.info(s"Consistency on l1 norm (Float): ${n1vD-n1mF}")
  
}
