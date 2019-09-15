/************************************************************************************************
 *  Test for implementation of FomDistDenseVec
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/FomDistDenseVecTest.scala", $intp)
 ***********************************************************************************************/

import distfom.{DistributedDenseVector, FomDistDenseVec}

import FomDistDenseVec.implicits._

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseVector => DV, norm}

import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

import scala.util.Random

object FomDistDenseVecTest {

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


  // vector size & number of elements per block
  val denseSize = 10000
  val eb = 217

  // utility to build a DV[Float] out of a DV[Double]
  def getDVFloat(a: DV[Double]): DV[Float] = {
    new DV[Float](a.data.map(_.toFloat), 0, 1, a.size)
  }

  // create 3 vectors
  val avD = DV.rand(size = denseSize, rand = Rand.uniform)
  val avF = getDVFloat(avD)
  val bvD = DV.rand(size = denseSize, rand = Rand.uniform)
  val bvF = getDVFloat(bvD)
  val cvD = DV.rand(size = denseSize, rand = Rand.uniform)
  val cvF = getDVFloat(cvD)

  // distribute them
  val afmD = new FomDistDenseVec(
    DistributedDenseVector.fromBreezeDense(avD, elementsPerBlock = eb,
      sc)
  )
  val afmF = new FomDistDenseVec(
    DistributedDenseVector.fromBreezeDense(avF, elementsPerBlock = eb,
      sc)
  )


    val bfmD = new FomDistDenseVec(
    DistributedDenseVector.fromBreezeDense(bvD, elementsPerBlock = eb,
      sc)
  )
  val bfmF = new FomDistDenseVec(
    DistributedDenseVector.fromBreezeDense(bvF, elementsPerBlock = eb,
      sc)
  )

    val cfmD = new FomDistDenseVec(
    DistributedDenseVector.fromBreezeDense(cvD, elementsPerBlock = eb,
      sc)
  )
  val cfmF = new FomDistDenseVec(
    DistributedDenseVector.fromBreezeDense(cvF, elementsPerBlock = eb,
      sc)
  )

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


  // consistency check up to padding
  def consistencyCheckDouble(loc: DV[Double],
    dist: FomDistDenseVec[Double]) = {
    val pullLocal = DistributedDenseVector.toBreeze(dist.ddvec)

    /* we need to compare norms modulo the zero padding that
     * can take place in distributing
     */
    val onsize = norm(pullLocal.slice(0, loc.size) - loc)
    val offsize = norm(pullLocal.slice(loc.size, pullLocal.size))

    onsize + offsize
  }

  def consistencyCheckFloat(loc: DV[Float],
    dist: FomDistDenseVec[Float]) = {
    val pullLocal = DistributedDenseVector.toBreeze(dist.ddvec)

    /* we need to compare norms modulo the zero padding that
     * can take place in distributing
     */
    val onsize = norm(pullLocal.slice(0, loc.size) - loc)
    val offsize = norm(pullLocal.slice(loc.size, pullLocal.size))

    onsize + offsize
  }


  // vector operation's test
  val opvD = ((avD * bvD) - ((cvD * 0.312 + avD) + bvD) * cvD)
  val opvF = ((avF * bvF) - ((cvF * 0.312f + avF) + bvF) * cvF)
  val opfmD = ((afmD * bfmD) - ((cfmD * 0.312 + afmD) + bfmD) * cfmD)
  val opfmF = ((afmF * bfmF) - ((cfmF * 0.312f + afmF) + bfmF) * cfmF)
  
  testLogger.info(s"""Consistency on vector operation (Double):
${consistencyCheckDouble(loc = opvD, dist = opfmD)}
""")

  testLogger.info(s"""Consistency on vector operation (Float):
${consistencyCheckFloat(loc = opvF, dist = opfmF)}
""")


  // dot operation's test
  val dotvD = avD.dot(bvD)
  val dotvF = avF.dot(bvF)
  val dotmD = afmD.dot(bfmD)
  val dotmF = afmF.dot(bfmF)

  testLogger.info(s"Consistency on dot (Double): ${dotvD-dotmD}")
  testLogger.info(s"Consistency on dot (Float): ${dotvF-dotmF}")

  // test on the l2 norm
  val n2vD = norm(cvD)
  val n2vF = norm(cvF)
  val n2mD = cfmD.norm()
  val n2mF = cfmF.norm()

  testLogger.info(s"Consistency on l2 norm (Double): ${n2vD-n2mD}")
  testLogger.info(s"Consistency on l2 norm (Float): ${n2vD-n2mF}")

  // test on the l1 norm
  val n1vD = cvD.data.map(math.abs(_)).foldLeft(0.0)(_ + _)
  val n1vF = cvF.data.map(math.abs(_)).foldLeft(0.0)(_ + _)
  val n1mD = cfmD.norm(1.0)
  val n1mF = cfmF.norm(1.0)

  testLogger.info(s"Consistency on l1 norm (Double): ${n1vD-n1mD}")
  testLogger.info(s"Consistency on l1 norm (Float): ${n1vD-n1mF}")

}
