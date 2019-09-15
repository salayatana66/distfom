/**********************************************************************************************
 *  Tests concerning the implementation of DistributedDenseVectors
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/DistributedDenseVectorTest.scala", $intp)
 *********************************************************************************************/

import distfom.{DistributedDenseVector => DDV}

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseVector => DV, norm}
import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

object DistributedDenseVectorTest {

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

  /* consistency check: equality up to padding
   * a must be at least as long as b
   */
  def consCheckFloat(a: DV[Float], b: DV[Float]) = norm(a.slice(0, b.size) - b) +
  norm(a.slice(b.size, a.size))
  def consCheckDouble(a: DV[Double], b: DV[Double]) = norm(a.slice(0, b.size) - b) +
  norm(a.slice(b.size, a.size))

  /* The first test is that if we construct a DDV from a DV
   * when we recover it on the driver they agree up to a padding by zeros;
   * for this test it is good not to have a number of elements per block
   * that exactly divides the vector size
   */
  val denseSize = 10001
  val eb = 217

  // initialize random vectors
  val dvecDouble = DV.rand(size = denseSize, rand = Rand.uniform)
  val dvecFloat = new DV[Float](dvecDouble.data.map(_.toFloat),
    0, 1, dvecDouble.size)

  val ddvecDouble = DDV.fromBreezeDense(dvecDouble, elementsPerBlock = eb,
    sc)
  val ddvecFloat = DDV.fromBreezeDense(dvecFloat, elementsPerBlock = eb,
    sc)

  // set names on the driver
  ddvecDouble.self.setName("ddvecDouble")
  ddvecFloat.self.setName("ddvecFloat")

  testLogger.info(s"""Number of partitions for DDV : ${ddvecDouble.
    self.partitioner.get.numPartitions}""")

  val bdvecDouble = DDV.toBreeze(ddvecDouble)
  val bdvecFloat = DDV.toBreeze(ddvecFloat)

  testLogger.info(s"consistency of from/to Breeze(Double): ${consCheckDouble(bdvecDouble,dvecDouble)}")
  testLogger.info(s"consistency of from/to Breeze(Float): ${consCheckFloat(bdvecFloat,dvecFloat)}")

  /* This test checks agreement on an arithmetic operation;
   * it is important to check in the SparkUI that the DAG does not lead
   * to a repartitioning; all operations are distributed should match on the same
   * number of partitions
   */
  val dvecDoubleList = List.range(1,4).
    map(x => DV.rand(size = denseSize, rand = Rand.uniform))
  
  val dvecFloatList = dvecDoubleList.map(x =>
    new DV[Float](x.data.map(_.toFloat), 0, 1, x.size))

  val ddvecDoubleList = dvecDoubleList.zipWithIndex.map(x => {
    val out = DDV.fromBreezeDense(x._1, elementsPerBlock = eb,
      sc)
    out.self.setName(s"ddvecList${x._2}")
    out
  })

  val ddvecFloatList = dvecFloatList.zipWithIndex.map(x => {
    val out = DDV.fromBreezeDense(x._1, elementsPerBlock = eb,
      sc)
    out.self.setName(s"ddvecList${x._2}")
    out
  })

  // Execute an operation
  val resDouble =( dvecDoubleList(0)*dvecDoubleList(1) - dvecDoubleList(2) ) + dvecDoubleList(2) *
  dvecDoubleList(1)

  val resFloat =( dvecFloatList(0)*dvecFloatList(1) - dvecFloatList(2) ) + dvecFloatList(2) *
  dvecFloatList(1)

  val dresDouble = ( ddvecDoubleList(0)*ddvecDoubleList(1) - ddvecDoubleList(2) ) + ddvecDoubleList(2) *
  ddvecDoubleList(1)

  val dresFloat = ( ddvecFloatList(0)*ddvecFloatList(1) - ddvecFloatList(2) ) + ddvecFloatList(2) *
  ddvecFloatList(1)

  // pull back results on driver
  val bresDouble = DDV.toBreeze(dresDouble)
  val bresFloat = DDV.toBreeze(dresFloat)

  testLogger.info(s"consistency test on vector operations (Double): ${consCheckDouble(bresDouble,resDouble)}")
  testLogger.info(s"consistency test on vector operations (Float): ${consCheckFloat(bresFloat,resFloat)}")
  
  /* This test checks the fromRDD procedure
   * it creates an Array(Long, Random) to construct the vector and the rdd version
   */
  val rddSize = 1000000L
  // elements per block
  val rddElPD = 2000

  val randomDoubles = DV.rand(size = rddSize.toInt, rand = Rand.uniform)
  val randomFloats = new DV[Float](randomDoubles.data.map(
    x => x.toFloat), 0, 1, randomDoubles.data.size)

  val rddDoubles = sc.parallelize(randomDoubles.data.zipWithIndex.
    map(x => (x._2.toLong, x._1)), numSlices=20)
  val rddFloats = sc.parallelize(randomFloats.data.zipWithIndex.
    map(x => (x._2.toLong, x._1)), numSlices=20)

  val drDoubles = DDV.fromRDD(elements = rddSize, elementsPerBlock = rddElPD, rdd = rddDoubles)
  val drFloats = DDV.fromRDD(elements = rddSize, elementsPerBlock = rddElPD, rdd = rddFloats)

  drDoubles.self.setName("drDoubles")
  drFloats.self.setName("drFloats")

  val brDoubles = DDV.toBreeze(drDoubles)
  val brFloats =  DDV.toBreeze(drFloats)

  testLogger.info(s"consistency for fromRDD (Double): ${consCheckDouble(brDoubles, randomDoubles)}")
  testLogger.info(s"consistencyFloat for fromRDD (Float): ${consCheckFloat(brFloats, randomFloats)}")
  
}
