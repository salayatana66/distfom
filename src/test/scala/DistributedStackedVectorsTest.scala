/**********************************************************************************************
 *  Test for implementation of DistributedStackedDenseVectors
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/DistributedStackedVectorsTest.scala", $intp)
 **********************************************************************************************/

import distfom.{DistributedStackedDenseVectors => DSV}

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseVector => DV, DenseMatrix => DM, norm}
import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

object DistributedStackedDenseVectorTest {

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
   * a must be have at least as many columns as b
   */
  def consCheckFloat(a: DM[Float], b: DM[Float]) = {
    val aFlat = a.flatten()
    val bFlat = b.flatten()

    norm(aFlat.slice(0, bFlat.size) - bFlat) +
    norm(aFlat.slice(bFlat.size, aFlat.size))

  }

  def consCheckDouble(a: DM[Double], b: DM[Double]) = {
    val aFlat = a.flatten()
    val bFlat = b.flatten()

    norm(aFlat.slice(0, bFlat.size) - bFlat) +
    norm(aFlat.slice(bFlat.size, aFlat.size))

  }

  def convertMatToFloat(a: DM[Double]): DM[Float] = {
    new DM[Float](rows=a.rows, cols=a.cols,
    data=a.data.map(_.toFloat), offset=0,
    majorStride=a.majorStride, isTranspose=a.isTranspose)   
  }

  /* The first test is that if we construct a DSV from a DM
   * when we recover it on the driver they agree up to a padding by zeros;
   * for this test it is good not to have a number of elements per block
   * that exactly divides the number of columns
   */
  val denseSize = 10001
  val eb = 217
  val rowDim = 5

  // initialize random matrices
  val matDouble = DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform)
  val matFloat = convertMatToFloat(matDouble)

  val dmatDouble = DSV.fromBreezeDense(matDouble, elementsPerBlock=eb, sc)
  val dmatFloat = DSV.fromBreezeDense(matFloat, elementsPerBlock=eb, sc)

  // set names on the driver
  dmatDouble.self.setName("dmatDouble")
  dmatFloat.self.setName("dmatFloat")

  testLogger.info(s"""Number of partitions for DSV : ${dmatDouble.
    self.partitioner.get.numPartitions}""")

  val bdmatDouble = DSV.toBreeze(dmatDouble)
  val bdmatFloat = DSV.toBreeze(dmatFloat)

  testLogger.info(s"""consistency of from/to Breeze(Double): 
                  ${consCheckDouble(bdmatDouble, matDouble)}""")

  testLogger.info(s"""consistency of from/to Breeze(Double):
                  ${consCheckFloat(bdmatFloat, matFloat)}""")


  /* The second test is like the first but uses the transposed
   * matrix
   */

  // transpose preserving the shape
  val matDoubleT = new DM[Double](rows=matDouble.rows,
    cols=matDouble.cols, data=matDouble.data,
    offset=0, majorStride=matDouble.cols,
    isTranspose=true)
  val matFloatT = new DM[Float](rows=matFloat.rows,
    cols=matFloat.cols, data=matFloat.data,
    offset=0, majorStride=matFloat.cols,
    isTranspose=true)

  val dmatDoubleT = DSV.fromBreezeDense(matDoubleT, elementsPerBlock=eb, sc)
  val dmatFloatT = DSV.fromBreezeDense(matFloatT, elementsPerBlock=eb, sc)

  // set names on the driver
  dmatDoubleT.self.setName("dmatDoubleT")
  dmatFloatT.self.setName("dmatFloatT")

  testLogger.info(s"""Number of partitions for DSV : ${dmatDoubleT.
    self.partitioner.get.numPartitions}""")

  val bdmatDoubleT = DSV.toBreeze(dmatDoubleT)
  val bdmatFloatT = DSV.toBreeze(dmatFloatT)

  testLogger.info(s"""consistency of from/to Breeze(Double): 
                  ${consCheckDouble(bdmatDoubleT, matDoubleT)}""")

  testLogger.info(s"""consistency of from/to Breeze(Double):
                  ${consCheckFloat(bdmatFloatT, matFloatT)}""")

  /* This test checks agreement on an arithmetic operation;
   * it is important to check in the SparkUI that the DAG does not lead
   * to a repartitioning; all operations are distributed should match on the same
   * number of partitions
   */
  val matListDouble = List.range(1,4).
    map(x => DM.rand(rows=rowDim, cols=denseSize, rand = Rand.uniform))
  val matListFloat = matListDouble.map(convertMatToFloat)


  val dmatListDouble = matListDouble.zipWithIndex.map(x => {
    val out = DSV.fromBreezeDense(x._1, elementsPerBlock = eb,
      sc)
    out.self.setName(s"dmatListDouble${x._2}")
    out
  })

  val dmatListFloat = matListFloat.zipWithIndex.map(x => {
    val out = DSV.fromBreezeDense(x._1, elementsPerBlock = eb,
      sc)
    out.self.setName(s"dmatListFloat${x._2}")
    out
  })

    // Execute an operation
  val resDouble =( matListDouble(0) *:* matListDouble(1) - matListDouble(2) ) +
                   matListDouble(2) *:* matListDouble(1)

  val resFloat =( matListFloat(0) *:* matListFloat(1) - matListFloat(2) ) +
                   matListFloat(2) *:*  matListFloat(1)

  val dresDouble =( dmatListDouble(0) * dmatListDouble(1) - dmatListDouble(2) ) +
                   dmatListDouble(2) * dmatListDouble(1)

  val dresFloat =( dmatListFloat(0) * dmatListFloat(1) - dmatListFloat(2) ) +
                   dmatListFloat(2) * dmatListFloat(1)

  // pull back results on driver
  val bresDouble = DSV.toBreeze(dresDouble)
  val bresFloat = DSV.toBreeze(dresFloat)

  testLogger.info(s"""consistency test on vector operations (Double): 
                   ${consCheckDouble(bresDouble,resDouble)}""")
  testLogger.info(s"""consistency test on vector operations (Float): 
                   ${consCheckFloat(bresFloat,resFloat)}""")
  

  /* This test checks the fromRDD procedure
   * it creates an Array(Long, Random DV) to construct the vector and the rdd version
   */
  val rddSize = 100000L
  // elements per block
  val rddElPD = 2000
  val rowDimRdd = 7

  val rVecDoubles = for {
    j <- 0 until rddSize.toInt
  } yield (j.toLong, DV.rand(size = rowDimRdd.toInt, rand = Rand.uniform))

  /* stack them togethe
   * somehow here horzcat does not work :(
   */
  val rMatDoubles = new DM[Double](rows=rowDimRdd, cols=rddSize.toInt,
    data=rVecDoubles.map(_._2).flatMap(_.data).toArray,
    offset=0, majorStride=rowDimRdd, isTranspose=false)

  val rVecFloats = rVecDoubles.map( x => (x._1, new DV[Float](
    x._2.data.map(_.toFloat), 0, 1, x._2.data.size)))

  val rMatFloats = convertMatToFloat(rMatDoubles)

  val rddDoubles = sc.parallelize(rVecDoubles, numSlices=20)
  val rddFloats = sc.parallelize(rVecFloats, numSlices=20)

  val dmatDoubles = DSV.fromRDD(elements = rddSize, elementsPerBlock = rddElPD, rowDim=rowDimRdd,
    rdd = rddDoubles)
  val dmatFloats = DSV.fromRDD(elements = rddSize, elementsPerBlock = rddElPD, rowDim=rowDimRdd,
    rdd = rddFloats)

  dmatDoubles.self.setName("dmatDoubles")
  dmatFloats.self.setName("dmatFloats")

  val bdmatDoubles = DSV.toBreeze(dmatDoubles)
  val bdmatFloats =  DSV.toBreeze(dmatFloats)

  testLogger.info(s"""consistency for fromRDD (Double): 
                   ${consCheckDouble(bdmatDoubles, rMatDoubles)}""")

  testLogger.info(s"""consistency for fromRDD (Float):
                   ${consCheckFloat(bdmatFloats, rMatFloats)}""")


}
