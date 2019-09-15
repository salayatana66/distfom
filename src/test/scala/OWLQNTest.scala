/*************************************************************************************************
 *  Test for implementation of OWLQN
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/OWLQNTest.scala", $intp)
 ************************************************************************************************/

import distfom.{DistributedDenseVector => DDV,
  FomDistDenseVec => FV, OWLQN, Example, ArrayBatcher,
Models, DistributedDiffFunction}

import FV.implicits._

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseVector => DV, norm,
  SparseVector => SV}

import breeze.stats.distributions.Rand

import scala.math

import scala.collection.mutable.ListBuffer
// logger
import org.apache.log4j.{Logger,Level}

import scala.util.Random

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem,Path}

object OWLQNTest {

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

  val chkPath = new Path("/tmp/OWLQNTest")
  hfs.mkdirs(chkPath)

  sc.setCheckpointDir("/tmp/OWLQNTest")

  val testLogger = Logger.getLogger("OWLQNTest")
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
  val fvecDoubleStart = new FV[Double](ddvecDoubleStart, 2)

  val l2loss = Models.linearL2Loss(batcher)

  val l1reg1 = (i: Int) => if(i < 100) 1e-4 else 0.0

  // optimizer 
  val OMin = new OWLQN[Int, FV[Double], Double](m = 3, l1reg = l1reg1,
  maxIter = 10)

  // implicit conversions
  import FV.implicits._

  val finalWeight = OMin.minimize(f = l2loss, init = fvecDoubleStart)

  hfs.delete(chkPath)

}
