/*************************************************************************************************
 *  The Scalability Experiment 3 in the paper
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/ScalabilityExperiment_3.scala", $intp)
 ************************************************************************************************/

import distfom.{DualSpace, DistributedDenseVector => DDV,
  FomDistDenseVec => FDV, LBFGS, StochasticGradientDescent => SGD, BlockPartitioner}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV,
norm => bNorm}
import breeze.stats.distributions.Rand
//import distfom.MultiLabelSoftMaxLogLoss.MultiLabelTarget

// logger
import org.apache.log4j.{Logger, Level}

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem,Path}

import org.apache.spark.mllib.random.RandomRDDs._

object ScalabilityExperiment_3CostPreparation {

  /* the SparkSession is created with a boilerplate; in reality
  * this is just supplied to prevent the IDE from flagging errors, and
  * the tests can be run in cluster/client mode (on YARN or MESOS)
  */
  val spark = SparkSession.
    builder().
    appName("DistributedDenseVector").
    getOrCreate()

  val sc = spark.sparkContext

  sc.setLogLevel("FATAL")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/ScalabilityExp3")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExp3")

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment3"

  /* sampler of points uniformly in dim dimensional space
     from the unit ball
  */

  val unitBallSampler: Int => () => DV[Double] =
    (dim: Int) => () => {
    // sample a vector uniformly on the cube centered at the origin
    val uCube = DV.rand(dim, Rand.uniform).map(
      x=> -1.0 + 2.0 * x
    )
    // project on the unit sphere
    val uCubeNorm = bNorm(uCube)
    val homSphere = uCube.map(x => x / (if(uCubeNorm > 1e-16)
      uCubeNorm else 1e-16))

    // sample radius and distort it
    val radRaw = DV.rand(1, Rand.uniform).data(0)
    // distortion to sample uniformly in the unit ball
    val rad = math.pow(radRaw, 1.0/dim)

    val outVec = homSphere.map(x => x * rad)

    outVec
  }

  /*
   * Sampler specifying center and radius
   */
  val genericBallSampler: (Int, DV[Double], Double) => () => DV[Double] =
    (dim: Int, center: DV[Double], rad: Double) => {
    // quality checks
    require(dim == center.length, s"center must belong to R^$dim")
    require(rad > 0, "radius must be positive")

    // instantiate a unitBallSampler
    val locSampler = unitBallSampler(dim)

    () => {
      val svec = locSampler()
      svec.map(x => x * rad) + center
    }
  }

  // X space dimension
  val Ex = 2e5.toInt
  // Y space dimension
  val Ey = 2e5.toInt
  // number of y Balls (will get multiplied by 2 to get symmetry)
  val Ny = 20
  // elements per block
  val eb = 1e3.toInt
  // dimension space of X & Y
  val dimS = 55
  // partitions to break things down during sampling
  val numPart = 50

  // construct list of ySamplers for Random Centers
  import scala.collection.mutable.ArrayBuffer
  val rawYCentersBuffer = ArrayBuffer[DV[Double]]()
  @transient val dimSampler = Rand.randInt(0, Ny)

  for {
    j <- 0 until Ny
    dim = dimSampler.sample

    // zero vectors
    q_+ = DV.zeros[Double](dimS)
    q_- = DV.zeros[Double](dimS)

    // add the +- .5 at the sampled dimension
    _ = (q_+(dim) = .5)
    _ = (q_-(dim) = -.5)

    // append to the Buffer
    _ = (rawYCentersBuffer.append(q_+))
    _ = (rawYCentersBuffer.append(q_-))
  } yield ()

  val rawYCenters = rawYCentersBuffer.toList

  // sample X
  val xRaw = uniformRDD(sc, size = Ex, numPartitions = numPart).
    zipWithIndex
  val xPoints = xRaw.mapPartitions((iter: Iterator[(Double, Long)])=> {

    val pairList = iter.toList
    val sampler = unitBallSampler(dimS)

    pairList.map(x => (x._2, sampler())).
      toIterator
  }
  )

  xPoints.setName("xPoints")
  xPoints.persist.count

  // sample Y
  val yRaw = uniformRDD(sc, size = Ey, numPartitions = numPart).
    zipWithIndex

  val yPoints = yRaw.mapPartitions((iter: Iterator[(Double, Long)])=> {

    val pairList = iter.toList
    val dimSampler = Rand.randInt(0, rawYCenters.size)

    val ballSamplers = rawYCenters.map(x => genericBallSampler(dimS, x, .5))

    val outYs = for {
      pair <- pairList
      (_, idx) = pair
      dim = dimSampler.sample
      point = ballSamplers(dim)()
    } yield((idx, point))

    outYs.toIterator
  })

  yPoints.setName("yPoints")
  yPoints.persist.count

  // cost Function
  val costFun = (a: DV[Double], b: DV[Double]) => 0.5 * ((a-b).dot(a-b))

  // construct a pairing context
  val ctx = DualSpace.PairingContext(Ex, Ey, eb)
  val costs = DualSpace.computePairwiseCosts(xPoints, yPoints, ctx, costFun)
  costs.setName("costs_large")

  costs.persist.count
  // costs 	Memory Deserialized 1x Replicated 	8599 	86% 	712.1 GB 	0.0 B

  costs.repartition(1200)
    .saveAsObjectFile(s"$rddSaveDir/costs_large")

  hfs.delete(chkPath)

}

object ScalabilityExperiment_3Fit {
  /* the SparkSession is created with a boilerplate; in reality
  * this is just supplied to prevent the IDE from flagging errors, and
  * the tests can be run in cluster/client mode (on YARN or MESOS)
  */
  val spark = SparkSession.
    builder().
    appName("DistributedDenseVector").
    getOrCreate()

  val sc = spark.sparkContext

  sc.setLogLevel("FATAL")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/ScalabilityExp3")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExp3")

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment3"

  /* sampler of points uniformly in dim dimensional space
     from the unit ball
  */

  // X space dimension
  val Ex = 2e5.toInt
  // Y space dimension
  val Ey = 2e5.toInt
  // number of y Balls (will get multiplied by 2 to get symmetry)
  val Ny = 20
  // elements per block
  val eb = 1e3.toInt
  // dimension space of X & Y
  val dimS = 55
  // partitions to break things down during sampling
  val numPart = 50
  // partitions to write for intermediate state
  val wgtWritePart = 100

  // construct a pairing context
  val ctx = DualSpace.PairingContext(Ex, Ey, eb)

  val costs = sc.objectFile(path =s"$rddSaveDir/costs_large").
    asInstanceOf[RDD[((Int,Int), DualSpace.LocalCost[Double])]].
    partitionBy(ctx.pairingPartitioner)

  costs.setName("costs")

  costs.persist.count

  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // starting distributed potential
  val localPot = DV.zeros[Double](size=Ex+Ey)
  val startPot = new FDV(DDV.fromBreezeDense(localPot, elementsPerBlock=eb, sc))

  val sinkReg = 1e-1
  val transportLoss = DualSpace.potentialLossWSinkhornRegularization(ctx, sinkReg, costs, 2)

  // optimizer
  import distfom.BackTrackingLineSearch
  val lsearch = new BackTrackingLineSearch[Double](
    enforceStrongWolfeConditions = false, enforceWolfeConditions = false)
  val LBFGSMin = new LBFGS[FDV[Double], Double](m = 5,
    maxIter=2, tolerance=1e-4, interruptLinSteps = 1,
    baseLSearch = 1e2, suppliedLineSearch = Some(lsearch))

  // implicit conversions for differentiable functions
  import FDV.implicits._

  val finalState = LBFGSMin.minimizeAndReturnState(f = transportLoss, init = startPot)

  // save to initialize again from it
  finalState.x.ddvec.self.repartition(wgtWritePart)
    .saveAsObjectFile(s"$rddSaveDir/fstate1")

  hfs.delete(chkPath)

}
