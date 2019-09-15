/************************************************************************************************
*  Test for implementation of the Optimal Transport
 *  Loss
  *  When running with interactively with
*  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
*
*  import auto.paste.from.script._
*  val myS = AutoPasteFromScript("src/test/scala/OptimalTransportTest.scala", $intp)
************************************************************************************************/

import distfom.{DualSpace, DistributedDenseVector => DDV,
FomDistDenseVec => FDV}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import breeze.linalg.{norm, DenseMatrix => DM, DenseVector => DV, SparseVector => SV}
import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem,Path}

object OptimalTransportCostComputationTest {

  /* the SparkSession is created with a boilerplate; in reality
     * this is just supplied to prevent the IDE from flagging errors, and
     * the tests can be run in cluster/client mode (on YARN or MESOS)
     */
  val spark = SparkSession.
    builder().
    master("local").
    appName("DistributedDenseVector").
    getOrCreate()

  @transient val sc = spark.sparkContext

  sc.setLogLevel("FATAL")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  // X space dimension
  val Ex = 15
  // Y space dimension
  val Ey = 10
  // elements per block
  val eb = 5
  // dimension space of X & Y
  val dimS = 2

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

  // compute the layout of the costs on the driver
  val costLayOutDriver = for {
    xP <- xPoints
    yP <- yPoints
    (ix, vx) = xP
    (jy, vy) = yP
    cost = costFun(vx, vy)
  } yield (((ix / eb).toInt, (jy /eb).toInt), DualSpace.LocalCost((ix % eb).toInt,
    (jy % eb).toInt, cost))

  // parallelize
  val rXPoints = sc.parallelize(xPoints, 3)
  val rYPoints = sc.parallelize(yPoints, 3)

  val costs = DualSpace.computePairwiseCosts(rXPoints, rYPoints, ctx, costFun)

  // pull back costs to Driver
  val backCosts = costs.collect

  // unfold to a Map of Map to compare local & distributed computation
  val bCMap = backCosts.groupBy(_._1).mapValues(_.map(_._2).
    map(x => ((x.i, x.j), x.cost)).toMap)

  var discrepancy = 0.0
  for {
    elem <- costLayOutDriver
    (partKey, localCost) = elem
    lkMap = bCMap(partKey)
    backCost = lkMap(localCost.i -> localCost.j)
    _ = (discrepancy += math.abs(backCost - localCost.cost))
  } yield ()

  testLogger.info(s"Total discrepancy between computation on driver & distributed computation: $discrepancy")

}

object OptimalTransportRegFunCompTest {

  /* the SparkSession is created with a boilerplate; in reality
     * this is just supplied to prevent the IDE from flagging errors, and
     * the tests can be run in cluster/client mode (on YARN or MESOS)
     */
  val spark = SparkSession.
    builder().
    master("local").
    appName("DistributedDenseVector").
    getOrCreate()

  @transient val sc = spark.sparkContext

  sc.setLogLevel("FATAL")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  // X space dimension
  val Ex = 15
  // Y space dimension
  val Ey = 10
  // elements per block
  val eb = 5
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

  // compute the costs on the driver
  val costDriver = for {
    xP <- xPoints
    yP <- yPoints
    (ix, vx) = xP
    (jy, vy) = yP
    cost = costFun(vx, vy)
  } yield ((ix.toInt, jy.toInt), cost)

  // sinkhorn regularization
  val sinkReg = .1

  import breeze.optimize.DiffFunction
  import breeze.linalg.{sum => bSum, norm => bNorm}
  val localLoss = new DiffFunction[DV[Double]] {

    override def calculate(pot: DV[Double]) : (Double, DV[Double])= {
      require(pot.length == Ex + Ey, "potential length must equal Ex + Ey")

      var regLoss = 0.0
      val regGrad = DV.zeros[Double](pot.length)

      // loop over the cost to produce value & gradient for the regularization term
      val toReduce = for {
        costElem <- costDriver
        ((ix, jy), cost) = costElem

        // constraint violation
        consViolation = (pot(ix) + pot(jy+Ex) - cost) / sinkReg
        eViolation = math.exp(consViolation)

        value = -sinkReg * eViolation

        // gradient wrt to part "u" of the potential
        gradU = -eViolation
        // gradient wrt to part "v" of the potential
        gradV = -eViolation

        _ = (regLoss += value)
        _ = (regGrad(ix) += gradU)
        _ = (regGrad(jy + Ex) += gradV)
      } yield ()

      /* the total loss is - (sum of potential_u /Ex +
       + sum of potential_v / Ey +
       regLoss / (Ex*Ey))
       */
      val upot = pot.slice(0, Ex)
      val vpot = pot.slice(Ex, Ex+Ey)
      val totalLoss = -(bSum(upot) / Ex + bSum(vpot) / Ey + regLoss / (Ex * Ey))
      //val totalLoss = (regLoss)
      // the gradient is a vector of -ones - the gradient of regLoss
      val potGradData = Array.fill(pot.length)(1.0).zipWithIndex.map(
        x => if(x._2 < Ex) 1.0/Ex else 1.0/Ey
      )
      val potGrad = new DV[Double](data=potGradData,offset=0,stride=1,length=Ex+Ey)
      val gradient = -potGrad - (regGrad / (Ex*Ey).toDouble)
      //val gradient = regGrad


      //totalLoss/normFactor -> gradient/normFactor
      totalLoss -> gradient
    }

    // this is already amenable to microtests :)

  }

  // random potential for evaluation
  val rPot = DV.rand(size=Ex+Ey, rand=Rand.uniform)

  val (v1, g1) = localLoss.calculate(rPot)

  // parallelize
  val rXPoints = sc.parallelize(xPoints, 3)
  val rYPoints = sc.parallelize(yPoints, 3)

  val costs = DualSpace.computePairwiseCosts(rXPoints, rYPoints, ctx, costFun)
  costs.setName("costs")

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // distributed potential
  val dPot = new FDV(DDV.fromBreezeDense(rPot, elementsPerBlock=eb, sc))

 val dLoss2 = DualSpace.potentialLossWSinkhornRegularization(ctx, sinkReg, costs, 2)

  val (distValue, distGrad) = dLoss2.compute(dPot)

  testLogger.info(
    s"""
      discrepancy between local & distributed loss: ${v1 - distValue}
    """.stripMargin)

  // pull back the gradient
  val bDistGrad = DDV.toBreeze(distGrad.asInstanceOf[FDV[Double]].ddvec)
  testLogger.info(
    s"""
      discrepancy between local & distributed gradients:
      ${bNorm(g1 - bDistGrad)}
    """.stripMargin)

}

