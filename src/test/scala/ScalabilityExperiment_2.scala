/*************************************************************************************************
 *  The Scalability Experiment 2 in the paper
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/ScalabilityExperiment_2.scala", $intp)
 ************************************************************************************************/

import distfom.{ArrayBatcher, ArrayBatcherRefreshOnHold, BlockPartitioner, Example, LBFGS, LinearScorer, Models, ProductPartitioner, RepSDDV, DistributedStackedDenseVectors => DSV, FomDistStackedDenseVec => FDSV, MultiLabelSoftMaxLogLoss => MLogLoss}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import breeze.stats.distributions.Rand
//import distfom.MultiLabelSoftMaxLogLoss.MultiLabelTarget

// logger
import org.apache.log4j.{Logger, Level}

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem,Path}

import org.apache.spark.mllib.random.RandomRDDs._

import breeze.stats.distributions.{Uniform}

object ScalabilityExperiment_2WeightPrep {

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

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment2"


  // Paremeters for Feature space
  // numBaseFeas features are sampled uniformly; weights in [0, 1]
  val feasPerExample = 100
  val minFeas = 0
  val maxFeas = 5e7.toInt
  val feaPart = 100
  val wgtWritePart = 100
  val rowDim = 10

  val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, x._1)).
    map(x => (x._1, DV.rand(size=rowDim, rand=Rand.uniform)))

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights = DSV.fromRDDWithSugg[Double](suggestedNumPartitions = feaPart,
    elements = maxFeas, rowDim, rdd = weightRDD)

  val fdweights = new FDSV(dweights, 5)
  fdweights.setName("true_weights")
  fdweights.persist.count

  fdweights.dsvec.self.repartition(wgtWritePart)
    .saveAsObjectFile(s"$rddSaveDir/true_weights")

}

object ScalabilityExperiment_2DataPrep {

  import distfom.{RepSDDV, LinearScorer, ProductPartitioner}

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

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment2")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment2")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)


  /** *****************************************************
   * Data settings
   * ******************************************************/

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment2/"

  // Parameters for the examples
  val numSamplesFold = 2.5e8.toInt
  val numExaPartitions = 250
  val writeExaPartitions = 500

  // Paremeters for Feature space
  // numBaseFeas features are sampled uniformly; weights in [0, 1]
  val feasPerExample = 100
  val minFeas = 0
  val maxFeas = 5e7.toInt
  val feaPart = 100
  val wgtWritePart = 100
  val rowDim = 10

  // Import the field implicits

  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // load the weights; we need to get a BlockPartitioner
  val bp = BlockPartitioner.withNumPartitions(elements = maxFeas, suggestedNumPartitions = feaPart)
  val rawWeights = sc.objectFile(path = s"$rddSaveDir/true_weights").
    asInstanceOf[RDD[(Int, DM[Double])]]
  val repRawWgt = rawWeights.partitionBy(bp)
  val dweights = new DSV[Double](repRawWgt, rowDim = rowDim)

  val fdweights = new FDSV(dweights, 5)
  fdweights.setName("true_weights")
  fdweights.persist.count

  var linWg: RepSDDV[Double] = null

  // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a: String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  import distfom.QuotientPartitioner

  import breeze.linalg.{max => bMax, sum => bSum}
  import breeze.numerics.{exp => bExp, log => bLog}
  // for multinomial sampling
  import breeze.stats.distributions.Multinomial

  import MLogLoss.MultiLabelTarget

  // breeze imports

  def createTargets(exIter: Iterator[(Long, Example[String, Double, Double])],
                    scIter: Iterator[(Long, DV[Double])]) = {
    val scMap = scIter.toList.toMap
    val exList = exIter.toList

    val out = for {
      example <- exList
      // vec holds the scores
      (id, ex) = example
      vec = scMap(id)

      // transform the scores to probabilities via stable softmax
      norm_raw_scores = vec - bMax(vec)
      exp_scores = bExp(norm_raw_scores)
      smax_scores = exp_scores / bSum(exp_scores)

      // Instance probability to sample the scores
      distro = new Multinomial(params = smax_scores)
      target_label = distro.draw()

      target = MultiLabelTarget(Array(target_label),
        Array(1.0))

      scored_exa = Example[String,
        MultiLabelTarget[Double], Double](exid = ex.exid,
        target = target, weight = ex.weight, features = ex.features)
    } yield scored_exa

    out.toIterator
  }

  for {
    currentFold <- 0 until 5

    /* create examples without label
   *  samples features via breeze
   */

    batch_examples = for {
      exa <- uniformRDD(sc, size = numSamplesFold, numPartitions = numExaPartitions).
        zipWithIndex

      exid = s"fold${currentFold}_${exa._2}"
      gen = new Uniform(minFeas, maxFeas)
      feaGen = new Uniform(-1.0, 1.0)
      feaArray = gen.sample(feasPerExample).
        map(x => (math.max(minFeas, math.min(maxFeas - 1, math.ceil(x))).
          toLong, feaGen.sample)).distinct.toArray

      target = -1.0
      weight = 1.0
    } yield new Example[String, Double, Double](exid = exid,
      target = target, weight = weight, features = feaArray)

    _ = batch_examples.checkpoint()



    // instantiate a Batcher
    batcher = new ArrayBatcher[String, Double, Double](
      indexer = serMap.serFun, maxExampleIdx = numSamplesFold,
      maxFeaIdx = maxFeas, suggPartExamples = numExaPartitions, suggPartFeas = feaPart,
      batches = Array(batch_examples))

    // get a batch
    (weight, dFeas, dTgts) = batcher.next(depth = 5)

    _ = if (linWg == null) {
      // get linear weights
      linWg = RepSDDV.apply(fdweights.dsvec, dFeas.self.partitioner.get.
        asInstanceOf[ProductPartitioner].leftPartitioner)

      linWg.self.setName("RepSDDV")
      ()
    } else ()

    scores = LinearScorer.apply(dFeas, linWg)

    _ = scores.self.cache.count


    qp = QuotientPartitioner.withNumPartitions(numSamplesFold, 1000)
    // join examples & scores using the id
    mappedBatch = batch_examples.map(x => (serMap.serFun(x.exid), x)).
      partitionBy(qp)


    scoredExamples = mappedBatch.zipPartitions(scores.self.
      partitionBy(qp),
      preservesPartitioning = true)(createTargets)

    _ = scoredExamples.repartition(writeExaPartitions).
      saveAsObjectFile(path = s"$rddSaveDir/examples_fold_$currentFold")

  }
    hfs.delete(chkPath)
}

object ScalabilityExperiment_2LBFGSRegBatch {

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

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment2")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment2")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /*******************************************************
    *  Data settings
    *******************************************************/

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment2/"

  // Parameters for the examples
  val numSamplesFold = 2.5e8.toInt
  val numExaPartitions = 250
  val writeExaPartitions = 500

  // Paremeters for Feature space
  // numBaseFeas features are sampled uniformly; weights in [0, 1]
  val feasPerExample = 100
  val minFeas = 0
  val maxFeas = 5e7.toInt
  val feaPart = 100
  val wgtWritePart = 100
  val rowDim = 10

  // random sampling initialization
  val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, x._1)).
    map(x => (x._1, DV.rand(size=rowDim, rand=Rand.uniform)))

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights =  DSV.fromRDDWithSugg[Double](suggestedNumPartitions = feaPart,
    elements = maxFeas, rowDim, rdd = weightRDD)

  val fsweights = new FDSV(dweights, 5)
  fsweights.setName("start_weights")

  import MLogLoss.MultiLabelTarget

  // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a:String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  // Load data
  val batch_examples = for {
    currentFold <- 0 to 4
  } yield sc.objectFile(path =s"$rddSaveDir/examples_fold_$currentFold").
    asInstanceOf[RDD[Example[String,
    MultiLabelTarget[Double], Double]]].repartition(numExaPartitions) /*.
    filter(x => serMap.serFun(x.exid) < numSamplesFoldLoc).repartition(numExaPartitionsLoc)
*/
  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, MultiLabelTarget[Double], Double](
    indexer=serMap.serFun, maxExampleIdx=numSamplesFold,
    maxFeaIdx=maxFeas, suggPartExamples=numExaPartitions, suggPartFeas=feaPart,
    batches=batch_examples.toArray)

  val logloss = Models.linearLogLoss(batcher=batcher, rowDim=rowDim, depth=3)

  // optimizer
  val LBFGSMin = new LBFGS[FDSV[Double], Double](m=5,
    maxIter=10, tolerance=1e-4, lowerFunTolerance=0.5,
    interruptLinSteps=3, baseLSearch=5e6)

  // implicit conversions
  import FDSV.implicits._

  // one iteration with a large step
  val finalState = LBFGSMin.minimizeAndReturnState(f=logloss,
    init=fsweights)

  hfs.delete(chkPath)

}

object ScalabilityExperiment_2LBFGSOneBatch {

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

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment2")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment2")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /*******************************************************
    *  Data settings
    *******************************************************/

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment2/"


  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  /******************************************************
    *  Random initialization of Weights
    *  a Random uniform initialization
    ******************************************************/

  // Parameters for the examples
  val numSamplesFold = 2.5e8.toInt
  val numExaPartitions = 250
  val writeExaPartitions = 500

  // Paremeters for Feature space
  // numBaseFeas features are sampled uniformly; weights in [0, 1]
  val feasPerExample = 100
  val minFeas = 0
  val maxFeas = 5e7.toInt
  val feaPart = 100
  val wgtWritePart = 100
  val rowDim = 10

  // random sampling initialization
  val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, x._1)).
    map(x => (x._1, DV.rand(size=rowDim, rand=Rand.uniform)))

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights =  DSV.fromRDDWithSugg[Double](suggestedNumPartitions = feaPart,
    elements = maxFeas, rowDim, rdd = weightRDD)

  val fsweights = new FDSV(dweights, 5)
  fsweights.setName("start_weights")

  import MLogLoss.MultiLabelTarget

  // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a:String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  // Load data
  val currentFold = 0
  val batch_examples = sc.objectFile(path=s"$rddSaveDir/examples_fold_$currentFold").
    asInstanceOf[RDD[Example[String,
    MultiLabelTarget[Double], Double]]].repartition(numExaPartitions)/*.
    filter(x => serMap.serFun(x.exid) < numSamplesFoldLoc).repartition(numExaPartitionsLoc)*/


  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, MultiLabelTarget[Double], Double](
      indexer=serMap.serFun, maxExampleIdx=numSamplesFold,
      maxFeaIdx=maxFeas, suggPartExamples=numExaPartitions, suggPartFeas=feaPart,
      batches=Array(batch_examples))

  val logloss = Models.linearLogLoss(batcher=batcher, rowDim=rowDim, depth=3)

  // optimize1r
  val LBFGSMin = new LBFGS[FDSV[Double], Double](m=5,
    maxIter=10, tolerance=1e-4, lowerFunTolerance=0.5,
    interruptLinSteps=3, baseLSearch=5e6)

  // implicit conversions
  import FDSV.implicits._

  // one iteration with a large step
  val finalState = LBFGSMin.minimizeAndReturnState(f=logloss,
    init=fsweights)

  hfs.delete(chkPath)

}

object ScalabilityExperiment_2LBFGSRefreshBatch {

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

  // Access the file system
  @transient val hdconf = sc.hadoopConfiguration
  @transient val hfs = FileSystem.get(hdconf)

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment2")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment2")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

  /*******************************************************
    *  Data settings
    *******************************************************/

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment2/"

  // Parameters for the examples
  val numSamplesFold = 2.5e8.toInt
  val numExaPartitions = 250
  val writeExaPartitions = 500

  // Paremeters for Feature space
  // numBaseFeas features are sampled uniformly; weights in [0, 1]
  val feasPerExample = 100
  val minFeas = 0
  val maxFeas = 5e7.toInt
  val feaPart = 100
  val wgtWritePart = 100
  val rowDim = 10

  // random sampling initialization
  val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, x._1)).
    map(x => (x._1, DV.rand(size=rowDim, rand=Rand.uniform)))

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights =  DSV.fromRDDWithSugg[Double](suggestedNumPartitions = feaPart,
    elements = maxFeas, rowDim, rdd = weightRDD)

  val fsweights = new FDSV(dweights, 5)
  fsweights.setName("start_weights")

  import MLogLoss.MultiLabelTarget

  // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a:String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  // TODO: commented code to be used for finding initializations
  // TODO: commented code to find the first learning rates in LFBGS
  /*val numSamplesFoldLoc = 2e7.toInt
  val numExaPartitionsLoc = 50*/

  // Load data
  val batch_examples = for {
    currentFold <- 0 to 4
  } yield sc.objectFile(path =s"$rddSaveDir/examples_fold_$currentFold").
    asInstanceOf[RDD[Example[String,
    MultiLabelTarget[Double], Double]]].repartition(numExaPartitions) /*.
    filter(x => serMap.serFun(x.exid) < numSamplesFoldLoc).repartition(numExaPartitionsLoc)
*/
  // instantiate a Batcher
  val batcher = new ArrayBatcherRefreshOnHold[String, MultiLabelTarget[Double], Double](
    indexer=serMap.serFun, maxExampleIdx=numSamplesFold,
    maxFeaIdx=maxFeas, suggPartExamples=numExaPartitions, suggPartFeas=feaPart,
    batches=batch_examples.toArray)

  val logloss = Models.linearLogLoss(batcher=batcher, rowDim=rowDim, depth=3)

  // optimizer
  val LBFGSMin = new LBFGS[FDSV[Double], Double](m=5,
    maxIter=10, tolerance=1e-4, lowerFunTolerance=0.5,
    interruptLinSteps=3, baseLSearch=5e6)

  // implicit conversions
  import FDSV.implicits._

  // one iteration with a large step
  val finalState = LBFGSMin.minimizeAndReturnState(f=logloss,
    init=fsweights)

  hfs.delete(chkPath)

}
