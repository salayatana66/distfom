/*************************************************************************************************
 *  The Scalability Experiment 1 in the paper
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/ScalabilityExperiment_1.scala", $intp)
 ************************************************************************************************/

import distfom.{DistributedDenseVector => DDV,
  Example, ArrayBatcher, ArrayBatcherRefreshOnHold, Models, 
  FomDistDenseVec => FDDV, LBFGS, BlockPartitioner, RepDDV, LinearScorer, ProductPartitioner}

import org.apache.spark.sql.SparkSession

import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseVector => DV}

// logger
import org.apache.log4j.{Logger, Level}

// hadoop imports to access hdfs
import org.apache.hadoop.fs.{FileSystem,Path}

import org.apache.spark.mllib.random.RandomRDDs._

import breeze.stats.distributions.{Uniform}


object ScalabilityExperiment_1WeightPrep {

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
  val rddSaveDir = "/giuseppe_conte/scalability_experiment1"

  // Parameters for the examples
  val numSamplesFold = 1e8.toInt
  val numExaPartitions = 100
 
  // Paremeters for Feature space 
  // numBaseFeas features are sampled uniformly; weights in [0, 1]
  val feasPerExample = 30
  val minFeas = 0
  val maxFeas = 1e9.toInt
  val feaPart = 100
  val wgtWritePart = 50

  val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, x._1))

    // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights = DDV.fromRDDWithSugg(suggestedNumPartitions = feaPart,
    elements = maxFeas, rdd = weightRDD)

  val fdweights = new FDDV(dweights, 5)
  fdweights.setName("true_weights")
  fdweights.persist.count

  fdweights.ddvec.self.repartition(wgtWritePart)
    .saveAsObjectFile(s"$rddSaveDir/true_weights")

}


object ScalabilityExperiment_1DataPrep {

  import distfom.{RepDDV, LinearScorer, ProductPartitioner}

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

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment1")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment1")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)


  /*******************************************************
   *  Data settings
   *******************************************************/

  // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment1/"

  // Parameters for the examples
  val numSamplesFold = 1e8.toInt
  val numExaPartitions = 100

  // Paremeters for Feature space 
  // numBaseFeas features are sampled uniformly in [-1, 1]
  val feasPerExample = 30
  val minFeas = 0
  val maxFeas = 1e9.toInt
  val feaPart = 100

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // load the weights; we need to get a BlockPartitioner
  val bp = BlockPartitioner.withNumPartitions(elements=maxFeas, suggestedNumPartitions=feaPart)
  val rawWeights = sc.objectFile(path=s"$rddSaveDir/true_weights").
    asInstanceOf[RDD[(Int, DV[Double])]]
  val repRawWgt = rawWeights.partitionBy(bp)
  val dweights = new DDV[Double](repRawWgt)

  val fdweights = new FDDV(dweights, 5)
  fdweights.setName("true_weights")
  fdweights.persist.count

  var linWg: RepDDV[Double] = null

  // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a: String) => a match {
      case idRegx(fold, id) => id.toLong
    }
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
      linWg = RepDDV.apply(fdweights.ddvec, dFeas.self.partitioner.get.
        asInstanceOf[ProductPartitioner].leftPartitioner)

      linWg.self.setName("RepDDV")
      ()
    } else ()

    scores = LinearScorer.apply(dFeas, linWg)

    _ = scores.self.cache.count

    // join examples & scores using the id
    joinedRDD = scores.self.join(
      batch_examples.map(x => (serMap.serFun(x.exid), x)))

    scoredExamples = joinedRDD.map(x =>
      new Example[String, Double, Double](exid = x._2._2.exid,
        target = x._2._1, weight = x._2._2.weight,
        features = x._2._2.features)
    )

    _ = scoredExamples.repartition(numExaPartitions).
    saveAsObjectFile(path = s"$rddSaveDir/examples_fold_$currentFold ")
  }

  hfs.delete(chkPath)
}

object ScalabilityExperiment_1LBFGSRegBatch {

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

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment1")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment1")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

   /*******************************************************
   *  Data settings
   *******************************************************/

    // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment1/"

  // Parameters for the examples
  val numSamplesFold = 1e8.toInt
  val numExaPartitions = 100

  // Paremeters for Feature space 
  // numBaseFeas features are sampled uniformly
  val feasPerExample = 30
  val minFeas = 0
  val maxFeas = 1e9.toInt
  val feaPart = 100

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  /******************************************************
   *  Random initialization of Weights
   *  a Random uniform initialization
   ******************************************************/


    val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, .3*x._1))

    // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights = DDV.fromRDDWithSugg(suggestedNumPartitions = feaPart,
    elements = maxFeas, rdd = weightRDD)

  val fsweights = new FDDV(dweights, 5)
  fsweights.setName("start_weights")

 // Load data
  val batch_examples = for {
    currentFold <- 0 to 4
  } yield sc.objectFile(path=s"$rddSaveDir/examples_fold_$currentFold").
    asInstanceOf[RDD[Example[String, Double, Double]]]

    // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a:String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, Double, Double](
    indexer=serMap.serFun, maxExampleIdx=numSamplesFold,
    maxFeaIdx=maxFeas, suggPartExamples=numExaPartitions, suggPartFeas=feaPart,
    batches=batch_examples.toArray)

    val l2loss = Models.linearL2Loss(batcher)

  // optimize1r
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m=5,
    maxIter=10, tolerance=1e-4, lowerFunTolerance=1.0,
    interruptLinSteps=2, baseLSearch=1e5)

  // implicit conversions
  import FDDV.implicits._

  val finalState = LBFGSMin.minimizeAndReturnState(f=l2loss, init=fsweights)

  hfs.delete(chkPath)

}

object ScalabilityExperiment_1LBFGSOneBatch {

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

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment1")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment1")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

   /*******************************************************
   *  Data settings
   *******************************************************/

    // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment1/"

  // Parameters for the examples
  val numSamplesFold = 1e8.toInt
  val numExaPartitions = 100

  // Paremeters for Feature space 
  // numBaseFeas features are sampled uniformly
  val feasPerExample = 30
  val minFeas = 0
  val maxFeas = 1e9.toInt
  val feaPart = 100

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  /******************************************************
   *  Random initialization of Weights
   *  a Random uniform initialization
   ******************************************************/


    val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, .3*x._1))

    // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights = DDV.fromRDDWithSugg(suggestedNumPartitions = feaPart,
    elements = maxFeas, rdd = weightRDD)

  val fsweights = new FDDV(dweights, 5)
  fsweights.setName("start_weights")

  // Load data
  val currentFold = 0
  val batch_examples = sc.objectFile(path=s"$rddSaveDir/examples_fold_$currentFold").
    asInstanceOf[RDD[Example[String, Double, Double]]]

    // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a:String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcher[String, Double, Double](
    indexer=serMap.serFun, maxExampleIdx=numSamplesFold,
    maxFeaIdx=maxFeas, suggPartExamples=numExaPartitions, suggPartFeas=feaPart,
    batches=Array(batch_examples))

    val l2loss = Models.linearL2Loss(batcher)

  // optimize1r
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m=5,
    maxIter=10, tolerance=1e-4, lowerFunTolerance=1.0,
    interruptLinSteps=2, baseLSearch=1e5)

  // implicit conversions
  import FDDV.implicits._

  val finalState = LBFGSMin.minimizeAndReturnState(f=l2loss, init=fsweights)

  hfs.delete(chkPath)

}

object ScalabilityExperiment_1LBFGSRefreshBatch {

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

  @transient val chkPath = new Path("/tmp/ScalabilityExperiment1")
  hfs.mkdirs(chkPath)
  sc.setCheckpointDir("/tmp/ScalabilityExperiment1")

  @transient val testLogger = Logger.getLogger("Current Test")
  testLogger.setLevel(Level.INFO)

   /*******************************************************
   *  Data settings
   *******************************************************/

    // directory for saving rdds
  val rddSaveDir = "/giuseppe_conte/scalability_experiment1/"

  // Parameters for the examples
  val numSamplesFold = 1e8.toInt
  val numExaPartitions = 100

  // Paremeters for Feature space 
  // numBaseFeas features are sampled uniformly
  val feasPerExample = 30
  val minFeas = 0
  val maxFeas = 1e9.toInt
  val feaPart = 100

  // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  /******************************************************
   *  Random initialization of Weights
   *  a Random uniform initialization
   ******************************************************/


    val weightRDD = uniformRDD(sc, size = maxFeas, numPartitions = feaPart).
    zipWithIndex.map(x => (x._2, .3*x._1))

    // Import the field implicits
  import distfom.CommonImplicitGenericFields._
  import distfom.CommonImplicitGenericSmoothFields._

  // create the model weights
  val dweights = DDV.fromRDDWithSugg(suggestedNumPartitions = feaPart,
    elements = maxFeas, rdd = weightRDD)

  val fsweights = new FDDV(dweights, 5)
  fsweights.setName("start_weights")

 // Load data
  val batch_examples = for {
    currentFold <- 0 to 4
  } yield sc.objectFile(path=s"$rddSaveDir/examples_fold_$currentFold").
    asInstanceOf[RDD[Example[String, Double, Double]]]

    // for serialization of examples' id
  object serMap extends Serializable {
    // mapping examples ids to Long

    val idRegx = raw"fold(\d+)_(\d+)".r
    val serFun = (a:String) => a match {
      case idRegx(fold, id) => id.toLong
    }
  }

  // instantiate a Batcher
  val batcher = new ArrayBatcherRefreshOnHold[String, Double, Double](
    indexer=serMap.serFun, maxExampleIdx=numSamplesFold,
    maxFeaIdx=maxFeas, suggPartExamples=numExaPartitions, suggPartFeas=feaPart,
    defaultDepth=5, batches=batch_examples.toArray)

    val l2loss = Models.linearL2Loss(batcher)

  // optimize1r
  val LBFGSMin = new LBFGS[FDDV[Double], Double](m=5,
    maxIter=10, tolerance=1e-4, lowerFunTolerance=1.0,
    interruptLinSteps=2, baseLSearch=1e5)

  // implicit conversions
  import FDDV.implicits._

  val finalState = LBFGSMin.minimizeAndReturnState(f=l2loss, init=fsweights)

  hfs.delete(chkPath)

}
