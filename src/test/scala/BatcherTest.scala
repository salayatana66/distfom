/*******************************************************************************************
 *  Test for implementation of the Batcher
 *  When running with interactively with
 *  "autopasting from scripts": https://github.com/salayatana66/auto-paste-from-script
 *
 *  import auto.paste.from.script._
 *  val myS = AutoPasteFromScript("src/test/scala/BatcherTest.scala", $intp)
 *******************************************************************************************/

import distfom.{Example, ArrayBatcher,
  ArrayBatcherNoHold,
  DistributedFeatures => DFea,
  DistributedTargets => DTarg, RichTarget}

import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import breeze.linalg.{SparseVector => SV, norm}

import breeze.stats.distributions.Rand

// logger
import org.apache.log4j.{Logger,Level}

object BatcherTest {

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

  // utilities to check RDDs partition by partition
  object partitionInspector extends Serializable {

    val feaMapper = (i: Int, ie: Iterator[((Long, Int), SV[Float])])  => {
      ie.toList.map(x => (i, x)).
        toIterator
    }

    val tgtMapper = (i:Int, ie: Iterator[(Long, RichTarget[String,Float,Float])]) => {
      ie.toList.map(x => (i, x)).
        toIterator
    }
  }


  /* This test checks that the way the Batcher distributes
   * features is correct
   */

  val examplesLoc = Array(Example(exid="a", target=.5f,
    weight = 1.0f, 
    features=Array((2L, -.33f), (5L, 1.25f))),

    Example(exid="b", target= -0.5f,
    weight = 1.0f, 
    features=Array((3L, 5f), (6L, 1.42f), (7L, -.22f))),
    
    Example(exid="c", target=1.09f,
    weight = 1.2f, 
    features=Array((1L, 1.0f), (3L, 1.7f), (4L, -4.5f))),

    Example(exid="d", target=.15f,
    weight = 1.08f, 
    features=Array((3L, .44f), (5L, 1.67f)))

  )

  val examplesRdd1 = sc.parallelize(examplesLoc.slice(0,2),
    numSlices=4)
  val examplesRdd2 = sc.parallelize(examplesLoc.slice(2,4),
    numSlices=4)

  // for serialization
  object serMap extends Serializable {
  // mapping examples ids to Long
    val eidMap = Map("a" -> 1L, "b" -> 2L, "c" -> 25L, "d" -> 50L)
    val serFun = (a:String) => eidMap(a)
  }
  val batcher = new ArrayBatcher[String, Float, Float](
    indexer=serMap.serFun, maxExampleIdx=100L,
    maxFeaIdx=10L, suggPartExamples=3, suggPartFeas=3,
    batches = Array(examplesRdd1, examplesRdd2))

  // distribute the feas
  val dFeas1 = batcher.nextFeatures()
  
  testLogger.info(s"""
                   Num partitions: ${dFeas1.self.partitioner.get.numPartitions}
                   which should equal 16
                   """)

  // compute partition by partition
  val dFeasComp1 = dFeas1.self.mapPartitionsWithIndex(partitionInspector.feaMapper).
    collect()

    // distribute the feas
  val dFeas2 = batcher.nextFeatures()
  
  testLogger.info(s"""
                   Num partitions: ${dFeas2.self.partitioner.get.numPartitions}
                   which should equal 16
                   """)

  // compute partition by partition
  val dFeasComp2 = dFeas2.self.mapPartitionsWithIndex(partitionInspector.feaMapper).
    collect()

  // distribute the feas again to check we're back to index 0
  val dFeas3 = batcher.nextFeatures()
  
  testLogger.info(s"""
                   Num partitions: ${dFeas3.self.partitioner.get.numPartitions}
                   which should equal 16
                   """)

  // compute partition by partition
  val dFeasComp3 = dFeas3.self.mapPartitionsWithIndex(partitionInspector.feaMapper).
    collect()

/* In a correct solution:
 * "a" is sharded between partitions 0 & 4
 * "b" ----------------------------- 4 & 8
 * "c" ----------------------------- 0 & 4
 * "d" ----------------------------- 5
 */

  // now calling the batch shouldn't increment the index
  batcher.holdBatch()
    // distribute the feas
  val nextBatches = batcher.next()


  // compute partition by partition
  val dTgtsComp = nextBatches._3.self.mapPartitionsWithIndex(partitionInspector.tgtMapper).
    collect()

  // now we should be able to get a new batch
  batcher.stopHoldingBatch()
  val nextBatchesAgain = batcher.next()


  // compute partition by partition
  val dTgtsCompAgain = nextBatchesAgain._3.self.mapPartitionsWithIndex(partitionInspector.tgtMapper).
    collect()

  // quick test about a batcher holding a single batch
  val singleBatcher = new ArrayBatcher[String, Float, Float](
    indexer=serMap.serFun, maxExampleIdx=100L,
    maxFeaIdx=10L, suggPartExamples=3, suggPartFeas=3,
    batches = Array(examplesRdd1))

  // distribute the feas
  val sdFeas1 = singleBatcher.nextFeatures()
  // this should not prompt a refresh
  val sdFeas2 = singleBatcher.nextFeatures()
  // this should prompt a refresh
  val sdFeas3 = singleBatcher.next()
  // this hsould not prompt a refresh
  val sdFeas4 = singleBatcher.next()

  /* In a correct solution:
 * "a" should to go partition 0
 * "b" ---------------------- 0
 * "c" ---------------------- 0
 * "d" ---------------------- 1
 */

  val batcherNoHold = new ArrayBatcherNoHold[String, Float, Float](
    indexer=serMap.serFun, maxExampleIdx=100L,
    maxFeaIdx=10L, suggPartExamples=3, suggPartFeas=3,
    batches = Array(examplesRdd1, examplesRdd2))

    // now calling the batch shouldn't do anything
  batcherNoHold.holdBatch()
    // distribute the feas
  val nextBatchesNoHold = batcherNoHold.next()


  // compute partition by partition
  val dTgtsCompNoHold = nextBatchesNoHold._3.
    self.mapPartitionsWithIndex(partitionInspector.tgtMapper).
    collect()

  // now we should be able to get a new batch
  val nextBatchesNoHoldAgain = batcherNoHold.next()

  // compute partition by partition
  val dTgtsCompNoHoldAgain = nextBatchesNoHoldAgain._3.
    self.mapPartitionsWithIndex(partitionInspector.tgtMapper).
    collect()


}
