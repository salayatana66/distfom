/******************************************************
 * Records to store training examples (scored or not)\
 * @author Andrea Schioppa
 ******************************************************/

package distfom

import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.rdd.{RDD, PairRDDFunctions}

import breeze.linalg.{SparseVector => SV, DenseVector => DV,
DenseMatrix => DM}

import scala.reflect.{ClassTag, classTag}
import scala.reflect.runtime.universe.{TypeTag, typeTag}

import scala.language.implicitConversions


import CommonImplicitGenericFields._

/** Represent an example which can be scored or not
 * exid is the example identifier which we leave as
 * a generic type
 */

case class Example[E, T, @specialized (Double, Float) F](
  exid: E, target: T, weight: F, features: Array[(Long, F)])
{

  // constructor with default weight
  def this(exid: E, target: T, features: Array[(Long, F)])
    (implicit fieldImp: GenericField[F]) =
    this(exid = exid, target = target, weight = fieldImp.opFromDouble(1.0),
      features = features)

}


/******************************************************
 * Partitioner mapping i to i / elementsPerPartition;
 * to be used with examples indices
 ******************************************************/

class QuotientPartitioner(val elements: Long, val elementsPerPartition: Int)
    extends Partitioner with Serializable {

  require(elements > 0, s"Number of elements $elements is not > 0")
  require(elementsPerPartition > 0,
    s"Number of elements per partition $elementsPerPartition is not > 0")

  override val numPartitions = math.ceil(elements * 1.0 / elementsPerPartition).toInt

  override def getPartition(key: Any): Int = {
    key match {
      case i: Long => {
        require(0 <= i && i < elements, s"key $i not in range [0, $elements)")
          (i / elementsPerPartition).toInt
      }
      case _ =>
        throw new IllegalArgumentException(s"Type of key $key is not Long")
    }
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: QuotientPartitioner =>
        (this.elementsPerPartition == r.elementsPerPartition) && (this.elements == r.elements)
      case _ =>
        false
    }
  }
}

object QuotientPartitioner {

  def apply(elements : Long, elementsPerPartition : Int) : QuotientPartitioner =
    new QuotientPartitioner(elements, elementsPerPartition)

  def withNumPartitions(elements: Long, suggestedNumPartitions : Int) :
      QuotientPartitioner = {
    val scale = 1.0/suggestedNumPartitions
    val elementsPerPartition = math.max(math.round(scale*elements), 1.0).toInt
    apply(elements, elementsPerPartition)
  }
}


/******************************************************
 * Partitioner combining Quotient and Block for
 * scoring examples; similar idea to the one of the
 * GridPartitioner in Spark
 ******************************************************/

class ProductPartitioner(val leftPartitioner: QuotientPartitioner,
  val rightPartitioner: BlockPartitioner) extends Partitioner with Serializable {

  override val numPartitions: Int = leftPartitioner.numPartitions *
  rightPartitioner.numPartitions

  override def getPartition(key: Any): Int = {
    key match {
      case Tuple2(i: Long, k: Int) => {
        require(0 <= i && i < leftPartitioner.elements,
          s"left key $i not in range [0, ${leftPartitioner.elements})")

        require(0 <= k && k < rightPartitioner.elements,
          s"right key $k not in range [0, ${rightPartitioner.elements})")

        leftPartitioner.getPartition(i) + leftPartitioner.numPartitions *
        rightPartitioner.getPartition(k)
      }
      case _ =>
        throw new IllegalArgumentException(s"Type of key $key is not Tuple2[Long, Int]")
    }
  }

  override def equals(obj: Any) = obj match {

    case o : ProductPartitioner => (o.leftPartitioner == this.leftPartitioner) &&
      (o.rightPartitioner == this.rightPartitioner)
    case _ => false

  }
}


/******************************************************
 * Features Distributed for scoring
 * In (Long, Int), Long is a uid for the example in the batch,
 * while Int is a key partition for the block of features
 * scoring
******************************************************/
class DistributedFeatures[@specialized (Double, Float) F ]
  (@transient val self: RDD[((Long, Int), SV[F])], val pp : ProductPartitioner)
    extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: ProductPartitioner) => a == pp
    case _ : Any => false
  }

  require(partitionerCheck, "self partitioner must be a ProductPartitioner")

}

/** Class representing an example where the features are not
 * present, just the ints representing the partitions holding the
 * features
 */

case class RichTarget[E, T, @specialized (Double, Float) F](
  exid: E, target: T, weight: F, features: Array[Int])
{

  // constructor with default weight
  def this(exid: E, target: T, features: Array[Int])
    (implicit fieldImp: GenericField[F]) =
    this(exid = exid, target = target, weight = fieldImp.zero,
      features = features)

}


/******************************************************
 * Distributed targets; the Long holds a unique identifier,
 * we use RichTarget to represent the target as it helps storing
 * data that we need later when computing the gradients
 ******************************************************/
class DistributedTargets[E, T, @specialized (Double, Float) F]
  (@transient val self: RDD[(Long, RichTarget[E, T, F])],
  val qp: QuotientPartitioner )
    extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: QuotientPartitioner) => a == qp
    case _ : Any => false
  }

  require(partitionerCheck, "partitioner check failed")

}

/**
 * Trait representing a Batcher which at each
 * iteration can pull out a different subset of the
 * examples
 */
trait Batcher[E, T, @specialized (Double, Float) F] {

  /* for an array operation downstream we need F to have
   * a ClassTag; but unfortunately we cannot put a type bound
   * in a parametrized trait
   * We thus put a field fTag and push it down using an implicit
   * when we deal with the Array in distributeFeatures
   */
  val fTag: ClassTag[F]
  val tTag: TypeTag[F]

  // hold current batch
  def holdBatch() : Unit

  // stop holding batch
  def stopHoldingBatch() : Unit

  // next batch
  def next(depth: Int = 2): Tuple3[F, DistributedFeatures[F],
    DistributedTargets[E, T, F]]

  // next features
  def nextFeatures(): DistributedFeatures[F]

  // number of batches
  def numBatches(): Int

  import breeze.storage.Zero._

  // get the total weight of examples
  def totalWeight(targets: DistributedTargets[E, T, F],
    depth: Int = 2): F = {

    val out = if(tTag.tpe.toString == "Float") targets.self.
      map(x => x._2.weight.asInstanceOf[Float]).treeAggregate(0.0f)(
        seqOp = _ + _, combOp = _ + _, depth = depth)
    else if (tTag.tpe.toString == "Double") targets.self.
      map(x => x._2.weight.asInstanceOf[Double]).treeAggregate(0.0)(
        seqOp = _ + _, combOp = _ + _, depth = depth)
    else throw new Error("unimplemented")

    out.asInstanceOf[F]
  }

  // distribute the features
  def distributeFeatures(maxExampleIdx : Long,
    maxFeaIdx: Long, suggPartExamples : Int,
    suggPartFeas: Int, rdd: RDD[Example[E, T, F]])(indexer :E => Long)
    (implicit zeroValue: breeze.storage.Zero[F]):
      DistributedFeatures[F] = {

    val exaPartitioner = QuotientPartitioner.withNumPartitions(
      elements = maxExampleIdx, suggestedNumPartitions = suggPartExamples)

    val feaPartitioner = BlockPartitioner.withNumPartitions(
      elements = maxFeaIdx, suggestedNumPartitions = suggPartFeas)

    val partitioner = new ProductPartitioner(exaPartitioner, feaPartitioner)

    val keyModuler = (k : Long) => (k/feaPartitioner.elementsPerBlock).toInt

    implicit val locTag = fTag

    val rolledRdd = for {

      example <- rdd
      exIdx = indexer(example.exid)

      // group features by key blocks
      groups = example.features.groupBy(x => keyModuler(x._1))
      Tuple2(feaKey, feas) <- groups

      /** sparse vector constructor assumes
       *  indices are already ordered
       *  note how we shift the feature keys similarly
       *  as they get shifted when distributing a dense
       *  breeze vector to become a Distributed Dense Vector
       */
      ordFeas = feas.map(x=> (x._1-feaKey*
        feaPartitioner.elementsPerBlock, x._2)).sortBy(x => x._1).toArray

      svec = new SV[F](index = ordFeas.map(_._1.toInt),
        data = ordFeas.map(_._2),
        length = feaPartitioner.elementsPerBlock)

    } yield ((exIdx, feaKey), svec)

    new DistributedFeatures(rolledRdd.partitionBy(partitioner), partitioner)

  }

  // distribute the targets (with feature information need later for back-propagation)
  def distributeTargets(maxExampleIdx : Long, maxFeaIdx: Long, suggPartExamples : Int,
                        suggPartFeas: Int, rdd: RDD[Example[E, T, F]])(indexer : E => Long)
    (implicit zeroValue: breeze.storage.Zero[F]):
      DistributedTargets[E, T, F] = {

    val exaPartitioner = QuotientPartitioner.withNumPartitions(
      elements = maxExampleIdx, suggestedNumPartitions = suggPartExamples)

    val feaPartitioner = BlockPartitioner.withNumPartitions(
      elements = maxFeaIdx, suggestedNumPartitions = suggPartFeas)

    val keyModuler = (k : Long) => k/feaPartitioner.elementsPerBlock

    val rolledRdd = for {
      example <- rdd
      exid = example.exid
      idx = indexer(exid)
      target = example.target
      weight = example.weight
      feaKeys = example.features.
             map(x => keyModuler(x._1).toInt).
             distinct

    } yield (idx, RichTarget(exid=exid, target=target, weight=weight,
                  features=feaKeys))

    new DistributedTargets(rolledRdd.partitionBy(exaPartitioner),
    exaPartitioner)

  }
}

// Construct batches from an Array of rdds
class ArrayBatcher [E, T, @specialized (Double, Float) F : ClassTag : TypeTag]
  (val indexer: E=>Long, val maxExampleIdx: Long, val maxFeaIdx: Long,
    val suggPartExamples: Int, val suggPartFeas: Int,
    @transient val batches: Array[RDD[Example[E, T, F]]], val name: String = "unnamed")
  (implicit zeroValue: breeze.storage.Zero[F])
    extends Batcher[E, T, F] with Logging with Serializable {

  /** ArrayBatcher is a proper class so we can now get the
   * ClassTag
   */
  override val fTag: ClassTag[F] = classTag[F]
  override val tTag: TypeTag[F] = typeTag[F]
  // batches' index we point to
  private var idx = -1
  // access idx
  def getIdx: Int = idx
  // batches in epoch
  def batchesInEpoch: Int = batches.size
  // did we complete an epoch?
  def oneEpochCompleted : Boolean = this.getIdx + 1 >= this.batchesInEpoch
  // shall we hold the batch?
  protected var holdingBatch = false
  // did we ever load a batch (of features or targets)?
  private var loadedBatch = false
  // pointers to data types for caching
  @transient private var batchFeas : DistributedFeatures[F] = null
  @transient private var batchTgts : DistributedTargets[E, T, F] = null
  @transient private var totalWeight: Option[F] = None

  def incrementBatch(i: Int = 1): Unit = {
    idx += i
    idx = (idx % batches.size)
    logDebug(s"idx  incremented to ${idx}")
  }

  def holdBatch() : Unit = {
    logDebug(s"Batcher_${name} idx @ $idx, we will start holding the batch")
    holdingBatch = true
  }

  // stop holding batch
  def stopHoldingBatch() : Unit = {
    logDebug(s"Batcher_${name} idx @ $idx, we will stop holding the batch")
    holdingBatch = false
  }

  def numBatches = batches.size

  // next batch
  def next(depth: Int = 2): Tuple3[F, DistributedFeatures[F],
    DistributedTargets[E, T, F]] = {

    if(!holdingBatch || !loadedBatch) {
      idx += 1
      idx = (idx % batches.size)
      logDebug(s"idx  incremented to ${idx}")
    }

    // are we holding a already loaded batch?
    if(loadedBatch & holdingBatch & (batchFeas != null)
    & (batchTgts != null) & (!totalWeight.isEmpty))
      return (totalWeight.get, batchFeas, batchTgts)

    // have we already loaded and there is only one batch?
    if(loadedBatch & (batches.size == 1) & (batchFeas != null) &
    (batchTgts != null) & (!totalWeight.isEmpty))
      return (totalWeight.get, batchFeas, batchTgts)

    // free resources if they were ever allocated
    if(batchFeas != null) batchFeas.self.unpersist()
    if(batchTgts != null) batchTgts.self.unpersist()

    // loading & caching the batch of features
    val newBatchFeas = distributeFeatures(maxExampleIdx=this.maxExampleIdx,
      maxFeaIdx=this.maxFeaIdx, suggPartExamples=this.suggPartExamples,
      suggPartFeas=this.suggPartFeas, rdd=this.batches(idx))(this.indexer)
    newBatchFeas.self.setName(s"Batcher_${name}_batchFea_$idx")
    logDebug(s"Batcher_${name} Caching new batch of features")
    newBatchFeas.self.cache.count

    // loading & caching the batch of targets
    val newBatchTgts = distributeTargets(maxExampleIdx=this.maxExampleIdx,
      maxFeaIdx=this.maxFeaIdx, suggPartExamples=this.suggPartExamples,
      suggPartFeas=this.suggPartFeas, rdd=batches(idx))(this.indexer)
    newBatchTgts.self.setName(s"Batcher_${name} batchTgt_$idx")
    logDebug(s"Batcher_${name} Caching new batch of targets")
    newBatchTgts.self.cache.count

    totalWeight = Some(this.totalWeight(targets=newBatchTgts, depth=depth))

    // a batch got loaded
    loadedBatch = true

    batchFeas = newBatchFeas
    batchTgts = newBatchTgts

    (totalWeight.get, batchFeas, batchTgts)
  }

  // next features
  def nextFeatures(): DistributedFeatures[F] = {

    if(!holdingBatch || !loadedBatch) {
      idx += 1
      idx = (idx % batches.size)
      logDebug(s"idx  incremented to ${idx}")
    }

    // are we holding an already loaded batch?
    if(loadedBatch & holdingBatch & (batchFeas != null))
      return batchFeas

    // have we already loaded and there is only one batch?
    if(loadedBatch & (batches.size == 1) & (batchFeas != null))
      return batchFeas

    // free resources if they were ever allocated
    if(batchFeas != null) batchFeas.self.unpersist()
    if(batchTgts != null) batchTgts.self.unpersist()

    // loading & caching the batch of features
    val newBatchFeas = distributeFeatures(maxExampleIdx=this.maxExampleIdx,
      maxFeaIdx=this.maxFeaIdx, suggPartExamples=this.suggPartExamples,
      suggPartFeas=this.suggPartFeas, rdd=this.batches(idx))(this.indexer)
    newBatchFeas.self.setName(s"Batcher_${name} batchFea_$idx")
    logDebug(s"Batcher_${name} Caching new batch of features")
    newBatchFeas.self.cache.count

    // a batch got loaded
    loadedBatch = true

    batchFeas = newBatchFeas

    batchFeas
  }

  def freeResources: Unit = {
    logDebug(s"Batcher_${name} Freeing resources")
    if(batchFeas != null) batchFeas.self.unpersist()
    if(batchTgts != null) batchTgts.self.unpersist()
  }
  
}

// Construct batches from an Array of rdds
class ArrayBatcherNoHold [E, T, @specialized (Double, Float) F : ClassTag : TypeTag]
  (override val indexer: E=>Long, override val maxExampleIdx: Long,
    override val maxFeaIdx: Long, override val suggPartExamples: Int,
    override val suggPartFeas: Int,
    @transient override val batches: Array[RDD[Example[E, T, F]]])
  (implicit zeroValue: breeze.storage.Zero[F])
    extends ArrayBatcher[E, T, F](indexer=indexer, maxExampleIdx=maxExampleIdx,
      maxFeaIdx=maxFeaIdx, suggPartExamples=suggPartExamples,
      suggPartFeas=suggPartFeas, batches=batches) {

  holdingBatch = false

  override def holdBatch() = {
    logDebug("This Batcher never holds a batch")
  }

  // stop holding batch
  override def stopHoldingBatch() : Unit = {
    logDebug("With this Batcher stopping holding the batch is not needed")
  }

}

// Refresh if you hold the batch; so line search will take place on new batch
class ArrayBatcherRefreshOnHold [E, T, @specialized (Double, Float) F : ClassTag : TypeTag]
  (override val indexer: E=>Long, override val maxExampleIdx: Long,
    override val maxFeaIdx: Long, override val suggPartExamples: Int,
    override val suggPartFeas: Int, val defaultDepth: Int = 2,
    @transient override val batches: Array[RDD[Example[E, T, F]]])
  (implicit zeroValue: breeze.storage.Zero[F])
    extends ArrayBatcher[E, T, F](indexer=indexer, maxExampleIdx=maxExampleIdx,
      maxFeaIdx=maxFeaIdx, suggPartExamples=suggPartExamples,
      suggPartFeas=suggPartFeas, batches=batches) {

  override def holdBatch() = {
    logDebug("Start holding the batch but will first advance the batch")
    val _ = next(this.defaultDepth)
  }


}


/******************************************************
 * Replicated version of a Dense Vector for scoring
 ******************************************************/
class RepDDV[@specialized (Float, Double) F]
  (@transient val self: RDD[((Long, Int), DV[F])], val pp: ProductPartitioner)
    extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: ProductPartitioner) => a == pp
    case _ : Any => false
 }
  require(partitionerCheck, "partitioner check failed")
}

object RepDDV {
  // construct it from a DistributedDenseVector

  def apply[@specialized (Float, Double) F](dvec : DistributedDenseVector[F],
      qp : QuotientPartitioner) : RepDDV[F] = {

    val partitioner = new ProductPartitioner(leftPartitioner = qp,
      rightPartitioner = dvec.self.partitioner.get.asInstanceOf[BlockPartitioner])

    // We make sure all partitions of the QuotientPartitioner qp
    // are represented
    val partKeys = Range(0, qp.numPartitions).toList.
      map(_ * qp.elementsPerPartition)

    val rolledRdd = for {
      part <- dvec.self
      Tuple2(key, vec) = part

      id <- partKeys
    } yield ( (id.toLong, key), vec)


    new RepDDV[F](rolledRdd.partitionBy(partitioner), partitioner)
  }
}

/******************************************************
 * Replicated version of a Stacked Dense Vector for scoring
 ******************************************************/
class RepSDDV[@specialized (Float, Double) F]
  (@transient val self: RDD[((Long, Int), DM[F])],
  val pp: ProductPartitioner)
    extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: ProductPartitioner) => a == pp
    case _ : Any => false
 }
  require(partitionerCheck, "partitioner check failed")
}

object RepSDDV {
  // construct it from a DistributedStackedDenseVectors

  def apply[@specialized (Float, Double) F](dsvec : DistributedStackedDenseVectors[F],
      qp : QuotientPartitioner) : RepSDDV[F] = {

    val partitioner = new ProductPartitioner(leftPartitioner = qp,
      rightPartitioner = dsvec.self.partitioner.get.asInstanceOf[BlockPartitioner])

    // We make sure all partitions of the QuotientPartitioner qp
    // are represented
    val partKeys = Range(0, qp.numPartitions).toList.
      map(_ * qp.elementsPerPartition)

    val rolledRdd = for {
      part <- dsvec.self
      Tuple2(key, vec) = part

      id <- partKeys
    } yield ( (id.toLong, key), vec)


    new RepSDDV[F](rolledRdd.partitionBy(partitioner), partitioner)
  }
}

/******************************************************
 * Represent scores; S is the score type
 ******************************************************/
class Scores[S](@transient val self: RDD[(Long, S)],
  val qp: QuotientPartitioner) extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: QuotientPartitioner) => a == qp
    case _ : Any => false
 }
  require(partitionerCheck, "partitioner check failed")
}

/******************************************************
 * Represent losses; with info for back-propagation
 * The information for back-propagation is a generic
 * as it can depend on the kind of model
 ******************************************************/
case class RichLoss[S, G](loss: S, grad: Option[G])

class Losses[S, G](@transient val self: RDD[(Long, RichLoss[S, G])],
  val qp: QuotientPartitioner) {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: QuotientPartitioner) => a == qp
    case _ : Any => false
 }
  require(partitionerCheck, "partitioner check failed")
}

/******************************************************
 * Example Id for pairing positive and negative examples
 * Using a link
 ******************************************************/

case class LinkedExLabel[E, L, @specialized (Float, Double) F]
(exId: E, linkId: L, relWeight: F)

// sampler is a map taking a label, target, weight and features
abstract class NegativeSampler[E, L, T, @specialized (Float, Double) F](
  val sampler: (LinkedExLabel[E, L, F], T, F, Array[(Long, F)]) =>
  Array[Example[LinkedExLabel[E, L, F], T, F]]) extends Logging with Serializable {

  type ExaType = Example[LinkedExLabel[E, L, F], T, F]
  def sample(exas: RDD[ExaType]): Array[RDD[ExaType]] = ???
}

// uniform with fixed iterations
class NegativeIterativeSampler[E, L, T, @specialized (Float, Double) F](
  override val sampler: (LinkedExLabel[E, L, F], T, F, Array[(Long, F)]) =>
  Array[Example[LinkedExLabel[E, L, F], T, F]], val iterations: Int) extends
    NegativeSampler(sampler){

  override def sample(exas: RDD[ExaType]): Array[RDD[ExaType]] = {
    for {
      iter <- 0 until this.iterations
      out = exas.flatMap(x => sampler(x.exid, x.target, x.weight, x.features))
    } yield out
  }.toArray
}

/******************************************************
 * Represent positive & negative scores
 ******************************************************/
class PositiveLabels[L, S](@transient val self: RDD[(Long, (L, S))],
  val qp: QuotientPartitioner) extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: QuotientPartitioner) => a == qp
    case _ : Any => false
  }

  require(partitionerCheck, "partitioner check failed")

}

class NegativeLabels[L, S](@transient val self: RDD[(Long, (L, S))],
  val qp: QuotientPartitioner) extends Serializable {

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: QuotientPartitioner) => a == qp
    case _ : Any => false
  }

  require(partitionerCheck, "partitioner check failed")

}



// TODO: add a batcher which filters from a single massive RDD
