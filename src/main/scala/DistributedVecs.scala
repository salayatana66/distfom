package distfom

import org.apache.spark.rdd.{RDD, PairRDDFunctions}

import org.apache.spark.{Partitioner, SparkContext}

import scala.reflect.ClassTag

import scala.language.implicitConversions

import breeze.linalg.{DenseVector => DV, DenseMatrix => DM}

import scala.math

class BlockPartitioner(val elements: Long, val elementsPerBlock : Int)
    extends Partitioner with Serializable {

  require(elements > 0, s"Number of elements $elements is not > 0")
  require(elementsPerBlock > 0, s"Number of elements per block $elementsPerBlock is not > 0")

  override val numPartitions = math.ceil(elements * 1.0 / elementsPerBlock).toInt

  override def getPartition(key: Any): Int = {
    key match {
      case i: Int => {
        require(0 <= i && i < elements, s"key $i not in range [0, $elements)")
        i
      }
      case _ =>
        throw new IllegalArgumentException(s"Type of key $key is not Int")
    }
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: BlockPartitioner =>
        (this.elementsPerBlock == r.elementsPerBlock) && (this.elements == r.elements)
      case _ =>
        false
    }
  }

}

object BlockPartitioner {

  def apply(elements : Long, elementsPerBlock : Int) : BlockPartitioner =
    new BlockPartitioner(elements, elementsPerBlock)

  def withNumPartitions(elements: Long, suggestedNumPartitions : Int) : BlockPartitioner = {
    val scale = 1.0/suggestedNumPartitions
    val elementsPerBlock = math.max(math.round(scale*elements), 1.0).toInt
    apply(elements, elementsPerBlock)
  }

}

import CommonImplicitGenericFields._

class DistributedDenseVector[@specialized (Float, Double) F: ClassTag](@transient val self: RDD[(Int, DV[F])])(implicit fieldImp: GenericField[F])
    extends DistVec[DistributedDenseVector[F]] with Serializable {

  type DDV = DistributedDenseVector[F]

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: BlockPartitioner) => true
    case _ : Any => false
  }

  require(partitionerCheck, "self partitioner must be a BlockPartitioner")

  def mapBlocks(f : DV[F] => DV[F]) : DDV = {

    val liftedF = (il : Iterator[(Int, DV[F])]) => {

      val l = il.toList
      require(l.size == 1,
        "Distributed Dense Vector can have only one vector per partition")

      val Tuple2(lidx, ldv) = l(0)

      List((lidx, f(ldv))).iterator
    }

    val newSelf = self.mapPartitions(liftedF, preservesPartitioning = true)

    new DistributedDenseVector(newSelf)

  }

  def mapBlocks(other : DDV)(f : (DV[F], DV[F]) => DV[F]) : DDV = {
    require(self.partitioner == other.self.partitioner,
      "mapBlocks : this and other must have same partitioner")

    val liftedF = (il : Iterator[(Int, DV[F])],
      ir : Iterator[(Int, DV[F])]) => {

      // iterators can easily get empty, e.g.
      // il.size whould deplete il; so we convert
      // to list
      val l = il.toList
      val r = ir.toList

      require(l.size == 1,
        "Distributed Dense Vector can have only one vector per partition on left")
      require(r.size == 1,
        "Distributed Dense Vector can have only one vector per partition on right")

      val Tuple2(lidx, ldv) = l(0)
      val Tuple2(ridx, rdv) = r(0)

      require(lidx == ridx, "mapBlocks: left and right partitions must have same index")

      List((lidx, f(ldv, rdv))).iterator
    }

    val newSelf = self.zipPartitions(other.self,
      preservesPartitioning = true)(liftedF)

    new DistributedDenseVector(newSelf)
  }

  def mapBlocksWithKey(other : DDV)(f : (Int, DV[F], DV[F]) => DV[F]) : DDV = {
    require(self.partitioner == other.self.partitioner,
      "mapBlocks : this and other must have same partitioner")

    val liftedF = (il : Iterator[(Int, DV[F])],
      ir : Iterator[(Int, DV[F])]) => {

      // iterators can easily get empty, e.g.
      // il.size whould deplete il; so we convert
      // to list
      val l = il.toList
      val r = ir.toList

      require(l.size == 1,
        "Distributed Dense Vector can have only one vector per partition on left")
      require(r.size == 1,
        "Distributed Dense Vector can have only one vector per partition on right")

      val Tuple2(lidx, ldv) = l(0)
      val Tuple2(ridx, rdv) = r(0)

      require(lidx == ridx, "mapBlocks: left and right partitions must have same index")

      List((lidx, f(lidx, ldv, rdv))).iterator
    }

    val newSelf = self.zipPartitions(other.self,
      preservesPartitioning = true)(liftedF)

    new DistributedDenseVector(newSelf)
  }

  def mapBlocksWithKey(f : (Int, DV[F]) => DV[F]) : DDV = {

    val liftedF = (il : Iterator[(Int, DV[F])]) => {

      // iterators can easily get empty, e.g.
      // il.size whould deplete il; so we convert
      // to list
      val l = il.toList

      require(l.size == 1,
        "Distributed Dense Vector can have only one vector per partition on left")

      val Tuple2(lidx, ldv) = l(0)


      List((lidx, f(lidx, ldv))).iterator
    }

    val newSelf = self.mapPartitions(liftedF).
      partitionBy(this.self.partitioner.get.
        asInstanceOf[BlockPartitioner])

    new DistributedDenseVector(newSelf)
  }


  override def persist() = {
    self.persist()
    this
  }

  override def unpersist(blocking: Boolean) = {
    self.unpersist(blocking)
    this
  }

  // lift f to a map between dense vectors
  private def binaryLift(f : (F, F) => F) : ((DV[F], DV[F]) => DV[F]) = (a : DV[F], b: DV[F]) => {

    val combinedData = a.data.zip(b.data).map(x =>  f(x._1, x._2)).toArray[F]

    new DV[F](combinedData, 0, 1, a.size)

  }

  override def +(other: DDV) = this.mapBlocks(other)(binaryLift(fieldImp.opAdd)(_, _))
  override def -(other: DDV) = this.mapBlocks(other)(binaryLift(fieldImp.opSub)(_, _))
  override def *(other: DDV) = this.mapBlocks(other)(binaryLift(fieldImp.opMul)(_, _))
  override def /(other: DDV) = this.mapBlocks(other)(binaryLift(fieldImp.opDiv)(_, _))

}

object DistributedDenseVector {

  import scala.collection.mutable.ArrayBuffer

  type DDV[F] = DistributedDenseVector[F]

  def fromBreezeDense[@specialized (Double, Float) F: ClassTag](b: DV[F], elementsPerBlock: Int,
    sc : SparkContext)(implicit fieldImpl: GenericField[F]) : DDV[F] = {
    require(b.offset == 0, "offset must be set to 0")
    require(b.stride == 1, "stride must be set to 1")

    val numBlocks = math.ceil(1.0*b.data.length/elementsPerBlock).toInt
    val toParallelize = new ArrayBuffer[(Int, DV[F])]()

    for {
      j <- 0 until numBlocks

      dataSlice = b.data.slice(j*elementsPerBlock, math.min((j+1)*
        elementsPerBlock, b.data.size))
      dataPad = Array.fill(elementsPerBlock - dataSlice.size)(fieldImpl.zero)


    } yield {
      toParallelize +=
      Tuple2(j , new DV[F](dataSlice ++ dataPad, 0, 1, elementsPerBlock))
    }

    val bp = new BlockPartitioner(elements = b.data.size,
      elementsPerBlock = elementsPerBlock)

    val newSelf = sc.parallelize(toParallelize, numSlices = bp.numPartitions).
      partitionBy(bp)

    new DistributedDenseVector(newSelf)
  }

  def toBreeze[@specialized (Float, Double) F: ClassTag](vec : DDV[F])(implicit fieldImpl: GenericField[F]) : DV[F] = {

    val elements = vec.self.partitioner.get.
      asInstanceOf[BlockPartitioner].elements

    val elementsPerBlock = vec.self.partitioner.get.
      asInstanceOf[BlockPartitioner].elementsPerBlock

    
    val collBlocks = vec.self.collect().
      sortBy( _._1)

    // DV.vertcat requires an implicit zero parameter to be passed
    implicit val localZero: breeze.storage.Zero[F] = new breeze.storage.Zero[F] {
      def zero: F = fieldImpl.zero
    }

    DV.vertcat(collBlocks.map(_._2) :_*)

  }


  def fromBreezeDense[@specialized (Double, Float) F: ClassTag](suggestedNumPartitions: Int,
    sc: SparkContext)(b: DV[F])(implicit fieldImpl: GenericField[F])
 : DDV[F] = {
    val scale = 1.0/suggestedNumPartitions
    val elementsPerBlock = math.max(math.round(scale*b.data.size), 1.0).toInt

    fromBreezeDense(b, elementsPerBlock, sc)
  }

  def fromRDD[@specialized (Double, Float) F: ClassTag](elements : Long, elementsPerBlock : Int,
    rdd: RDD[(Long, F)])(implicit fieldImpl: GenericField[F]): DDV[F] = {

    implicit val localZero: breeze.storage.Zero[F] = new breeze.storage.Zero[F] {
      def zero: F = fieldImpl.zero
  }

    val rddWithBlocks = for {
      row <- rdd
      Tuple2(idx, vl) = row

      _ = require((idx >= 0) && (idx < elements),
        s"idx ${idx} not in range [0,${elements})")

      block = idx / elementsPerBlock
      idxInBlock = idx - block*elementsPerBlock

    } yield (block.toInt, (idxInBlock.toInt, vl))

    val bp = new BlockPartitioner(elements = elements,
      elementsPerBlock = elementsPerBlock)

    val repRdd = rddWithBlocks.partitionBy(bp)

    val partitionMapper = (it : Iterator[(Int, (Int, F))]) =>
    {
      val ls = it.toList

      require(ls.map(_._1).distinct.size == 1, "only one block id per partition allowed")

      for {
        Tuple2(block, Tuple2(idxInBlock, vl)) <- ls

        
      } yield require((idxInBlock >= 0) && (idxInBlock < elementsPerBlock),
        s"idxInBlock ${idxInBlock} must lie in [0, ${elementsPerBlock})")

      val outDV = DV.zeros[F](elementsPerBlock)

      for {
        Tuple2(block, Tuple2(idxInBlock, vl)) <- ls
      } yield (outDV(idxInBlock) = vl)

      val block = ls(0)._1

      List((block, outDV)).iterator
    }

    new DistributedDenseVector(
      repRdd.mapPartitions(partitionMapper, preservesPartitioning = true)
    )
  }

  def fromRDDWithSugg[@specialized (Double, Float) F: ClassTag](suggestedNumPartitions: Int,
    elements : Long, rdd: RDD[(Long, F)])(implicit fieldImpl: GenericField[F]): DDV[F] =
  {
    val scale = 1.0/suggestedNumPartitions
    val elementsPerBlock = math.max(math.round(scale*elements), 1.0).toInt
    fromRDD(elements, elementsPerBlock, rdd)

  }

}

/* This implementation of FomDistVec is based on
 * DistributedDenseVector
 * The dotDepth of the vector on the left
 * always decides the final dotDepth 
 */

import CommonImplicitGenericSmoothFields._

class FomDistDenseVec[@specialized (Double, Float) F: ClassTag] (
  @transient val ddvec : DistributedDenseVector[F],
  val dotDepth : Int = 2
)(implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
    extends FomDistVec[FomDistDenseVec[F], F] with Serializable {

  type FmDDV = FomDistDenseVec[F]
  type FmBase = FomDistVec[FmDDV, F]

  /* load implicit conversion from FmDDV < -- > FmBase */
  import FomDistDenseVec.implicits._

  def +(b: FmBase): FmBase = {

    new FmDDV(this.ddvec + b.ddvec, this.dotDepth)
  }

   def -(b: FmBase): FmBase = {

    new FmDDV(this.ddvec - b.ddvec, this.dotDepth)
   }

 def *(b: FmBase): FmBase = {

    new FmDDV(this.ddvec * b.ddvec, this.dotDepth)
 }

  def /(b: FmBase): FmBase = {
    new FmDDV(this.ddvec / b.ddvec, this.dotDepth)
  }

  def *(d: F): FmBase = {

    // helper method to multiply vector by a scalar
    val vectorMul: DV[F] => DV[F] = (v: DV[F]) => {
      val newData = v.data.map(x => fieldImpl.opMul(x, d))
      new DV[F](newData, 0, 1, newData.size)
    }

    val newVec = this.ddvec.mapBlocks(vectorMul)
    new FmDDV(newVec, this.dotDepth)

  }

  // helper function to lift a pairwise map
  def liftF(f: (F,F) => F) : (DV[F],DV[F]) => DV[F] =
    (a: DV[F], b: DV[F]) =>
  {
    val asize = a.data.size
    require(asize == b.data.size, "in liftF a and b must have same length")

    val outarray = a.data.zip(b.data).map(x=>f(x._1,x._2))

    new DV(outarray, 0, 1, asize)
  }


  import scala.reflect.runtime.universe._

  // helper function to lift a pairwise map accepting a key
  def keyedLiftF[K](f:  K)(implicit tagK: TypeTag[K], tagF: TypeTag[F])
      : (Int, DV[F],DV[F]) => DV[F] =  {

    // type check & conversion
    val kType = tagK.tpe.toString
    val fType = tagF.tpe.toString
    require(kType == s"(Int, $fType, $fType) => $fType",
      s"""
      For FomDistDenseVec keyedPairWiseFunction needs a function
      (Int, $fType, $fType) => $fType
      """)

    val cleanF = f.asInstanceOf[(Int, F, F) => F]

    // lift 
    (idx: Int, a: DV[F], b: DV[F]) => {

      val zipped = a.data.zip(b.data)

      // idx used as offset
      val computedData = zipped.zipWithIndex.map(x => cleanF(idx+x._2, x._1._1, x._1._2))

      new DV(computedData, 0, 1, computedData.size)
    }
  }

  def pairWiseFun(b: FmBase)(f: (F,F) =>F) : FmBase = {

    // lift the map f to work between DV[Double]s
    val liftedF = liftF(f)
    val newVec = this.ddvec.mapBlocks(b.ddvec)(liftedF)
    new FmDDV(newVec, this.dotDepth)
  }

  def keyedPairWiseFun[K](b: FmBase)
    (f : K)(implicit tagK: TypeTag[K], tagF: TypeTag[F]) : FmBase = {

    // lift the map f to work between DV[Double]s
    val liftedF = keyedLiftF(f)
    val newVec = this.ddvec.mapBlocksWithKey(b.ddvec)(liftedF)
    new FmDDV(newVec, this.dotDepth)
  }

  def applyElementWise(f: F => F) : FmBase = {

    val liftedF = (x:DV[F]) => new DV(x.data.map(f),0,1,x.data.size)

    val newVec = this.ddvec.mapBlocks(liftedF)

    new FmDDV(newVec,this.dotDepth)    
  }
 
  def name: String = this.ddvec.self.name

  def copy(): FmBase = new FmDDV(this.ddvec, this.dotDepth)

  def count() : Long = this.ddvec.self.count()

  def dot(other: FmBase) : F = {
    // distributed dot products
    val distDots = this.ddvec.mapBlocks(other.ddvec)(
      (a: DV[F], b: DV[F]) =>
      new DV[F](Array(smoothImpl.opDot(a, b)))
    ).self.map(x=>x._2(0))

    // aggregate the different dot products
    distDots.treeAggregate(fieldImpl.zero)(seqOp = fieldImpl.opAdd(_ , _),
        combOp= fieldImpl.opAdd(_ , _), depth = this.dotDepth)

  }

  def norm(p: Double = 2.0) : F = {

    val power = this.applyElementWise(smoothImpl.opAbsPow(_, p)).ddvec.
        self.map(x => x._2.data.reduce(fieldImpl.opAdd(_, _))).
        treeAggregate(fieldImpl.zero)(seqOp = fieldImpl.opAdd(_ , _),
          combOp= fieldImpl.opAdd(_ , _), depth = this.dotDepth)

      smoothImpl.opAbsPow(power, 1.0/p)
    }
  

  def setName(name: String): Unit = this.ddvec.self.setName(name)

  def persist() = {
    this.ddvec.self.persist()
    this
  }

  def unpersist(blocking: Boolean = true) = {
    this.ddvec.self.unpersist(blocking)
    this
  }

  def interruptLineage = this.ddvec.self.checkpoint

}

object FomDistDenseVec {

  type DDV[F] = DistributedDenseVector[F]
  type FmDDV[F] = FomDistDenseVec[F]
  type FmBase[F] = FomDistVec[FmDDV[F], F]

  object implicits {

    implicit def castToFmBase[F](tocast: FmDDV[F]): FmBase[F] = tocast  match {
    case a: FmDDV[F] => a
    case _ => throw new RuntimeException("""
   Failed to cast in FomDistDenseVec.implicits.castToFmBase
   """)
    }

    /* for some reason implementing
       this cast with pattern matching would result in
       hanging up for a
       long time. Resorted to using asInstanceOf
     */
    implicit def castToFmDDV[F](tocast: FmBase[F]): FmDDV[F] =
      tocast.asInstanceOf[FmDDV[F]]
  

  implicit def liftDiffFun[F](diffFun: DistributedDiffFunction[FmDDV[F],F]) :
      DistributedDiffFunction[FmBase[F], F] =
    new DistributedDiffFunction[FmBase[F], F] {

    override def computeValue(x: FmBase[F]): F = diffFun.
      computeValue(x)

    override def computeGrad(x: FmBase[F]): FmBase[F] = diffFun.
      computeGrad(x)

  // should return value & gradient
    def compute(x: FmBase[F]) : (F, FmBase[F]) = 
      diffFun.compute(x)
 
  // hold current batch
  override def holdBatch() : Unit = diffFun.holdBatch()

  // stop holding batch
  override def stopHoldingBatch() :Unit = diffFun.stopHoldingBatch()

  }

  }
}

/* Class representing a matrix (row, cols) where row is small compared toarray
 * cols and we distribute only across cols; this is the case of stacking
 * row vectors on top of each other
 */
class DistributedStackedDenseVectors[@specialized (Float, Double) F: ClassTag]
  (@transient val self: RDD[(Int, DM[F])], val rowDim: Int)(implicit fieldImp: GenericField[F])
    extends DistVec[DistributedStackedDenseVectors[F]] with Serializable {

  type DSV = DistributedStackedDenseVectors[F]

  private def partitionerCheck : Boolean = self.partitioner match {
    case Some(a: BlockPartitioner) => true
    case _ : Any => false
  }

  require(partitionerCheck, "self partitioner must be a BlockPartitioner")

   def mapBlocks(f : DM[F] => DM[F]) : DSV = {

    val liftedF = (il : Iterator[(Int, DM[F])]) => {

      val l = il.toList
      require(l.size == 1,
        "Distributed Stacked Dense Vectors can have only one matrix per partition")

      val Tuple2(lidx, ldm) = l(0)
      require(ldm.rows == this.rowDim,
        s"""Incorrect matrix dimension inside the partition, should be
            ${this.rowDim} but got ${ldm.rows}
        """)

      List((lidx, f(ldm))).iterator
    }

    val newSelf = self.mapPartitions(liftedF, preservesPartitioning = true)

    new DSV(newSelf, this.rowDim)

  }

    def mapBlocks(other : DSV)(f : (DM[F], DM[F]) => DM[F]) : DSV = {
    require(self.partitioner == other.self.partitioner,
      "mapBlocks : this and other must have same partitioner")

    val liftedF = (il : Iterator[(Int, DM[F])],
      ir : Iterator[(Int, DM[F])]) => {

      // iterators can easily get empty, e.g.
      // il.size whould deplete il; so we convert
      // to list
      val l = il.toList
      val r = ir.toList

      require(l.size == 1,
        "Distributed Stacked Dense Vectors can have only one matrix per partition on left")
      require(r.size == 1,
        "Distributed Stacked Dense Vectors can have only one matrix per partition on right")

      val Tuple2(lidx, ldv) = l(0)
      val Tuple2(ridx, rdv) = r(0)

      require(lidx == ridx, "mapBlocks: left and right partitions must have same index")
      require(ldv.rows == this.rowDim,
        s"""Incorrect matrix dimension inside the partition, should be
            ${this.rowDim} but got ${ldv.rows}
        """)

      require(ldv.rows == rdv.rows, "mapBlock matrices should have same number of rows")
      require(ldv.cols == rdv.cols, "mapBlock matrices should have same number of columns")
      List((lidx, f(ldv, rdv))).iterator
    }

    val newSelf = self.zipPartitions(other.self,
      preservesPartitioning = true)(liftedF)

    new DSV(newSelf, this.rowDim)
  }

    def mapBlocksWithKey(other : DSV)(f : (Int, DM[F], DM[F]) => DM[F]) : DSV = {
    require(self.partitioner == other.self.partitioner,
      "mapBlocks : this and other must have same partitioner")

    val liftedF = (il : Iterator[(Int, DM[F])],
      ir : Iterator[(Int, DM[F])]) => {

      // iterators can easily get empty, e.g.
      // il.size whould deplete il; so we convert
      // to list
      val l = il.toList
      val r = ir.toList

      require(l.size == 1,
        "Distributed Stacked Dense Vectors can have only one matrix per partition on left")
      require(r.size == 1,
        "Distributed Stacked Dense Vectors can have only one matrix per partition on right")


      val Tuple2(lidx, ldv) = l(0)
      val Tuple2(ridx, rdv) = r(0)

      require(lidx == ridx, "mapBlocks: left and right partitions must have same index")
      require(ldv.rows == this.rowDim,
        s"""Incorrect matrix dimension inside the partition, should be
            ${this.rowDim} but got ${ldv.rows}
        """)

      require(ldv.rows == rdv.rows, "mapBlock matrices should have same number of rows")
      require(ldv.cols == rdv.cols, "mapBlock matrices should have same number of columns")

      List((lidx, f(lidx, ldv, rdv))).iterator
    }

    val newSelf = self.zipPartitions(other.self,
      preservesPartitioning = true)(liftedF)

    new DSV(newSelf, this.rowDim)
  }

  override def persist() = {
    self.persist()
    this
  }

  override def unpersist(blocking: Boolean) = {
    self.unpersist(blocking)
    this
  }

  // lift f to a map between dense vectors
  private def binaryLift(f : (F, F) => F) : ((DM[F], DM[F]) => DM[F]) = (a : DM[F], b: DM[F]) => {

    require(a.isTranspose == b.isTranspose, "Two matrices must have the same major mode")

    val combinedData = a.data.zip(b.data).map(x =>  f(x._1, x._2)).toArray[F]

    // row_major mode
    if(a.isTranspose)
      new DM[F](rows=a.rows, cols=a.cols, data=combinedData, offset=0,
        majorStride=a.cols, isTranspose=true)
    else
      new DM[F](rows=a.rows, cols=a.cols, data=combinedData, offset=0,
        majorStride=a.rows, isTranspose=false)

  }

  override def +(other: DSV) = this.mapBlocks(other)(binaryLift(fieldImp.opAdd)(_, _))
  override def -(other: DSV) = this.mapBlocks(other)(binaryLift(fieldImp.opSub)(_, _))
  override def *(other: DSV) = this.mapBlocks(other)(binaryLift(fieldImp.opMul)(_, _))
  override def /(other: DSV) = this.mapBlocks(other)(binaryLift(fieldImp.opDiv)(_, _))

}

object DistributedStackedDenseVectors {

  import scala.collection.mutable.ArrayBuffer

  type DSV[F] = DistributedStackedDenseVectors[F]

  def fromBreezeDense[@specialized (Double, Float) F: ClassTag]
    (b: DM[F], elementsPerBlock: Int,
     sc : SparkContext)(implicit fieldImpl: GenericField[F]) = {

    val numBlocks = math.ceil(1.0*b.cols/elementsPerBlock).toInt
    val toParallelize = new ArrayBuffer[(Int, DM[F])]()

    for {
      j <- 0 until numBlocks


      /* the way we access the data slice depends on the mode; it is easier
       *  to slice in the column major mode
       *  In any mode the missing elements are always proportional
       *  to the number of rows, but the insertion level differs
       */

      padDataSlice = if(!b.isTranspose) {

        // in col_major mode the stride is the row size
      val unpadDataSlice =  b.data.slice(j*elementsPerBlock *
          b.majorStride, math.min((j+1)*
            elementsPerBlock * b.majorStride, b.data.size))

        // fill in the missing rows computing the number of missing columns
        val dataPad =  Array.fill(
          (elementsPerBlock - unpadDataSlice.size / b.majorStride) *
          b.rows)(fieldImpl.zero)

        // insertion is sequential
        unpadDataSlice ++ dataPad

      } else {
        val toFill = new ArrayBuffer[F]()
        for {
          /* if the matrix is in row major mode
           * we iterate over the rows
           */

          m <- 0 until b.rows

          low = m*b.majorStride + j*elementsPerBlock

          // we can sweep up the row up to desired column size
          high = math.min(m*b.majorStride
                          + (j+1)*elementsPerBlock,
                          m*b.majorStride
                            + b.cols)

          // missing entries to reach the desired column size
          missCols = elementsPerBlock - (high - low)

          // concatenate data with padding
          toAppend = b.data.slice(low, high) ++
          Array.fill(missCols)(fieldImpl.zero)

        } yield (
          // grow the buffer for the matrix
          toFill ++= toAppend
        )

         toFill.toArray
      }

      // note how the stride differs depending on the mode
      outDM = new DM[F](rows = b.rows, cols = elementsPerBlock,
        data = padDataSlice, offset = 0, majorStride = if(!b.isTranspose) b.majorStride
        else elementsPerBlock,
        isTranspose = b.isTranspose)
  
    }  yield {
      toParallelize +=
      Tuple2(j , outDM)
    }
      
    toParallelize.toArray

    val bp = new BlockPartitioner(elements = b.cols,
      elementsPerBlock = elementsPerBlock)

    val newSelf = sc.parallelize(toParallelize, numSlices = bp.numPartitions).
      partitionBy(bp)

    new DSV(newSelf, b.rows)
 
  }

  def toBreeze[@specialized (Float, Double) F: ClassTag](svec : DSV[F])
    (implicit fieldImpl: GenericField[F]) : DM[F] = {

    val elements = svec.self.partitioner.get.
      asInstanceOf[BlockPartitioner].elements

    val elementsPerBlock = svec.self.partitioner.get.
      asInstanceOf[BlockPartitioner].elementsPerBlock

    val collBlocks = svec.self.collect().
      sortBy( _._1)

    // DV.vertcat requires an implicit zero parameter to be passed
    implicit val localZero: breeze.storage.Zero[F] = new breeze.storage.Zero[F] {
      def zero: F = fieldImpl.zero
    }

    DM.horzcat(collBlocks.map(_._2) :_*)

  }

  def fromBreezeDense[@specialized (Double, Float) F: ClassTag](suggestedNumPartitions: Int,
    sc: SparkContext)(b: DM[F])(implicit fieldImpl: GenericField[F]): DSV[F] = {
    val scale = 1.0/suggestedNumPartitions
    val elementsPerBlock = math.max(math.round(scale*b.cols), 1.0).toInt

    fromBreezeDense(b, elementsPerBlock, sc)
  }

  // Note this method now only works with row major mode
  def fromRDD[@specialized (Double, Float) F: ClassTag](elements : Long, elementsPerBlock : Int,
    rowDim: Int, rdd: RDD[(Long, DV[F])])(implicit fieldImpl: GenericField[F]): DSV[F] = {

    implicit val localZero: breeze.storage.Zero[F] = new breeze.storage.Zero[F] {
      def zero: F = fieldImpl.zero
  }

    val rddWithBlocks = for {
      row <- rdd
      Tuple2(idx, vl) = row

      _ = require((idx >= 0) && (idx < elements),
        s"idx ${idx} not in range [0,${elements})")

      block = idx / elementsPerBlock
      idxInBlock = idx - block*elementsPerBlock

    } yield (block.toInt, (idxInBlock.toInt, vl))

    val bp = new BlockPartitioner(elements = elements,
      elementsPerBlock = elementsPerBlock)

    val repRdd = rddWithBlocks.partitionBy(bp)

    val partitionMapper = (it : Iterator[(Int, (Int, DV[F]))]) =>
    {
      val ls = it.toList

      require(ls.map(_._1).distinct.size == 1, "only one block id per partition allowed")

      for {
        Tuple2(block, Tuple2(idxInBlock, _)) <- ls

        
      } yield require((idxInBlock >= 0) && (idxInBlock < elementsPerBlock),
        s"idxInBlock ${idxInBlock} must lie in [0, ${elementsPerBlock})")

      for {
        Tuple2(block, Tuple2(_, vec)) <- ls

        
      } yield require(vec.size == rowDim,
        s"Each vector must contain ${rowDim} rows")

      val unpadData = ls.map(_._2).sortBy(x => x._1).
        flatMap(_._2.data).toArray

      val padData =  Array.fill(
          elementsPerBlock * rowDim - unpadData.size)(fieldImpl.zero)

        // insertion is sequential
        val outDM = new DM[F](rows=rowDim, cols=elementsPerBlock,
          data=unpadData ++ padData, offset=0, majorStride=rowDim,
        isTranspose=false)

      val block = ls(0)._1

      List((block, outDM)).iterator
    }

    new DSV(
      repRdd.mapPartitions(partitionMapper, preservesPartitioning = true),
      rowDim
    )
  }

  def fromRDDWithSugg[@specialized (Double, Float) F: ClassTag](suggestedNumPartitions: Int,
    elements : Long, rowDim: Int,
    rdd: RDD[(Long, DV[F])])(implicit fieldImpl: GenericField[F]): DSV[F] =
  {
    val scale = 1.0/suggestedNumPartitions
    val elementsPerBlock = math.max(math.round(scale*elements), 1.0).toInt
    fromRDD(elements, elementsPerBlock, rowDim, rdd)

  }
}

class FomDistStackedDenseVec[@specialized (Double, Float) F: ClassTag] (
  @transient val dsvec : DistributedStackedDenseVectors[F],
  val dotDepth : Int = 2
)(implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
    extends FomDistVec[FomDistStackedDenseVec[F], F] with Serializable {

  type FmDSV = FomDistStackedDenseVec[F]
  type FmBase = FomDistVec[FmDSV, F]

    /* load implicit conversion from FmDDV < -- > FmBase */
  import FomDistStackedDenseVec.implicits._

  def +(b: FmBase): FmBase = {

    new FmDSV(this.dsvec + b.dsvec, this.dotDepth)
  }

   def -(b: FmBase): FmBase = {

    new FmDSV(this.dsvec - b.dsvec, this.dotDepth)
   }

 def *(b: FmBase): FmBase = {

    new FmDSV(this.dsvec * b.dsvec, this.dotDepth)
 }

  def /(b: FmBase): FmBase = {
    new FmDSV(this.dsvec / b.dsvec, this.dotDepth)
  }

  def *(d: F): FmBase = {

    // helper method to multiply vector by a scalar
    val matMul: DM[F] => DM[F] = (m: DM[F]) => {
      val newData = m.data.map(x => fieldImpl.opMul(x, d))
      new DM[F](rows=m.rows, cols=m.cols, data=newData,
                offset=0, majorStride=m.majorStride, isTranspose=m.isTranspose)
    }

    val newVec = this.dsvec.mapBlocks(matMul)
    new FmDSV(newVec, this.dotDepth)

  }

    // helper function to lift a pairwise map
  def liftF(f: (F,F) => F) : (DM[F], DM[F]) => DM[F] =
    (a: DM[F], b: DM[F]) =>
  {
    val asize = a.data.size
    require(a.cols == b.cols, "in liftF a and b must have same number of cols")
    require(a.rows == b.rows, "in liftF a and b must have same number of rows")
    require(a.isTranspose == b.isTranspose, "in liftF a and b must have same major mode")

    val outarray = a.data.zip(b.data).map(x=>f(x._1,x._2))

    new DM(rows=a.rows, cols=a.cols, data=outarray, offset=0,
      majorStride=a.majorStride, isTranspose=a.isTranspose)
  }

  import scala.reflect.runtime.universe._

  // helper function to lift a pairwise map accepting a key
  def keyedLiftF[K](f:  K)(implicit tagK: TypeTag[K], tagF: TypeTag[F])
      : (Int, DM[F], DM[F]) => DM[F] =  {

    // type check & conversion
    val kType = tagK.tpe.toString
    val fType = tagF.tpe.toString
    require(kType == s"(Int, $fType, $fType) => $fType",
      s"""
      For FomDistStackedDenseVec keyedPairWiseFunction needs a function
      (Int, $fType, $fType) => $fType
      """)

    val cleanF = f.asInstanceOf[(Int, F, F) => F]

    // lift 
    (idx: Int, a: DM[F], b: DM[F]) => {
      require(a.cols == b.cols, "in liftF a and b must have same number of cols")
      require(a.rows == b.rows, "in liftF a and b must have same number of rows")
      require(a.isTranspose == b.isTranspose, "in liftF a and b must have same major mode")

      val zipped = a.data.zip(b.data)

      /* idx used as offset; we need to recurse different
       * in major and minor mode
       */
      val computedData = if(!a.isTranspose) zipped.zipWithIndex.map(x =>
        cleanF(idx + x._2/a.rows, x._1._1, x._1._2))
      else
        zipped.zipWithIndex.map(x =>
          cleanF(idx + x._2 % a.cols, x._1._1, x._1._2))

    new DM(rows=a.rows, cols=a.cols, data=computedData, offset=0,
      majorStride=a.majorStride, isTranspose=a.isTranspose)

    }

  }

  def pairWiseFun(b: FmBase)(f: (F,F) =>F) : FmBase = {

    // lift the map f to work between DV[Double]s
    val liftedF = liftF(f)
    val newVec = this.dsvec.mapBlocks(b.dsvec)(liftedF)
    new FmDSV(newVec, this.dotDepth)
  }

  def keyedPairWiseFun[K](b: FmBase)
    (f : K)(implicit tagK: TypeTag[K], tagF: TypeTag[F]) : FmBase = {

    // lift the map f to work between DV[Double]s
    val liftedF = keyedLiftF(f)
    val newVec = this.dsvec.mapBlocksWithKey(b.dsvec)(liftedF)
    new FmDSV(newVec, this.dotDepth)
  }

  def applyElementWise(f: F => F) : FmBase = {

    val liftedF = (x:DM[F]) => new DM[F](rows=x.rows,
      cols=x.cols, data=x.data.map(f), offset=0,
      majorStride=x.majorStride, isTranspose=x.isTranspose)

    val newVec = this.dsvec.mapBlocks(liftedF)

    new FmDSV(newVec, this.dotDepth)    
  }

  def name: String = this.dsvec.self.name

  def copy(): FmBase = new FmDSV(this.dsvec, this.dotDepth)

  def count() : Long = this.dsvec.self.count()

  def dot(other: FmBase) : F = {

    /* distributed dot products
     * we use flatten to fall back to the
     *  vector case
     */
    val distDots = this.dsvec.mapBlocks(other.dsvec)(
      (a: DM[F], b: DM[F]) =>
      new DM[F](rows=1,cols=1,data=
        Array(smoothImpl.opDot(a.flatten(), b.flatten())),
        offset=0,majorStride=1,isTranspose=false)
    ).self.map(x=>x._2.data(0))

    // aggregate the different dot products
    distDots.treeAggregate(fieldImpl.zero)(seqOp = fieldImpl.opAdd(_ , _),
        combOp= fieldImpl.opAdd(_ , _), depth = this.dotDepth)

  }

  def norm(p: Double = 2.0) : F = {

    val power = this.applyElementWise(smoothImpl.opAbsPow(_, p)).
      dsvec.self.map(x => x._2.data.reduce(fieldImpl.opAdd(_, _))).
        treeAggregate(fieldImpl.zero)(seqOp = fieldImpl.opAdd(_ , _),
          combOp= fieldImpl.opAdd(_ , _), depth = this.dotDepth)

      smoothImpl.opAbsPow(power, 1.0/p)
    }

  def setName(name: String): Unit = this.dsvec.self.setName(name)

  def persist() = {
    this.dsvec.self.persist()
    this
  }

  def unpersist(blocking: Boolean = true) = {
    this.dsvec.self.unpersist(blocking)
    this
  }

  def interruptLineage = this.dsvec.self.checkpoint


}

object FomDistStackedDenseVec {

  type DSV[F] = DistributedStackedDenseVectors[F]
  type FmDSV[F] = FomDistStackedDenseVec[F]
  type FmBase[F] = FomDistVec[FmDSV[F], F]

  object implicits {

    implicit def castToFmBase[F](tocast: FmDSV[F]): FmBase[F] = tocast  match {
    case a: FmDSV[F] => a
    case _ => throw new RuntimeException("""
   Failed to cast in FomDistDenseVec.implicits.castToFmBase
   """)
    }

    /* for some reason implementing
       this cast with pattern matching would result in
       hanging up for a
       long time. Resorted to using asInstanceOf
     */
    implicit def castToFmDSV[F](tocast: FmBase[F]): FmDSV[F] =
      tocast.asInstanceOf[FmDSV[F]]
  

  implicit def liftDiffFun[F](diffFun: DistributedDiffFunction[FmDSV[F],F]) :
      DistributedDiffFunction[FmBase[F], F] =
    new DistributedDiffFunction[FmBase[F], F] {

    override def computeValue(x: FmBase[F]): F = diffFun.
      computeValue(x)

    override def computeGrad(x: FmBase[F]): FmBase[F] = diffFun.
      computeGrad(x)

  // should return value & gradient
    def compute(x: FmBase[F]) : (F, FmBase[F]) = 
      diffFun.compute(x)
 
  // hold current batch
  override def holdBatch() : Unit = diffFun.holdBatch()

  // stop holding batch
  override def stopHoldingBatch() :Unit = diffFun.stopHoldingBatch()

    }

  }
}
