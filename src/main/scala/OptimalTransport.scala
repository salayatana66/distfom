package distfom

import org.apache.spark.{Partitioner}
import org.apache.spark.rdd.{RDD, PairRDDFunctions}

import scala.collection.mutable.{Map => MutMap}

import breeze.linalg.{DenseVector => DV, SparseVector => SV,
  DenseMatrix => DM}

import scala.reflect.runtime.universe.{TypeTag, typeTag}

import scala.reflect.{ClassTag, classTag}

import scala.collection.mutable.{Map => MutMap, ArrayBuffer}

object DualSpace extends Logging with Serializable {
  type DDF[T, F] = DistributedDiffFunction[T, F]
  type FDDV[T] = FomDistDenseVec[T]

  /* computations on dual / dual + regularized space,
    * i.e. involving potentials u, v
   */

  type DDV[F] = DistributedDenseVector[F]

  /* Potentials u,v will be stored in the same
   * vector; this helps to keep the partitioning consistent;
   * Ex, Ey are the sample dimensions in the two spaces
   */
  case class PairingContext(Ex: Long, Ey: Long, elementsPerBlock: Int) {
    require(Ex % elementsPerBlock == 0, "Ex must be divisible by elements per block")
    require(Ey % elementsPerBlock == 0, "Ey must be divisible by elements per block")

    // This is the boundary of X/Y space
    val lastXPart = Ex / elementsPerBlock - 1

    // produce a pairing partitioner
    val pairingPartitioner = new PairingPartitioner(
      BlockPartitioner(Ex, elementsPerBlock),
      BlockPartitioner(Ey, elementsPerBlock)
    )

    // partitioner for union of u & v potentials
    val unionPartitioner = new BlockPartitioner(Ex + Ey, elementsPerBlock)

    def mapPair(i: Long, j: Long) =
      (mapX(i), mapY(j))

    def mapX(i: Long) = (i/this.elementsPerBlock).toInt
    def mapY(j: Long, withOffset: Boolean = false) =
      if(withOffset)
      (j/this.elementsPerBlock + this.lastXPart).toInt
      else (j/this.elementsPerBlock).toInt

    def unfoldX(i: Int) = {for {
      j <- Range(0, this.pairingPartitioner.rightPartitioner.
        numPartitions)
    } yield (i, j) }.toArray

    def unfoldY(j: Int) = {for {
      i <- Range(0, this.pairingPartitioner.leftPartitioner.
        numPartitions)
    } yield (i, j) }.toArray

  }

  case class LocalCost[@specialized (Float, Double) F](i: Int, j: Int, cost: F)

  def computePairwiseCosts[@specialized(Float, Double) F](
                                                           leftPoints: RDD[(Long, DV[F])],
                                                           rightPoints: RDD[(Long, DV[F])],
                                                           pairingCtx: PairingContext,
                                                           costFun:
                                                           (DV[F], DV[F]) => F): RDD[((Int, Int), LocalCost[F])] = {

    // unfold the left Points
    val unfoldedX = for {
      point <- leftPoints
      (ix, vec) = point
      iLoc = (ix % pairingCtx.elementsPerBlock).toInt
      pairedIndices <- pairingCtx.unfoldX(pairingCtx.mapX(ix))
    } yield (pairedIndices, (iLoc, vec))

    val repUnfoldedX = unfoldedX.partitionBy(pairingCtx.pairingPartitioner)

    val unfoldedY = for {
      point <- rightPoints
      (jy, vec) = point
      jLoc = (jy % pairingCtx.elementsPerBlock).toInt
      pairedIndices <- pairingCtx.unfoldY(pairingCtx.mapY(jy))
    } yield (pairedIndices, (jLoc, vec))

    val repUnfoldedY = unfoldedY.partitionBy(pairingCtx.pairingPartitioner)

    //local cost computation
    def locCostComp(xpoints: Iterator[((Int, Int), (Int, DV[F]))],
           ypoints: Iterator[((Int, Int), (Int, DV[F]))]):
    Iterator[((Int, Int), LocalCost[F])] = {
      val ixp = xpoints.toList
      // we use map to lookup the points in the same partition
      val myp = ypoints.toList.groupBy(_._1).mapValues(_.map(_._2))

      val costs = for {
        xpoint <- ixp
        (partLookup, (iLoc, xvec)) = xpoint
        yvals = myp(partLookup)
        yval <- yvals
        (jLoc, yvec) = yval
        cost = costFun(xvec, yvec)
      } yield (partLookup, LocalCost(iLoc, jLoc, cost))

      costs.toIterator
    }

    val outCosts = repUnfoldedX.zipPartitions(repUnfoldedY, preservesPartitioning = true)(locCostComp)

    outCosts
  }

  /* compute the value & gradient of the Sinkhorn regularization
   * as the local computational cost is essentially the same whether or
   * not we compute just Loss or Loss & Gradient, at the moment the
   * complete calculation is implemented
   */

  /*
   * Note: to overcome serialization issues sinkRegLoss is a wrapper,
   * where the actual computation takes place in sinkRegLossVectorLevel.
   */
  def sinkRegLossVectorLevel[@specialized(Float, Double) F: ClassTag : TypeTag](pairingContext: PairingContext,
    sinkReg: F, pot: DDV[F], costs: RDD[((Int, Int), LocalCost[F])],
      depth: Int = 2
  )(implicit fieldImpl: GenericField[F],
    zeroValue: breeze.storage.Zero[F]): (F, DDV[F])  = {
    // separate pot in u & v space
    val uPot = pot.self.filter(_._1 <= pairingContext.lastXPart)
    val vPot = pot.self.filter(_._1 > pairingContext.lastXPart)

    // replicate uPot & vPot for distributed scoring
    val uRPot = {
      for {
        elem <- uPot
        (iPart, uVec) = elem
        keyPart <- pairingContext.unfoldX(iPart)
      } yield keyPart -> uVec
    }.
      repartition(pairingContext.pairingPartitioner.numPartitions).
      partitionBy(pairingContext.pairingPartitioner)

    val vRPot = {
      for {
        elem <- vPot
        (jPart, vVec) = elem
        keyPart <- pairingContext.unfoldY(jPart - 1 - pairingContext.lastXPart.toInt)
      } yield keyPart -> vVec
    }.
      repartition(pairingContext.pairingPartitioner.numPartitions).
      partitionBy(pairingContext.pairingPartitioner)

    // type tag to decide how to use the exponential
    val tagF = typeTag[F].tpe.toString

    def localComputer(uIter: Iterator[Tuple2[(Int, Int), DV[F]]],
                      vIter: Iterator[Tuple2[(Int, Int), DV[F]]],
                      costIter: Iterator[Tuple2[(Int, Int), LocalCost[F]]])
    : Iterator[Either[F, (Int, DV[F])]] = {


      // turn uIter, vIter into maps for lookup
      val uMap = uIter.toList.toMap
      val vMap = vIter.toList.toMap

      val costList = costIter.toList
      // check that uMap & vMap hold the same key pairs and generate Map for the
      // output vectors
      require(uMap.keys.toSet == vMap.keys.toSet, "uMap & vMap need to contain the same set of key pairs")

      /* maps to hold partial values of loss & gradient
         note how we need to use a Mutable Map for the loss
         while a Map for vectors as vectors are mutable
       */
      val lossMap = MutMap[(Int, Int), F]()
      for {
        key <- uMap.keys.toSet[Tuple2[Int, Int]]
      } yield (lossMap(key) = fieldImpl.zero)

      // we need to separate contributions to the u & v part
      val uGradMap = uMap.keys.toSet[Tuple2[Int, Int]].map(x =>
        (x, DV.zeros[F](pairingContext.elementsPerBlock))).
        toMap
      val vGradMap = vMap.keys.toSet[Tuple2[Int, Int]].map(x =>
        (x, DV.zeros[F](pairingContext.elementsPerBlock))).
        toMap


      // For the cost we do not need to load in memory
      for {
        cost <- costList
        (keyPart, locCost) = cost

        // lookup the potentials
        uPot = uMap(keyPart)(locCost.i)
        vPot = vMap(keyPart)(locCost.j)

        // constraint violation
        consViolation = fieldImpl.opDiv(
          fieldImpl.opSub(fieldImpl.opAdd(uPot, vPot),
            locCost.cost), sinkReg)

        eViolation = if (tagF == "Float")
          math.exp(consViolation.asInstanceOf[Float]).toFloat.
            asInstanceOf[F]
        else if (tagF == "Double")
          math.exp(consViolation.asInstanceOf[Double]).
            asInstanceOf[F]
        else
          throw new Error("Unimplemented")

        // update the value
        loss = lossMap(keyPart)
        _ = (lossMap.update(keyPart, fieldImpl.opSub(loss,
          fieldImpl.opMul(sinkReg, eViolation))))
        // update the gradients
        uGrad = uGradMap(keyPart)(locCost.i)
        _ = (uGradMap(keyPart)(locCost.i) = fieldImpl.opSub(
          uGrad, eViolation
        ))
        vGrad = vGradMap(keyPart)(locCost.j)
        _ = (vGradMap(keyPart)(locCost.j) = fieldImpl.opSub(
          vGrad, eViolation
        ))

      } yield ()

      /* the local computation will not perform any local reduction
       * for losses: forget the (i,j) index
       */
      val outLosses = lossMap.map(x => Left[F, (Int, DV[F])](x._2)).
        toList
      // for u gradients take only the i index
      val outUGrads = uGradMap.map(x => Right[F, (Int, DV[F])](x._1._1, x._2)).
        toList
      // for v gradients take only the j index; note how we shift the j
      val outVGrads = vGradMap.map(x => Right[F, (Int, DV[F])](
        x._1._2 + pairingContext.lastXPart.toInt + 1, x._2)).
        toList

      (outLosses ++ outUGrads ++ outVGrads).toIterator
    }

    // construct the losses and gradients to aggregrate

    val toAgg = uRPot.zipPartitions(vRPot, costs, preservesPartitioning = true)(
      localComputer
    )

    // use tree aggregate to compute the total loss
    val totalLoss = toAgg.filter(_.isLeft).map(x => x match {
      case Left(x: F) => x
      case _ => throw new Error("did not get an F")
    }).
      treeAggregate(fieldImpl.zero)(
        seqOp = fieldImpl.opAdd, combOp = fieldImpl.opAdd, depth = depth
      )

    // the regularization loss is normalized dividing by Ex * Ey
    val normFactor = fieldImpl.opFromDouble(
      (pairingContext.Ex * pairingContext.Ey).toDouble)
    val rescaledLoss = fieldImpl.opDiv(totalLoss, normFactor)

    // the gradients can now be easily aggregated
    val reducer = (a: DV[F], b: DV[F]) => if (tagF == "Float") {
      (a.asInstanceOf[DV[Float]] + b.asInstanceOf[DV[Float]]).asInstanceOf[DV[F]]
    } else if (tagF == "Double") {
      (a.asInstanceOf[DV[Double]] + b.asInstanceOf[DV[Double]]).asInstanceOf[DV[F]]
    } else throw new Error("unimplemented")

    val weightNormalizer = (a: DV[F]) => if (tagF == "Float")
      (a.asInstanceOf[DV[Float]] / normFactor.asInstanceOf[Float]).
        asInstanceOf[DV[F]]
    else if (tagF == "Double") {
      (a.asInstanceOf[DV[Double]] / normFactor.asInstanceOf[Double]).
        asInstanceOf[DV[F]]
    } else throw new Error("unimplemented")

    // we repartition & aggregate per key
    val rescaledGrad = toAgg.filter(_.isRight).map(x => x match {
      case Right(x: Tuple2[Int, DV[F]]) => x
      case _ => throw new Error("did not get a Tuple2[Int, DV[F]]")
    }).partitionBy(pairingContext.unionPartitioner).
      reduceByKey(reducer).mapValues(weightNormalizer)

    rescaledLoss -> new DDV[F](rescaledGrad)

  }

  def sinkRegLoss[@specialized(Float, Double) F: ClassTag : TypeTag](
           pairingContext: PairingContext,
           sinkReg: F, costs: RDD[((Int, Int), LocalCost[F])],
           depth: Int = 2)(
    implicit fieldImpl: GenericField[F],
    smoothImpl: GenericSmoothField[F],
    zeroValue: breeze.storage.Zero[F]) : DDF[FDDV[F], F] = new DDF[FDDV[F], F] {
    override def compute(x: FDDV[F]): (F, FDDV[F]) = {
      val (loss, grad) = sinkRegLossVectorLevel(pairingContext, sinkReg, x.ddvec, costs, depth)

      val outGrad = new FDDV[F](grad, x.dotDepth)

      loss -> outGrad
    }
  }

  // the potential part of the loss is just the sum of values
  def potentialLossVectorLevel[@specialized(Float, Double) F: ClassTag : TypeTag](
     pot: DDV[F],depth: Int = 2, pairingContext: PairingContext)(
    implicit fieldImpl: GenericField[F],
    zeroValue: breeze.storage.Zero[F]
  ) : (F, DDV[F]) = {

    val tagF = typeTag[F].tpe.toString
    import breeze.linalg.{sum => bSum}

    val sumMapper = (a: DV[F]) => if (tagF == "Float")
      bSum(a.asInstanceOf[DV[Float]]).asInstanceOf[F]
    else if (tagF == "Double")
      bSum(a.asInstanceOf[DV[Double]]).asInstanceOf[F]
    else throw new Error("unimplemented")

    // comput the contributions of u & v
    val uPotSum = pot.self.filter(_._1 <= pairingContext.lastXPart).
      map(x => sumMapper(x._2)).
      treeAggregate(fieldImpl.zero)(fieldImpl.opAdd, fieldImpl.opAdd, depth)
    val uPotLoss = fieldImpl.opDiv(uPotSum, fieldImpl.opFromDouble(pairingContext.Ex.toDouble))

    val vPotSum = pot.self.filter(_._1 > pairingContext.lastXPart).
      map(x => sumMapper(x._2)).
      treeAggregate(fieldImpl.zero)(fieldImpl.opAdd, fieldImpl.opAdd, depth)
    val vPotLoss = fieldImpl.opDiv(vPotSum, fieldImpl.opFromDouble(pairingContext.Ey.toDouble))

    val potLoss = fieldImpl.opAdd(uPotLoss, vPotLoss)

    val gradMapper = (i: Int, a: DV[F]) =>
      new DV(data = Array.fill(a.length)(fieldImpl.opFromDouble(1.0 /
        (if(i <= pairingContext.lastXPart) pairingContext.Ex else
          pairingContext.Ey))),
        offset = 0, stride = 1, length = a.length)

    val gradPot = pot.mapBlocksWithKey(gradMapper)

    potLoss -> gradPot

  }

  def potentialLoss[@specialized(Float, Double) F: ClassTag : TypeTag](
     depth: Int = 2, pairingContext: PairingContext)(
        implicit fieldImpl: GenericField[F],
        smoothImpl: GenericSmoothField[F],
        zeroValue: breeze.storage.Zero[F]): DDF[FDDV[F], F] = new DDF[FDDV[F], F] {

    override def compute(x: FDDV[F]): (F, FDDV[F]) = {
      val (loss, grad) = potentialLossVectorLevel(x.ddvec, depth, pairingContext)

      val outGrad = new FDDV[F](grad, x.dotDepth)

      loss -> outGrad
    }
  }

  // Note as we minimize take a -1 sign
  def potentialLossWSinkhornRegularization[@specialized(Float, Double) F: ClassTag : TypeTag](
           pairingContext: PairingContext,
           sinkReg: F, costs: RDD[((Int, Int), LocalCost[F])],
            depth: Int = 2)(
     implicit fieldImpl: GenericField[F],
     smoothImpl: GenericSmoothField[F],
     zeroValue: breeze.storage.Zero[F]) = {

    val pF = potentialLoss(depth, pairingContext)
    val sF = sinkRegLoss(pairingContext, sinkReg, costs, depth)

    import FomDistDenseVec.implicits._

    DistributedDiffFunction.linCombF[FDDV[F],F](fieldImpl.opFromDouble(-1.0), pF,
      fieldImpl.opFromDouble(-1.0), sF)

  }


}

/******************************************************
  Partitioner for the dual problem formulation
    Indeed i,j are handled individually by a BlockPartitioner partitioner
  and the pair (i, j) is handled by a PairingPartitioner
  ******************************************************/

class PairingPartitioner(val leftPartitioner: BlockPartitioner,
                         val rightPartitioner: BlockPartitioner) extends Partitioner with Serializable {

  override val numPartitions: Int = leftPartitioner.numPartitions *
    rightPartitioner.numPartitions

  override def getPartition(key: Any): Int = {
    key match {
      case Tuple2(i: Int, j: Int) => {
        require(0 <= i && i < leftPartitioner.elements,
          s"left key $i not in range [0, ${leftPartitioner.elements})")

        require(0 <= j && j < rightPartitioner.elements,
          s"right key $j not in range [0, ${rightPartitioner.elements})")

        leftPartitioner.getPartition(i) + leftPartitioner.numPartitions *
          rightPartitioner.getPartition(j)
      }
      case _ =>
        throw new IllegalArgumentException(s"Type of key $key is not Tuple2[Long, Int]")
    }
  }

  override def equals(obj: Any) = obj match {

    case o : PairingPartitioner => (o.leftPartitioner == this.leftPartitioner) &&
      (o.rightPartitioner == this.rightPartitioner)
    case _ => false

  }
}
