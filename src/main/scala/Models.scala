/******************************************************
 * Implementation of Machine Learning models
 * @author Andrea Schioppa
 ****************************************************/

package distfom

import org.apache.spark.{Partitioner}
import org.apache.spark.rdd.{RDD, PairRDDFunctions}

import breeze.linalg.{DenseVector => DV, SparseVector => SV,
  DenseMatrix => DM}

import scala.reflect.runtime.universe.{TypeTag, typeTag}

import scala.reflect.{ClassTag, classTag}

import scala.collection.mutable.{Map => MutMap, ArrayBuffer}

/**
 * Construct scores of Linear Models in a distributed way
 * @todo explore if calling directly Native Fortran Routines is more efficient
 * @todo explore if one can avoid duplication of code between Float & Double
 */
object LinearScorer {

  // Separate implementations for Float & Double
  def applyFloat(feas : DistributedFeatures[Float],
    rddv: RepDDV[Float])(implicit tag: TypeTag[Float]): Scores[Float] = {

    require(feas.pp == rddv.pp, "Partitioners must agree")

    val qp = feas.self.partitioner.
      get.asInstanceOf[ProductPartitioner].leftPartitioner

    // ie are the examples (with sparse vector format
    // iw are the weights (only one existing in the partition
    val liftedF = (ie : Iterator[((Long, Int), SV[Float])],
      iw : Iterator[((Long, Int),DV[Float])]) => {

      val le = ie.toList
      val lw = iw.toList

      val wgt = lw(0)._2

      val partialScores = for {
        ex <- le
        Tuple2(Tuple2(idx, _), svec) = ex

      } yield (idx, svec.dot(wgt))

      partialScores.iterator
    }

    // for each example aggregate across the weight
    // partition keys
    val scores = feas.self.zipPartitions(rddv.self)(liftedF).
      partitionBy(qp).
      reduceByKey( _ + _ )

    new Scores(scores, qp)
  }

  def applyDouble(feas : DistributedFeatures[Double],
    rddv: RepDDV[Double])(implicit tag: TypeTag[Double]): Scores[Double] = {

    require(feas.pp == rddv.pp, "Partitioners must agree")

    val qp = feas.self.partitioner.
      get.asInstanceOf[ProductPartitioner].leftPartitioner

    // ie are the examples (with sparse vector format
    // iw are the weights (only one existing in the partition
    val liftedF = (ie : Iterator[((Long, Int), SV[Double])],
      iw : Iterator[((Long, Int),DV[Double])]) => {

      val le = ie.toList
      val lw = iw.toList

      val wgt = lw(0)._2

      val partialScores = for {
        ex <- le
        Tuple2(Tuple2(idx, _), svec) = ex

      } yield (idx, svec.dot(wgt))

      partialScores.iterator
    }

    // for each example aggregate across the weight
    // partition keys
    val scores = feas.self.zipPartitions(rddv.self)(liftedF).
      partitionBy(qp).
      reduceByKey( _ + _ )

    new Scores(scores, qp)
  }

  def applyFloat(feas : DistributedFeatures[Float],
    rddv: RepSDDV[Float])(implicit tag: TypeTag[Float])
      : Scores[DV[Float]] = {

    require(feas.pp == rddv.pp, "Partitioners must agree")

    val qp = feas.self.partitioner.
      get.asInstanceOf[ProductPartitioner].leftPartitioner

    // ie are the examples (with sparse vector format
    // iw are the weights (only one existing in the partition
    val liftedF = (ie : Iterator[((Long, Int), SV[Float])],
      iw : Iterator[((Long, Int),DM[Float])]) => {

      val le = ie.toList
      val lw = iw.toList

      val wgt = lw(0)._2

      val partialScores = for {
        ex <- le
        Tuple2(Tuple2(idx, _), svec) = ex

      } yield (idx, wgt * svec)

      partialScores.iterator
    }

    // for each example aggregate across the weight
    // partition keys
    val scores = feas.self.zipPartitions(rddv.self)(liftedF).
      partitionBy(qp).
      reduceByKey( _ + _ )

    new Scores(scores, qp)
  }

    def applyDouble(feas : DistributedFeatures[Double],
    rddv: RepSDDV[Double])(implicit tag: TypeTag[Double])
      : Scores[DV[Double]] = {

    require(feas.pp == rddv.pp, "Partitioners must agree")

    val qp = feas.self.partitioner.
      get.asInstanceOf[ProductPartitioner].leftPartitioner

    // ie are the examples (with sparse vector format
    // iw are the weights (only one existing in the partition
    val liftedF = (ie : Iterator[((Long, Int), SV[Double])],
      iw : Iterator[((Long, Int),DM[Double])]) => {

      val le = ie.toList
      val lw = iw.toList

      val wgt = lw(0)._2

      val partialScores = for {
        ex <- le
        Tuple2(Tuple2(idx, _), svec) = ex

      } yield (idx, wgt * svec)

      partialScores.iterator
    }

    // for each example aggregate across the weight
    // partition keys
    val scores = feas.self.zipPartitions(rddv.self)(liftedF).
      partitionBy(qp).
      reduceByKey( _ + _ )

    new Scores(scores, qp)
  }

  def apply[F: TypeTag](feas : DistributedFeatures[F],
    rddv: RepDDV[F]): Scores[F] = {
    val tag = typeTag[F]

    if(tag.tpe.toString == "Float") applyFloat(feas.asInstanceOf[DistributedFeatures[Float]],
      rddv.asInstanceOf[RepDDV[Float]]).asInstanceOf[Scores[F]]
    else if (tag.tpe.toString == "Double") applyDouble(feas.asInstanceOf[DistributedFeatures[Double]],
      rddv.asInstanceOf[RepDDV[Double]]).asInstanceOf[Scores[F]]
    else
      throw new Error("unimplemented")

  }


  def apply[F: TypeTag](feas : DistributedFeatures[F],
    rddv: RepSDDV[F]): Scores[DV[F]] = {
    val tag = typeTag[F]

    if(tag.tpe.toString == "Float") applyFloat(feas.asInstanceOf[DistributedFeatures[Float]],
      rddv.asInstanceOf[RepSDDV[Float]]).asInstanceOf[Scores[DV[F]]]
    else if (tag.tpe.toString == "Double") applyDouble(feas.asInstanceOf[DistributedFeatures[Double]],
      rddv.asInstanceOf[RepSDDV[Double]]).asInstanceOf[Scores[DV[F]]]
    else
      throw new Error("unimplemented")

  }

}

/**
 * Construct scores for Factorization Machine
 * models
 * @todo implement a version just for scoring, i.e. no backpropagation, might be more efficient
 * when producing ranking scores
 * @todo explore using Native Fortran Routines to speed up computations
 */
object FactorizationMachine {

  /** represent a Factorization Machine's score
    * if toBackprop is a Some, it can be sent back
    * to compute gradients wrt to features
   */
  case class FMScore[@specialized (Float, Double) F](
    score: F, toBackProp: Option[Map[(Int,Int), DV[F]]])

  def applyFloat(feas : DistributedFeatures[Float],
    rddv: RepSDDV[Float])(implicit tag: TypeTag[Float])
      : Scores[FMScore[Float]] = {

    require(feas.pp == rddv.pp, "Partitioners must agree")

    val qp = feas.self.partitioner.
      get.asInstanceOf[ProductPartitioner].leftPartitioner

    // ie are the examples (with sparse vector format
    // iw are the weights (only one existing in the partition
    val liftedF = (ie : Iterator[((Long, Int), SV[Float])],
      iw : Iterator[((Long, Int),DM[Float])]) => {

      val le = ie.toList
      val lw = iw.toList

      val wgt = lw(0)._2

      val partialScores = for {
        ex <- le
        Tuple2(Tuple2(idx, fKey), svec) = ex
        locSig = wgt * svec
        // using pow would make svec dense so this
        // is more efficient
        locSig2 = (wgt *:* wgt) * (svec * svec)

        // number of rows = latent factors
        nRows = wgt.rows

        // produce the mutable map for the example's gradient
        baseMap = MutMap[(Int,Int), DV[Float]]()
        _ = for {
          Tuple2(j, v) <- svec.index.zip(svec.data)
        } yield (baseMap((fKey, j)) = wgt(::, j) * v)

      } yield (idx, (locSig, locSig2, baseMap))

      partialScores.iterator
    }

    // the keys are just aggregated by sum
    val keyReducer = (x: (DV[Float], DV[Float], MutMap[(Int, Int), DV[Float]]),
      y: (DV[Float], DV[Float], MutMap[(Int, Int), DV[Float]])) =>
    {
      (x._1 + y._1, x._2 + y._2, x._3 ++ y._3)
    }

    // the score is .5* (squared sum - sum of squares)
    // further reduced along the rows
    // the bookkeeping for back-propagation is just the sum

    val keyMapper = (x: (DV[Float], DV[Float], MutMap[(Int, Int), DV[Float]])) => {
      val score = (.5f * ((x._1*x._1) - x._2)).reduce(_ + _)
      val tobackProp = x._3.mapValues(y => x._1 - y).toMap

      FMScore[Float](score, Some(tobackProp))

    }

    // for each example aggregate across the weight
    // partition keys

    val scores = feas.self.zipPartitions(rddv.self)(liftedF).
      partitionBy(qp).
      reduceByKey(keyReducer).mapValues(keyMapper)

    new Scores(scores, qp)

  }

    def applyDouble(feas : DistributedFeatures[Double],
    rddv: RepSDDV[Double])(implicit tag: TypeTag[Double])
      : Scores[FMScore[Double]] = {

    require(feas.pp == rddv.pp, "Partitioners must agree")

    val qp = feas.self.partitioner.
      get.asInstanceOf[ProductPartitioner].leftPartitioner

    // ie are the examples (with sparse vector format
    // iw are the weights (only one existing in the partition
    val liftedF = (ie : Iterator[((Long, Int), SV[Double])],
      iw : Iterator[((Long, Int),DM[Double])]) => {

      val le = ie.toList
      val lw = iw.toList

      val wgt = lw(0)._2

      val partialScores = for {
        ex <- le
        Tuple2(Tuple2(idx, fKey), svec) = ex
        locSig = wgt * svec
        // using pow would make svec dense so this
        // is more efficient
        locSig2 = (wgt *:* wgt) * (svec * svec)

        // number of rows = latent factors
        nRows = wgt.rows
        // produce the mutable map for the example's gradient
        baseMap = MutMap[(Int, Int), DV[Double]]()
        _ = for {
          Tuple2(j, v) <- svec.index.zip(svec.data)
        } yield((baseMap((fKey, j)) = wgt(::, j) * v))

      } yield (idx, (locSig, locSig2, baseMap))

      partialScores.iterator
    }

    // the keys are just aggregated by sum
    val keyReducer = (x: (DV[Double], DV[Double], MutMap[(Int, Int), DV[Double]]),
      y: (DV[Double], DV[Double], MutMap[(Int, Int), DV[Double]])) =>
    {
      (x._1 + y._1, x._2 + y._2, x._3 ++ y._3)
    }

    // the score is .5* (squared sum - sum of squares)
    // further reduced along the rows
    // the bookkeeping for back-propagation is just the sum

    val keyMapper = (x: (DV[Double], DV[Double], MutMap[(Int, Int), DV[Double]])) => {
      val score = (.5 * ((x._1*x._1) - x._2)).reduce(_ + _)
      val tobackProp = x._3.mapValues(y => x._1 - y).toMap

      FMScore[Double](score, Some(tobackProp))

    }

    // for each example aggregate across the weight
    // partition keys

    val scores = feas.self.zipPartitions(rddv.self)(liftedF).
      partitionBy(qp).
      reduceByKey(keyReducer).mapValues(keyMapper)

    new Scores(scores, qp)

    }

  def apply[F: TypeTag](feas : DistributedFeatures[F],
    rddv: RepSDDV[F]): Scores[FMScore[F]] = {
    val tag = typeTag[F]

    if(tag.tpe.toString == "Float") applyFloat(feas.asInstanceOf[DistributedFeatures[Float]],
      rddv.asInstanceOf[RepSDDV[Float]]).asInstanceOf[Scores[FMScore[F]]]
    else if (tag.tpe.toString == "Double") applyDouble(feas.asInstanceOf[DistributedFeatures[Double]],
      rddv.asInstanceOf[RepSDDV[Double]]).asInstanceOf[Scores[FMScore[F]]]
    else
      throw new Error("unimplemented")

  }

}

/**
 * l2-loss; note how we split the forward & backward
 * computations in two different functions
 * @todo Possibly use generics to deduplicate code across L2, quantile and binary logloss
 */
object L2Loss {

  type DDV[F] = DistributedDenseVector[F]

  def localLossDouble(pred: Double, label: Double, weight: Double,
    autograd: Boolean = true) : (Double, Option[Double]) = {
    val loss = weight * (pred - label) * (pred - label)

    (loss, if(autograd) Some(2 * weight * (pred-label)) else None)
  }

  def localLossFloat(pred: Float, label: Float, weight: Float,
    autograd: Boolean = true) : (Float, Option[Float]) = {
    val loss = weight * (pred - label) * (pred - label)

    (loss, if(autograd) Some(2f * weight * (pred-label)) else None)
  }

  // Computation of losses and gradients to backpropagate
  def apply[E, T, @specialized (Float, Double) F: TypeTag](scores : Scores[F],
    targets: DistributedTargets[E, F, F],
    autograd: Boolean = true) : Losses[F, Tuple2[F, Array[Int]]] = {

    require(scores.self.partitioner.get == targets.self.partitioner.get,
      "distributed labels & scores must have same partitioner")

    // lift the loss scorers to work with iterators
    val liftedF = (ids : Iterator[(Long, F)],
      idt : Iterator[(Long, RichTarget[E, F, F])]) => {

      // get map of scores
      val mds = ids.toMap
      // get list of targets
      val ldt = idt.toList

      // to infer F's type and call the right loss
      val tag = typeTag[F]

      val rolled = for {
        // disassemble everything and lookup the score
        Tuple2(exId, rtgt) <- ldt
        weight = rtgt.weight
        target = rtgt.target
        feas = rtgt.features
        score = mds(exId)

        // compute the final loss & optional gradient
        out = if(tag.tpe.toString == "Float") {
          localLossFloat(pred=score.asInstanceOf[Float],
            label=target.asInstanceOf[Float],
            weight=weight.asInstanceOf[Float], autograd)
        }
        else if (tag.tpe.toString == "Double") {
          localLossDouble(pred=score.asInstanceOf[Double],
            label=target.asInstanceOf[Double],
            weight=weight.asInstanceOf[Double], autograd)
        }
        else throw new Error("unimplemented")

        // this seems to be needed
        loss = out._1.asInstanceOf[F]

        grad = out._2.map(x => (x.asInstanceOf[F], feas))
      } yield (exId, RichLoss(loss, grad))

      rolled.iterator
    }

    val lssRdd = scores.self.zipPartitions(targets.self,
      preservesPartitioning = true)(liftedF)

    new Losses[F, Tuple2[F, Array[Int]]](lssRdd, scores.qp)
  }

  // computation of the forward pass
  def forward[F: TypeTag](losses: Losses[F, Tuple2[F, Array[Int]]], totalWeight: F,
    depth: Int = 2): F = {
    val tag = typeTag[F]

    val agg = if(tag.tpe.toString == "Float") 
     { (losses.self.treeAggregate(0.0f)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Float],
       combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Float]
     } 
    else if (tag.tpe.toString == "Double") {
      (losses.self.treeAggregate(0.0)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Double],
        combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Double]
    }
    else
      throw new Error("unimplemented")

    agg.asInstanceOf[F]
  }

  // computation of the gradient
  def backward[F: TypeTag :ClassTag](losses: Losses[F, Tuple2[F, Array[Int]]],
    features: DistributedFeatures[F], totalWeight: F, depth: Int = 2)
      (implicit fieldImpl: GenericField[F]): DDV[F] = {

    val tag = typeTag[F]
    
    require(losses.qp == features.pp.leftPartitioner, "partitioners must agree")

    // unroll losses to find where the features live
    val unrolledLosses = losses.self.flatMap( ll => {
      require(! ll._2.grad.isEmpty, "Cannot backpropagate without gradients")
      val (gd, fkeys) = ll._2.grad.get

      for {
        k <- fkeys
      } yield ((ll._1, k), gd)
      
    })

    val unrolledLosses2 = unrolledLosses.partitionBy(features.pp)

    // pull out block partitioner for features
    val bp = features.pp.rightPartitioner

    // Local Backpropation for Float
    def localBackProp(iurll: Iterator[((Long, Int), F)],
      ifea: Iterator[((Long, Int), SV[F])]) = {

      // extract list of gradients
      val miurll = iurll.toMap
      val ifeal = ifea.toList

      // we now do local reduction of the gradients
      
      // construct all possible empty vectors needed
      val zeroVecs =  ifeal.map(_._1._2).distinct.
        map(x => (x, {
          if(tag.tpe.toString == "Float") {
          DV.zeros[Float](bp.elementsPerBlock)
        } else if (tag.tpe.toString == "Double") {
          DV.zeros[Double](bp.elementsPerBlock)
          } else throw new Error("unimplemented")
        }.asInstanceOf[DV[F]] ))

      val outVecs = MutMap(zeroVecs :_*)

      val rolled = for {
        ex <- ifeal

        /* get example id, fea partition
         * and features
         */
        ((eidx, feak), svec) = ex
        // get loss gradient
        gd = miurll((eidx, feak))

        // the feature vec gets multiplied by the gradient
        _  = if(tag.tpe.toString == "Float") {
          val outvec = svec.asInstanceOf[SV[Float]] * gd.asInstanceOf[Float]
          outVecs(feak).asInstanceOf[DV[Float]] += outvec
        } else if(tag.tpe.toString == "Double") {
          val outvec = svec.asInstanceOf[SV[Double]] * gd.asInstanceOf[Double]
          outVecs(feak).asInstanceOf[DV[Double]] += outvec
        }
        else throw new Error("unimplemented")

      } yield ()

      outVecs.iterator
    }
      
          /** reduce the sparse vector gradients
            * across examples and puts them back into a dense vector
            * reduceByKey works inside each partition; hence we need to
            * repartition via the BlockPartitioner before aggregating
            * for scalability we also to a first aggregation locally
           */

    val reducer = (a: DV[F], b: DV[F]) => if(tag.tpe.toString == "Float") {
      (a.asInstanceOf[DV[Float]] + b.asInstanceOf[DV[Float]]).asInstanceOf[DV[F]]
    } else if(tag.tpe.toString == "Double") {
      (a.asInstanceOf[DV[Double]] + b.asInstanceOf[DV[Double]]).asInstanceOf[DV[F]]
    } else throw new Error("unimplemented")

    val weightNormalizer = (a: DV[F]) => if(tag.tpe.toString == "Float")
      a.asInstanceOf[DV[Float]]/totalWeight.asInstanceOf[Float]
    else if (tag.tpe.toString == "Double") {
      a.asInstanceOf[DV[Double]]/totalWeight.asInstanceOf[Double]
    } else throw new Error("unimplemented")

    val vecGd =  unrolledLosses2.zipPartitions(features.self,
      preservesPartitioning = true)(localBackProp).
      partitionBy(bp).reduceByKey(reducer).
      mapValues(x => weightNormalizer(x.asInstanceOf[DV[F]]).asInstanceOf[DV[F]])

    new DDV[F](vecGd)
  }
}

object QuantileLoss {

  type DDV[F] = DistributedDenseVector[F]

  def localLossDouble(pred: Double, label: Double, weight: Double,
                      quantile: Double, autograd: Boolean = true) : (Double, Option[Double]) = {
    val loss = if(pred - label < 0) weight * (quantile - 1) * (pred - label)
    else weight * quantile * (pred - label)

    val grad = if(!autograd) None
    else if(pred - label >= 0) Some(weight * quantile)
    else Some(weight * (quantile - 1))

    loss -> grad
  }

  def localLossFloat(pred: Float, label: Float, weight: Float,
                     quantile: Float, autograd: Boolean = true) : (Float, Option[Float]) = {
    val loss = if(pred - label < 0) weight * (quantile - 1) * (pred - label)
    else weight * quantile * (pred - label)

    val grad = if(!autograd) None
    else if(pred - label >= 0) Some(weight * quantile)
    else Some(weight * (quantile - 1))

    loss -> grad
  }

  // computation of the forward pass
  def forward[F: TypeTag](losses: Losses[F, Tuple2[F, Array[Int]]], totalWeight: F,
                          depth: Int = 2): F = {
    val tag = typeTag[F]

    val agg = if(tag.tpe.toString == "Float")
    { (losses.self.treeAggregate(0.0f)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Float],
      combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Float]
    }
    else if (tag.tpe.toString == "Double") {
      (losses.self.treeAggregate(0.0)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Double],
        combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Double]
    }
    else
      throw new Error("unimplemented")

    agg.asInstanceOf[F]
  }

  // computation of the gradient
  def backward[F: TypeTag :ClassTag](losses: Losses[F, Tuple2[F, Array[Int]]],
                                     features: DistributedFeatures[F], totalWeight: F, depth: Int = 2)
                                    (implicit fieldImpl: GenericField[F]): DDV[F] = {

    val tag = typeTag[F]

    require(losses.qp == features.pp.leftPartitioner, "partitioners must agree")

    // unroll losses to find where the features live
    val unrolledLosses = losses.self.flatMap( ll => {
      require(ll._2.grad.isDefined, "Cannot backpropagate without gradients")
      val (gd, fkeys) = ll._2.grad.get

      for {
        k <- fkeys
      } yield ((ll._1, k), gd)

    })

    val unrolledLosses2 = unrolledLosses.partitionBy(features.pp)

    // pull out block partitioner for features
    val bp = features.pp.rightPartitioner

    // Local Backpropation for Float
    def localBackProp(iurll: Iterator[((Long, Int), F)],
                      ifea: Iterator[((Long, Int), SV[F])]) = {

      // extract list of gradients
      val miurll = iurll.toMap
      val ifeal = ifea.toList

      // we now do local reduction of the gradients

      // construct all possible empty vectors needed
      val zeroVecs =  ifeal.map(_._1._2).distinct.
        map(x => (x, {
          if(tag.tpe.toString == "Float") {
            DV.zeros[Float](bp.elementsPerBlock)
          } else if (tag.tpe.toString == "Double") {
            DV.zeros[Double](bp.elementsPerBlock)
          } else throw new Error("unimplemented")
          }.asInstanceOf[DV[F]] ))

      val outVecs = MutMap(zeroVecs :_*)

      val rolled = for {
        ex <- ifeal

        /* get example id, fea partition
         * and features
         */
        ((eidx, feak), svec) = ex
        // get loss gradient
        gd = miurll((eidx, feak))

        // the feature vec gets multiplied by the gradient
        _  = if(tag.tpe.toString == "Float") {
          val outvec = svec.asInstanceOf[SV[Float]] * gd.asInstanceOf[Float]
          outVecs(feak).asInstanceOf[DV[Float]] += outvec
        } else if(tag.tpe.toString == "Double") {
          val outvec = svec.asInstanceOf[SV[Double]] * gd.asInstanceOf[Double]
          outVecs(feak).asInstanceOf[DV[Double]] += outvec
        }
        else throw new Error("unimplemented")

      } yield ()

      outVecs.iterator
    }

    /** reduce the sparse vector gradients
      * across examples and puts them back into a dense vector
      * reduceByKey works inside each partition; hence we need to
      * repartition via the BlockPartitioner before aggregating
      * for scalability we also to a first aggregation locally
     */

    val reducer = (a: DV[F], b: DV[F]) => if(tag.tpe.toString == "Float") {
      (a.asInstanceOf[DV[Float]] + b.asInstanceOf[DV[Float]]).asInstanceOf[DV[F]]
    } else if(tag.tpe.toString == "Double") {
      (a.asInstanceOf[DV[Double]] + b.asInstanceOf[DV[Double]]).asInstanceOf[DV[F]]
    } else throw new Error("unimplemented")

    val weightNormalizer = (a: DV[F]) => if(tag.tpe.toString == "Float")
      a.asInstanceOf[DV[Float]]/totalWeight.asInstanceOf[Float]
    else if (tag.tpe.toString == "Double") {
      a.asInstanceOf[DV[Double]]/totalWeight.asInstanceOf[Double]
    } else throw new Error("unimplemented")

    val vecGd =  unrolledLosses2.zipPartitions(features.self,
      preservesPartitioning = true)(localBackProp).
      partitionBy(bp).reduceByKey(reducer).
      mapValues(x => weightNormalizer(x.asInstanceOf[DV[F]]).asInstanceOf[DV[F]])

    new DDV[F](vecGd)
  }

  // Computation of losses and gradients to backpropagate
  def apply[E, @specialized (Float, Double) F: TypeTag](scores : Scores[F],
                                                           targets: DistributedTargets[E, F, F],
                                                           quantile: F,
                                                           autograd: Boolean = true) : Losses[F, Tuple2[F, Array[Int]]] = {

    require(scores.self.partitioner.get == targets.self.partitioner.get,
      "distributed labels & scores must have same partitioner")

    // lift the loss scorers to work with iterators
    val liftedF = (ids : Iterator[(Long, F)],
                   idt : Iterator[(Long, RichTarget[E, F, F])]) => {

      // get map of scores
      val mds = ids.toMap
      // get list of targets
      val ldt = idt.toList

      // to infer F's type and call the right loss
      val tag = typeTag[F]

      val rolled = for {
        // disassemble everything and lookup the score
        Tuple2(exId, rtgt) <- ldt
        weight = rtgt.weight
        target = rtgt.target
        feas = rtgt.features
        score = mds(exId)

        // compute the final loss & optional gradient
        out = if(tag.tpe.toString == "Float") {
          localLossFloat(pred=score.asInstanceOf[Float],
            label=target.asInstanceOf[Float],
            weight.asInstanceOf[Float],
            quantile.asInstanceOf[Float], autograd)
        }
        else if (tag.tpe.toString == "Double") {
          localLossDouble(pred=score.asInstanceOf[Double],
            label=target.asInstanceOf[Double],
            weight=weight.asInstanceOf[Double],
            quantile.asInstanceOf[Double], autograd)
        }
        else throw new Error("unimplemented")

        // this seems to be needed
        loss = out._1.asInstanceOf[F]

        grad = out._2.map(x => (x.asInstanceOf[F], feas))
      } yield (exId, RichLoss(loss, grad))

      rolled.iterator
    }

    val lssRdd = scores.self.zipPartitions(targets.self,
      preservesPartitioning = true)(liftedF)

    new Losses[F, Tuple2[F, Array[Int]]](lssRdd, scores.qp)
  }

}

object BinLogLoss {

  type DDV[F] = DistributedDenseVector[F]

  def localLossDouble(raw_score: Double, label: Boolean, weight: Double,
                      autograd: Boolean = true) : (Double, Option[Double]) = {

    // probability of positive class
    val prob = 1.0 / (1.0 + scala.math.exp(-raw_score))
    val loss = if(label) -weight * scala.math.log(prob + 1e-24)
    else - weight * scala.math.log(1-prob + 1e-24)

    val grad = if(!autograd) None
    else if(label) Some(- weight * (1-prob))
    else Some(weight * prob)

    loss -> grad
  }

  def localLossFloat(raw_score: Float, label: Boolean, weight: Float,
                      autograd: Boolean = true) : (Float, Option[Float]) = {

    // probability of positive class
    val prob = 1.0f / (1.0f + scala.math.exp(-raw_score).toFloat)
    val loss = if(label) -weight * scala.math.log(prob).toFloat
    else - weight * scala.math.log(1-prob).toFloat

    val grad = if(!autograd) None
    else if(label) Some(- weight * (1-prob))
    else Some(weight * prob)

    loss -> grad
  }

  // computation of the forward pass
  def forward[F: TypeTag](losses: Losses[F, Tuple2[F, Array[Int]]], totalWeight: F,
                          depth: Int = 2): F = {
    val tag = typeTag[F]

    val agg = if(tag.tpe.toString == "Float")
    { (losses.self.treeAggregate(0.0f)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Float],
      combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Float]
    }
    else if (tag.tpe.toString == "Double") {
      (losses.self.treeAggregate(0.0)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Double],
        combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Double]
    }
    else
      throw new Error("unimplemented")

    agg.asInstanceOf[F]
  }

  // computation of the gradient
  def backward[F: TypeTag :ClassTag](losses: Losses[F, Tuple2[F, Array[Int]]],
                                     features: DistributedFeatures[F], totalWeight: F, depth: Int = 2)
                                    (implicit fieldImpl: GenericField[F]): DDV[F] = {

    val tag = typeTag[F]

    require(losses.qp == features.pp.leftPartitioner, "partitioners must agree")

    // unroll losses to find where the features live
    val unrolledLosses = losses.self.flatMap( ll => {
      require(ll._2.grad.isDefined, "Cannot backpropagate without gradients")
      val (gd, fkeys) = ll._2.grad.get

      for {
        k <- fkeys
      } yield ((ll._1, k), gd)

    })

    val unrolledLosses2 = unrolledLosses.partitionBy(features.pp)

    // pull out block partitioner for features
    val bp = features.pp.rightPartitioner

    // Local Backpropation for Float
    def localBackProp(iurll: Iterator[((Long, Int), F)],
                      ifea: Iterator[((Long, Int), SV[F])]) = {

      // extract list of gradients
      val miurll = iurll.toMap
      val ifeal = ifea.toList

      // we now do local reduction of the gradients

      // construct all possible empty vectors needed
      val zeroVecs =  ifeal.map(_._1._2).distinct.
        map(x => (x, {
          if(tag.tpe.toString == "Float") {
            DV.zeros[Float](bp.elementsPerBlock)
          } else if (tag.tpe.toString == "Double") {
            DV.zeros[Double](bp.elementsPerBlock)
          } else throw new Error("unimplemented")
          }.asInstanceOf[DV[F]] ))

      val outVecs = MutMap(zeroVecs :_*)

      val rolled = for {
        ex <- ifeal

        /* get example id, fea partition
         * and features
         */
        ((eidx, feak), svec) = ex
        // get loss gradient
        gd = miurll((eidx, feak))

        // the feature vec gets multiplied by the gradient
        _  = if(tag.tpe.toString == "Float") {
          val outvec = svec.asInstanceOf[SV[Float]] * gd.asInstanceOf[Float]
          outVecs(feak).asInstanceOf[DV[Float]] += outvec
        } else if(tag.tpe.toString == "Double") {
          val outvec = svec.asInstanceOf[SV[Double]] * gd.asInstanceOf[Double]
          outVecs(feak).asInstanceOf[DV[Double]] += outvec
        }
        else throw new Error("unimplemented")

      } yield ()

      outVecs.iterator
    }

    /** reduce the sparse vector gradients
      * across examples and puts them back into a dense vector
      * reduceByKey works inside each partition; hence we need to
      * repartition via the BlockPartitioner before aggregating
      * for scalability we also to a first aggregation locally
     */

    val reducer = (a: DV[F], b: DV[F]) => if(tag.tpe.toString == "Float") {
      (a.asInstanceOf[DV[Float]] + b.asInstanceOf[DV[Float]]).asInstanceOf[DV[F]]
    } else if(tag.tpe.toString == "Double") {
      (a.asInstanceOf[DV[Double]] + b.asInstanceOf[DV[Double]]).asInstanceOf[DV[F]]
    } else throw new Error("unimplemented")

    val weightNormalizer = (a: DV[F]) => if(tag.tpe.toString == "Float")
      a.asInstanceOf[DV[Float]]/totalWeight.asInstanceOf[Float]
    else if (tag.tpe.toString == "Double") {
      a.asInstanceOf[DV[Double]]/totalWeight.asInstanceOf[Double]
    } else throw new Error("unimplemented")

    val vecGd =  unrolledLosses2.zipPartitions(features.self,
      preservesPartitioning = true)(localBackProp).
      partitionBy(bp).reduceByKey(reducer).
      mapValues(x => weightNormalizer(x.asInstanceOf[DV[F]]).asInstanceOf[DV[F]])

    new DDV[F](vecGd)
  }

  def apply[E, @specialized (Float, Double) F: TypeTag](scores : Scores[F],
                                                        targets: DistributedTargets[E, Boolean, F],
                                                        autograd: Boolean = true) : Losses[F, Tuple2[F, Array[Int]]] = {

    require(scores.self.partitioner.get == targets.self.partitioner.get,
      "distributed labels & scores must have same partitioner")

    // lift the loss scorers to work with iterators
    val liftedF = (ids : Iterator[(Long, F)],
                   idt : Iterator[(Long, RichTarget[E, Boolean, F])]) => {

      // get map of scores
      val mds = ids.toMap
      // get list of targets
      val ldt = idt.toList

      // to infer F's type and call the right loss
      val tag = typeTag[F]

      val rolled = for {
        // disassemble everything and lookup the score
        Tuple2(exId, rtgt) <- ldt
        weight = rtgt.weight
        target = rtgt.target
        feas = rtgt.features
        score = mds(exId)

        // compute the final loss & optional gradient
        out = if(tag.tpe.toString == "Float") {
          localLossFloat(raw_score=score.asInstanceOf[Float],
            label=target, weight.asInstanceOf[Float],
            autograd)
        }
        else if (tag.tpe.toString == "Double") {
          localLossDouble(raw_score=score.asInstanceOf[Double],
            label=target, weight=weight.asInstanceOf[Double],
             autograd)
        }
        else throw new Error("unimplemented")

        // this seems to be needed
        loss = out._1.asInstanceOf[F]

        grad = out._2.map(x => (x.asInstanceOf[F], feas))
      } yield (exId, RichLoss(loss, grad))

      rolled.iterator
    }

    val lssRdd = scores.self.zipPartitions(targets.self,
      preservesPartitioning = true)(liftedF)

    new Losses[F, Tuple2[F, Array[Int]]](lssRdd, scores.qp)
  }

}

/**
 * Multi-classification
 */
object MultiLabelSoftMaxLogLoss {

  type DSV[F] = DistributedStackedDenseVectors[F]

  /** Note we assume targetWeights sum up to 1.0
   * to avoid having to adjust the totalWeight returned
   * by Batchers
   */
 case class MultiLabelTarget[@specialized (Float, Double) F]
   (targetIds: Array[Int], targetWeights: Array[F]) {
   require(targetIds.length == targetWeights.length,
     "targetIdx & targetWeights lengths must match")
 }

  /** numerically stable softmax
   * we compute the full gradient matrix
   * g_ij = \partial p_i / \partial s_j
   * as in practice we expect a smallish <= 100 number of labels
   */
  def safeSoftmaxDouble(raw_scores: DV[Double], autograd: Boolean = true):
    (DV[Double], Option[DM[Double]]) = {

    import breeze.linalg.{max => bMax, sum => bSum}

    // apparently breeze lacks a native exponential method
    val scores_max = bMax(raw_scores)
    val exp_scores = new DV[Double](data=raw_scores.data.map(x =>
      scala.math.exp(x - scores_max)),
      offset=0, stride=1, length=raw_scores.length)
    val soft_scores = exp_scores / bSum(exp_scores)

    var grad: DM[Double] = null

    if(autograd) {
      grad = DM.zeros(rows=soft_scores.length, cols=soft_scores.length)

      // put p_i on the diagonal
      for(i <- 0 until soft_scores.length) grad(i, i) += soft_scores(i)
      // update off the diagonal
      for {
        i <- 0 until soft_scores.length
        k <- 0 until soft_scores.length
      } yield (grad(i, k) -= soft_scores(i) * soft_scores(k))
    }
    
      if(autograd) {
        (soft_scores, Some(grad))
      } else (soft_scores, None)
  }

  def safeSoftmaxFloat(raw_scores: DV[Float], autograd: Boolean = true):
    (DV[Float], Option[DM[Float]]) = {

    import breeze.linalg.{max => bMax, sum => bSum}

    // apparently breeze lacks a native exponential method
    val scores_max = bMax(raw_scores)
    val exp_scores = new DV[Float](data=raw_scores.data.map(x =>
      scala.math.exp(x - scores_max).toFloat),
      offset=0, stride=1, length=raw_scores.length)
    val soft_scores = exp_scores / bSum(exp_scores)

    var grad: DM[Float] = null

    if(autograd) {
      grad = DM.zeros(rows=soft_scores.length, cols=soft_scores.length)

      // put p_i on the diagonal
      for(i <- 0 until soft_scores.length) grad(i, i) += soft_scores(i)
      // update off the diagonal
      for {
        i <- 0 until soft_scores.length
        k <- 0 until soft_scores.length
      } yield (grad(i, k) -= soft_scores(i) * soft_scores(k))
    }
    
      if(autograd) {
        (soft_scores, Some(grad))
      } else (soft_scores, None)
  }


  // preds are already after the softmax
  def localLossDouble(pred: DV[Double], label: MultiLabelTarget[Double],
    weight: Double, backpropGrad: Option[DM[Double]],
    autograd: Boolean = true) : (Double, Option[DV[Double]]) = {

    if(autograd) require(!backpropGrad.isEmpty,
      """With autograd softmax gradient must be supplied"""
    )

    // check labels are in the right range
    for {
      tgtId <- label.targetIds
    } yield require((tgtId >= 0) & (tgtId < pred.length),
      s"target ids must like in [0, $pred.length)")

    // loss
    var loss = 0.0
    for {
      (tid, tw) <- label.targetIds.zip(label.targetWeights)
    } yield (loss -= weight * tw * scala.math.log(pred(tid)+1e-24))
    
    var grad : DV[Double] = null
    if(autograd) {
      grad = DV.zeros[Double](backpropGrad.get.rows)

      for {
        k <- 0 until backpropGrad.get.rows
        (tid, tw) <- label.targetIds.zip(label.targetWeights)

        // contribution to the gradient
        gradContrib = weight * tw / (pred(tid) + 1e-24) *
        backpropGrad.get(tid, k)
      } yield (grad(k) -= gradContrib)

    }

    if(autograd) {
        (loss, Some(grad))
      } else (loss, None)
  }

    // preds are already after the softmax
  def localLossFloat(pred: DV[Float], label: MultiLabelTarget[Float],
    weight: Float, backpropGrad: Option[DM[Float]],
    autograd: Boolean = true) : (Float, Option[DV[Float]]) = {

    if(autograd) require(!backpropGrad.isEmpty,
      """With autograd softmax gradient must be supplied"""
    )

    // check labels are in the right range
    for {
      tgtId <- label.targetIds
    } yield require((tgtId >= 0) & (tgtId < pred.length),
      s"target ids must like in [0, $pred.length)")

    // loss
    var loss = 0.0f
    for {
      (tid, tw) <- label.targetIds.zip(label.targetWeights)
    } yield (loss -= weight * tw * scala.math.log(pred(tid)+1e-24).toFloat)
    
    var grad : DV[Float] = null
    if(autograd) {
      grad = DV.zeros[Float](backpropGrad.get.rows)

      for {
        k <- 0 until backpropGrad.get.rows
        (tid, tw) <- label.targetIds.zip(label.targetWeights)

        // contribution to the gradient
        gradContrib = weight * tw / (pred(tid) + 1e-24f) *
        backpropGrad.get(tid, k)
      } yield (grad(k) -= gradContrib)

    }

    if(autograd) {
        (loss, Some(grad))
      } else (loss, None)
  }


  // computation of the forward pass
  def forward[F: TypeTag](losses: Losses[F, Tuple2[DV[F], Array[Int]]], totalWeight: F,
    depth: Int = 2): F = {
    val tag = typeTag[F]

    val agg = if(tag.tpe.toString == "Float") 
     { (losses.self.treeAggregate(0.0f)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Float],
       combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Float]
     } 
    else if (tag.tpe.toString == "Double") {
      (losses.self.treeAggregate(0.0)(seqOp= (x, y) => x + y._2.loss.asInstanceOf[Double],
        combOp= (x, y) => x+y, depth=depth)) / totalWeight.asInstanceOf[Double]
    }
    else
      throw new Error("unimplemented")

    agg.asInstanceOf[F]
  }

    // computation of the gradient
  def backward[F: TypeTag :ClassTag](losses: Losses[F, Tuple2[DV[F], Array[Int]]],
    features: DistributedFeatures[F], totalWeight: F, rowDim: Int, depth: Int = 2)
      (implicit fieldImpl: GenericField[F]): DSV[F] = {

    val tag = typeTag[F]
    
    require(losses.qp == features.pp.leftPartitioner, "partitioners must agree")

    // unroll losses to find where the features live
    val unrolledLosses = losses.self.flatMap( ll => {
      require(! ll._2.grad.isEmpty, "Cannot backpropagate without gradients")
      val (gd, fkeys) = ll._2.grad.get

      for {
        k <- fkeys
      } yield ((ll._1, k), gd)
      
    })

    val unrolledLosses2 = unrolledLosses.partitionBy(features.pp)

    // pull out block partitioner for features
    val bp = features.pp.rightPartitioner

    // Local Backpropation for Float
    //@todo other ways / better ways to consume the iterators?
    def localBackProp(iurll: Iterator[((Long, Int), DV[F])],
      ifea: Iterator[((Long, Int), SV[F])]) = {


      // extract list of gradients
      val miurll = iurll.toMap
      val ifeal = ifea.toList

      // we now do local reduction of the gradients
      
      // construct all possible empty matrices needed
      val zeroMats =  ifeal.map(_._1._2).distinct.
        map(x => (x, {
          if(tag.tpe.toString == "Float") {
          DM.zeros[Float](rows=rowDim, cols=bp.elementsPerBlock)
        } else if (tag.tpe.toString == "Double") {
          DM.zeros[Double](rows=rowDim, cols=bp.elementsPerBlock)
          } else throw new Error("unimplemented")
        }.asInstanceOf[DM[F]] ))

      val outMats = MutMap(zeroMats :_*)

      /** define update of outMats
       *  feak -> index of feature
       *  svec -> SV[F] of features
       *  gd -> DV[F] the gradient backprop
       *  we update at feak with gd(along row) * svec (along column)
       */
      def updateOutMats(feak: Int, svec: SV[F], gd: DV[F]): Unit = {
        for {
          i <- 0 until rowDim
          (j, v) <- svec.index.zip(svec.data)

        } yield {
          if(tag.tpe.toString == "Float") {
            val currVal = outMats(feak)(i, j).asInstanceOf[Float]
            val contrib = gd(i).asInstanceOf[Float] * v.asInstanceOf[Float]
            // with DenseMatrix we must use the update method
            outMats(feak).update(i, j, (currVal + contrib).
              asInstanceOf[F]
            )
          } else if (tag.tpe.toString == "Double") {
            val currVal = outMats(feak)(i, j).asInstanceOf[Double]
            val contrib = gd(i).asInstanceOf[Double] * v.asInstanceOf[Double]
            // with DenseMatrix we must use the update method
            outMats(feak).update(i, j, (currVal + contrib).
              asInstanceOf[F]
            )
          } else throw new Error("unimplemented")
        }
      }

      val rolled = for {
        ex <- ifeal

        /** get example id, fea partition
         * and features
         */
        ((eidx, feak), svec) = ex
        // get loss gradient
        gd = miurll((eidx, feak))

        // the feature vec gets multiplied by the gradient
        _  = updateOutMats(feak, svec, gd)

      } yield ()

      outMats.iterator
    }
      
          /** reduce the sparse vector gradients
           * across examples and puts them back into
           * a dense vector
           * reduceByKey works inside each partition;
           * hence we need to repartition via the
           * BlockPartitioner before aggregating
           * for scalability we also to a first
           * aggregation locally
           */

    val reducer = (a: DM[F], b: DM[F]) => if(tag.tpe.toString == "Float") {
      (a.asInstanceOf[DM[Float]] + b.asInstanceOf[DM[Float]]).asInstanceOf[DM[F]]
    } else if(tag.tpe.toString == "Double") {
      (a.asInstanceOf[DM[Double]] + b.asInstanceOf[DM[Double]]).asInstanceOf[DM[F]]
    } else throw new Error("unimplemented")

    val weightNormalizer = (a: DM[F]) => if(tag.tpe.toString == "Float")
      a.asInstanceOf[DM[Float]] / totalWeight.asInstanceOf[Float]
    else if (tag.tpe.toString == "Double") {
      a.asInstanceOf[DM[Double]] / totalWeight.asInstanceOf[Double]
    } else throw new Error("unimplemented")

    val matGd =  unrolledLosses2.zipPartitions(features.self,
      preservesPartitioning = true)(localBackProp).
      partitionBy(bp).reduceByKey(reducer).
      mapValues(x => weightNormalizer(x.asInstanceOf[DM[F]]).asInstanceOf[DM[F]])

    new DSV[F](matGd, rowDim=rowDim)
  }

    // Computation of losses and gradients to backpropagate
  def apply[E, T, @specialized (Float, Double) F: TypeTag](scores : Scores[DV[F]],
    targets: DistributedTargets[E, MultiLabelTarget[F], F],
    autograd: Boolean = true) : Losses[F, Tuple2[DV[F], Array[Int]]] = {

    require(scores.self.partitioner.get == targets.self.partitioner.get,
      "distributed labels & scores must have same partitioner")

    // lift the loss scorers to work with iterators
    val liftedF = (ids : Iterator[(Long, DV[F])],
      idt : Iterator[(Long, RichTarget[E, MultiLabelTarget[F], F])]) => {

      // get map of scores
      val mds = ids.toMap
      // get list of targets
      val ldt = idt.toList

      // to infer F's type and call the right loss
      val tag = typeTag[F]

      val rolled = for {
        // disassemble everything and lookup the score
        Tuple2(exId, rtgt) <- ldt
        weight = rtgt.weight
        target = rtgt.target
        feas = rtgt.features
        score = mds(exId)

        // compute the final loss & optional gradient
        out = if(tag.tpe.toString == "Float") {
          // compute on the softmax
          val (sfmaxPred, sfmaxGrad) = safeSoftmaxFloat(
            raw_scores=score.asInstanceOf[DV[Float]],
            autograd=autograd)

          // compute on the logloss
          val (loglossPred, loglossGrad) =
            localLossFloat(pred=sfmaxPred, label=target.asInstanceOf[MultiLabelTarget[Float]],
              weight=weight.asInstanceOf[Float], backpropGrad=sfmaxGrad,
              autograd=autograd)

          (loglossPred, loglossGrad)
        }
        else if(tag.tpe.toString == "Double") {
          // compute on the softmax
          val (sfmaxPred, sfmaxGrad) = safeSoftmaxDouble(
            raw_scores=score.asInstanceOf[DV[Double]],
            autograd=autograd)

          // compute on the logloss
          val (loglossPred, loglossGrad) =
            localLossDouble(pred=sfmaxPred, label=target.asInstanceOf[MultiLabelTarget[Double]],
              weight=weight.asInstanceOf[Double], backpropGrad=sfmaxGrad,
              autograd=autograd)

          (loglossPred, loglossGrad)
        }
        else throw new Error("unimplemented")

        // this conversion seems to be needed
        loss = out._1.asInstanceOf[F]

        grad = out._2.map(x => (x.asInstanceOf[DV[F]], feas))
      } yield (exId, RichLoss(loss, grad))

      rolled.iterator
    }

    val lssRdd = scores.self.zipPartitions(targets.self,
      preservesPartitioning = true)(liftedF)

    new Losses[F, Tuple2[DV[F], Array[Int]]](lssRdd, scores.qp)
  }

}

/**
 * Ranking with Factorization Machines and implicit feedback
 */
object SigmoidPairedRanking extends Logging with Serializable {

  // case objects for implicit feedback
  sealed trait FeedBackType
  case object PositiveFeedBack extends FeedBackType
  case object NegativeFeedBack extends FeedBackType

  /** data we need to record for positive/negative examples
   * e.g.
   * PositiveExamples => (PositiveLabels, PositiveScores)
   * implicit label as we use implicit feedback
   */
  case class ImplicitLabel[E, @specialized (Double, Float) F](
    exId: E, feedback: FeedBackType, relWeight: F, weight: F
  )

  // represents a pair of label and score and keeps the example which generated it
  case class JoinedFeedScore[E, F](
    label: ImplicitLabel[E, F], score: F, exId: Long)

  // represents gradient info for bakpropagation
  case class Grad[F](positive: Array[(Long, F)], negative: Array[(Long, F)])
  // represents loss info
  case class Loss[F](loss: F, weight: F)

  // helper class which allows to accumulate losses & weights
  class LossBuffer[F](var loss: F, var weight :F) extends Serializable {
    def toLoss: Loss[F] = Loss(this.loss, this.weight)
  }

  // to accumulate gradients it is more trick, we use a Buffer
  import scala.collection.mutable.ArrayBuffer

  case class GradBuffer[F: TypeTag](positive: ArrayBuffer[(Long, F)],
    negative: ArrayBuffer[(Long, F)]) {
    def toGrad: Grad[F] = {
      val tag = typeTag[F]
      require(List("Float", "Double").contains(tag.tpe.toString),
        "Only Float & Double are supported")
      // the positive gets reduced after a uniqueness check
      require(this.positive.size > 0, "There must be a gradient for positive term")
      val firstIdx = this.positive.head._1
      // check all indices equal to positive
      this.positive.map(x => require(x._1 == firstIdx, "Only one example index per positive term"))

      val posArray = Array[(Long, F)]((firstIdx, if(tag.tpe.toString == "Double")
        this.positive.map(x => x._2.asInstanceOf[Double]).
        foldLeft(0.0)(_ + _).asInstanceOf[F]
      else
        this.positive.map(x => x._2.asInstanceOf[Float]).
        foldLeft(0.0f)(_ + _).asInstanceOf[F]
      ))

      val negArray = this.negative.toArray

      Grad(posArray, negArray)

    }

  }

  // for scoring we need the sigmoid function
  def sigmoidDouble(x: Double): Double =  1.0 / (1.0 + scala.math.exp(-x))

  /******************************************************
   *  Produce positive & negative labels
   *  Note we need to put ClassTags to repartition the RDD
   *  via an implicit conversion to PairRDDFunctions
   ******************************************************/
  def applyPositive[E: ClassTag, L: ClassTag, @specialized (Double, Float) F: ClassTag]
    (data: DistributedTargets[LinkedExLabel[E, L, F], FeedBackType, F],
      indexer: LinkedExLabel[E, L, F]=>Long,
      partitioner: QuotientPartitioner) : PositiveLabels[L, ImplicitLabel[E, F]] = {

    logDebug("Extracting positive labels")

    val reshapedData = for {
      exa <- data.self
      index = indexer(exa._2.exid)
      LinkedExLabel(exId, linkId, relWeight) = exa._2.exid
      feedback = exa._2.target
      weight = exa._2.weight

    } yield (index, (linkId, ImplicitLabel(exId, feedback, relWeight, weight)))

    val repartitioned = reshapedData.partitionBy(partitioner)

    new PositiveLabels(repartitioned, partitioner)
  }

  def applyNegative[E: ClassTag, L: ClassTag, @specialized (Float, Double) F: ClassTag]
    (data: DistributedTargets[LinkedExLabel[E, L, F], FeedBackType, F],
      indexer: LinkedExLabel[E, L, F]=>Long,
      partitioner: QuotientPartitioner) : NegativeLabels[L, ImplicitLabel[E, F]] = {

    logDebug("Extracting Negative labels")
    val reshapedData = for {
      exa <- data.self
      index = indexer(exa._2.exid)
      LinkedExLabel(exId, linkId, relWeight) = exa._2.exid
      feedback = exa._2.target
      weight = exa._2.weight

    } yield (index, (linkId, ImplicitLabel(exId, feedback, relWeight, weight)))

    val repartitioned = reshapedData.partitionBy(partitioner)

    new NegativeLabels(repartitioned, partitioner)
  }

  /** compute losses on a negative batch
   * by the time it is called postDistFeas, ... are assumed to be cached
   */
  type DSV[F] = DistributedStackedDenseVectors[F]

  import FactorizationMachine.FMScore
  def computeLosses[E: ClassTag, L: ClassTag,
    @specialized (Double, Float) F : ClassTag : TypeTag](
    posDistFeas: DistributedFeatures[F], posDistTgts:
        DistributedTargets[LinkedExLabel[E, L, F], FeedBackType, F],
      negDistFeas: DistributedFeatures[F],
      negDistTgts: DistributedTargets[LinkedExLabel[E, L, F], FeedBackType, F],
      posScores: Scores[FMScore[F]],
      negScores: Scores[FMScore[F]],
      indexer: LinkedExLabel[E, L, F]=>Long,
      linkPartitioner: Partitioner
  )(implicit genericField: GenericField[F]): RDD[(L, (Loss[F], Grad[F]))] =
  {

    // Extract the quotient partitioner
    val qp = posDistFeas.pp.leftPartitioner

    // create positive labels
    val posLabels = applyPositive(posDistTgts, indexer=indexer,
      partitioner=qp)
    posLabels.self.setName("Positive Labels")

    // now the negative scores
    val negLabels = applyNegative(negDistTgts, indexer=indexer,
      partitioner=qp)
    negLabels.self.setName("Negative Labels")    

    import FactorizationMachine.FMScore
    val liftedJoiner = (is: Iterator[(Long, FMScore[F])],
      il: Iterator[(Long, (L, ImplicitLabel[E, F]))]) =>

    {
      val ms = is.map(x => (x._1, x._2.score)).toMap
      val ll = il.toList

      val out = for {
        elem <- ll
        (longIdx, (linkId, impLabel)) = elem
        joined = JoinedFeedScore[E, F](label=impLabel, score=ms(longIdx),
          exId=longIdx)
      } yield (linkId, joined)

      out.toIterator
    }


    // Join the positive labels & scores
    val posJoined = posScores.self.zipPartitions(
      posLabels.self, preservesPartitioning=true)(liftedJoiner).
      partitionBy(linkPartitioner)
    posJoined.setName("posJoined")

    // Join the negative labels & scores
    val negJoined = negScores.self.zipPartitions(
      negLabels.self, preservesPartitioning=true)(liftedJoiner).
      partitionBy(linkPartitioner)
    negJoined.setName("negJoined")

    val liftedScoring = (ipos: Iterator[(L, JoinedFeedScore[E, F])],
      ineg: Iterator[(L, JoinedFeedScore[E, F])]) => {

      val mpos = ipos.toMap
      val lneg = ineg.toList

      val losses = MutMap[L, (LossBuffer[F], GradBuffer[F])]()

      mpos.map(x => (losses(x._1) = (new LossBuffer(genericField.zero, genericField.zero),
        GradBuffer(ArrayBuffer[(Long, F)]() , ArrayBuffer[(Long,F)]()))))

      // negatives get used for scoring
      lneg.map( neg => {

        val (link, negJoined) = neg
        val posJoined = mpos(link)
        val JoinedFeedScore(posLabel, posScore, posExId) = posJoined
        val JoinedFeedScore(negLabel, negScore, negExId) = negJoined

        // the sigmoid loss between difference of scores
        val diff = genericField.opToDouble(posScore) - genericField.opToDouble(negScore)
        val sigDiff =  sigmoidDouble(diff)

        // local weight
        val w = genericField.opMul(posLabel.weight, negLabel.relWeight)

        //  morally we update loss with -w * log(sigDiff) but log needs to be regularized
        val regLog =  scala.math.log(sigDiff + 1e-24)

        losses(link)._1.loss = genericField.opSub(losses(link)._1.loss,
          genericField.opMul(w, genericField.opFromDouble(regLog)))
        

        // we update weight
        losses(link)._1.weight = genericField.opAdd(losses(link)._1.weight, w)

        // local gradient
        val g = (- genericField.opToDouble(w)
          / (sigDiff + 1e-24) * sigDiff * (1-sigDiff))

        // update losses
        losses(link)._2.positive.append((posExId, genericField.opFromDouble(g)))
        losses(link)._2.negative.append((negExId, genericField.opFromDouble(-g)))
      })

      losses.mapValues( x=> (x._1.toLoss, x._2.toGrad)).toIterator
    }

    val distributedLosses = posJoined.
      zipPartitions(negJoined, preservesPartitioning=true)(liftedScoring)
    distributedLosses
  }


  // gradient computation on a Batch of negative examples
  def computeGradients[E: ClassTag, L: ClassTag,
    @specialized (Double, Float) F : ClassTag : TypeTag]
    (distributedLosses: RDD[(L, (Loss[F], Grad[F]))],
    posScores: Scores[FMScore[F]], negScores: Scores[FMScore[F]],
    posDistFeas: DistributedFeatures[F], negDistFeas: DistributedFeatures[F],
      rowDim: Int, distributedWeight: F)(implicit fieldImpl: GenericField[F]) = {

    // partitioner extraction
    val scoresQp = posScores.self.partitioner.get.asInstanceOf[QuotientPartitioner]
    val feasPp = posDistFeas.pp
    val feasBp = feasPp.rightPartitioner
    val elemBlock = feasBp.elementsPerBlock

    require(scoresQp == feasPp.leftPartitioner, "partitioners must agree")

    // reshape the distributed losses
    val posLoss = { for {
      loss <- distributedLosses
      posLoss = loss._2._2.positive
      (idx, grad) <- posLoss
    } yield (idx, grad)
    }.partitionBy(scoresQp)


    val negLoss = { for {
      loss <- distributedLosses
      negLoss = loss._2._2.negative
      (idx, grad) <- negLoss
    } yield (idx, grad)
    }.partitionBy(scoresQp)

    val tag = typeTag[F]

    // local product of DV[F] and F
    def localVecProd(v: DV[F], s: F): DV[F] = {
      val newVecData = v.data.map(x => fieldImpl.opMul(x, s))
      new DV[F](data=newVecData, offset=v.offset, stride=v.stride,
          length=v.length)
    }

    // local sum of DV[F] and DV[F]
    def localVecSum(v: DV[F], w: DV[F]): DV[F] = {
      if(tag.tpe.toString == "Float")
        (v.asInstanceOf[DV[Float]] + w.asInstanceOf[DV[Float]]).asInstanceOf[DV[F]]
      else if(tag.tpe.toString == "Double")
        (v.asInstanceOf[DV[Double]] + w.asInstanceOf[DV[Double]]).asInstanceOf[DV[F]]
      else throw new Error("Unimplemented")
    }

    // Join losses & scores
    def liftedJoinScores(iloss: Iterator[(Long, F)],
      iscore: Iterator[(Long, FMScore[F])]) = {
      // map loss in memory
      val mloss = iloss.toMap

      // targets are used only to lookup feature indices
      val lscore = iscore.toList

      val out = for {
        (idx, FMScore(_, toBackProp)) <- lscore
        _ = (require(!toBackProp.isEmpty, "gradient info for score needed"))
        // this is not actually the full gradient but will help to compute it
        scoreGrad = toBackProp.get
        lossGrad = mloss(idx)
        // change the scoreGrad to a Map of Lists
        scoreMap = scoreGrad.toList.map(x => (x._1._1, (x._1._2, x._2))).
        groupBy(_._1).mapValues(_.map(_._2))

        (feaKey, scoreGradList) <- scoreMap
        lossGradList = scoreGradList.map(x => (x._1, localVecProd(x._2, lossGrad)))

      } yield ((idx, feaKey), lossGradList)

      out.toIterator
    }

    val posPairing = posLoss.zipPartitions(posScores.self,
      preservesPartitioning=true)(liftedJoinScores).
      partitionBy(feasPp)


    def liftedGradient(ipair : Iterator[((Long, Int), List[(Int, DV[F])])],
      ifea: Iterator[((Long, Int), SV[F])]) = {

      val lpair = ipair.toList
      val mfea = ifea.toMap

      // accumulator as a map
      val accum = MutMap[(Int, Int), DV[F]]()

      val zerosF: DV[F] = if(tag.tpe.toString == "Float")
        DV.zeros[Float](rowDim).asInstanceOf[DV[F]]
      else if (tag.tpe.toString == "Double")
        DV.zeros[Double](rowDim).asInstanceOf[DV[F]]
      else throw new Error("unimplemented")

      for {
        (key, vecList) <- lpair
        sparse = mfea(key)
        sparseLookup = sparse.index.zip(sparse.data).toMap
        (localFeaKey, backPropagated) <- vecList
        tupleKey = (key._2, localFeaKey)
        _ = (if(!accum.contains(tupleKey)) accum(tupleKey)=zerosF)
        fea = sparseLookup.getOrElse(localFeaKey, fieldImpl.zero)
        toAdd = localVecProd(backPropagated, fea)
        _ = (accum(tupleKey) = localVecSum(accum(tupleKey), toAdd))

      } yield ()

      // now for each first key in accum we create an empty DM of the right dimension
      val out = MutMap[Int, DM[F]]()

      val zerosDMF: DM[F] = if(tag.tpe.toString == "Float")
        DM.zeros[Float](rowDim, elemBlock).asInstanceOf[DM[F]]
      else if (tag.tpe.toString == "Double")
        DM.zeros[Double](rowDim, elemBlock).asInstanceOf[DM[F]]
      else throw new Error("unimplemented")

      for {
        globalFeaKey <- accum.keys.map(_._1).toList.distinct
        _ = (out(globalFeaKey) = zerosDMF)
      } yield ()

      for {
        ((globalKey, localKey), grad) <- accum
        j <- 0 until grad.size
      } yield (out(globalKey).update(j, localKey, grad(j)))

      out.toIterator

    }

    // local sum of DM[F] and DM[F]
    def localMatSum(v: DM[F], w: DM[F]): DM[F] = {
      if(tag.tpe.toString == "Float")
        (v.asInstanceOf[DM[Float]] + w.asInstanceOf[DM[Float]]).asInstanceOf[DM[F]]
      else if(tag.tpe.toString == "Double")
        (v.asInstanceOf[DM[Double]] + w.asInstanceOf[DM[Double]]).asInstanceOf[DM[F]]
      else throw new Error("Unimplemented")
    }

    val posGrad =
      posPairing.zipPartitions(posDistFeas.self, preservesPartitioning=true)(liftedGradient).
        partitionBy(feasBp).reduceByKey(localMatSum(_ , _))

    val vecPosGrad  = new DistributedStackedDenseVectors[F](posGrad, rowDim)

    val negPairing = negLoss.zipPartitions(negScores.self,
      preservesPartitioning=true)(liftedJoinScores).
      partitionBy(feasPp)

    val negGrad =
      negPairing.zipPartitions(negDistFeas.self, preservesPartitioning=true)(liftedGradient).
        partitionBy(feasBp).reduceByKey(localMatSum(_ , _))

    val vecNegGrad = new DistributedStackedDenseVectors[F](negGrad, rowDim)

    // local division of DM[F] and F
    def localMatDiv(v: DM[F], s: F): DM[F] = {
      if(tag.tpe.toString == "Float")
        (v.asInstanceOf[DM[Float]] / s.asInstanceOf[Float]).asInstanceOf[DM[F]]
      else if(tag.tpe.toString == "Double")
        (v.asInstanceOf[DM[Double]] / s.asInstanceOf[Double]).asInstanceOf[DM[F]]
      else throw new Error("Unimplemented")

    }


    (vecPosGrad + vecNegGrad).mapBlocks(x => localMatDiv(x, distributedWeight))
  }

  // do a cycle of loss & gradient computations on negative batches
  def computeOnNegBatches[E: ClassTag, L : ClassTag,
    @specialized (Float, Double) F : ClassTag : TypeTag](
    posDistFeas: DistributedFeatures[F],
      posDistTgts: DistributedTargets[LinkedExLabel[E, L, F], FeedBackType, F],
      negsRDD: Array[RDD[Example[LinkedExLabel[E, L, F], FeedBackType, F]]],
      weight: DistributedStackedDenseVectors[F],
      posBatcher: ArrayBatcher[LinkedExLabel[E, L, F], FeedBackType, F],
      indexer: LinkedExLabel[E, L, F]=>Long,
      linkPartitioner: Partitioner,
      rowDim: Int,
      autograd: Boolean = true,
      negBatcherDepth: Int = 2
  )(implicit zeroValue: breeze.storage.Zero[F],
    fieldImpl: GenericField[F]): (F, Option[DSV[F]]) = {
    // Extract the quotient partitioner
    val qp = posDistFeas.pp.leftPartitioner

    // construct the positive labels
    val posLabels = applyPositive(posDistTgts, indexer=indexer,
      partitioner=qp)
    posLabels.self.setName("posLabels")

    // distribute the weights for scoring
    val repDSV = RepSDDV.apply(weight, posDistFeas.self.partitioner.get.
      asInstanceOf[ProductPartitioner].leftPartitioner)

    // the positives are only scored once at the beginning
    val posScores = FactorizationMachine.apply(posDistFeas, repDSV)
    posScores.self.setName("posScores")

    logDebug("Persisting positive scores")
    posScores.self.cache.count

    /** create the negative Batcher we will sample from
     * Note: as design choice we pass the posBatcher as argument to
     * read some parameters needed to construct the negative batcher &
     * moreover, the negative samples are to be supplied via negsRDD
     * so we keep the sampling separated from this function.
     * In this way other ways to sample / score against the negatives
     * are easier to implement. 
     */
    logDebug("Creating a negative batcher from the negative samples")
    val negBatcher = new ArrayBatcher[LinkedExLabel[E, L, F], FeedBackType, F](
      indexer=posBatcher.indexer, maxExampleIdx=posBatcher.maxExampleIdx,
      maxFeaIdx=posBatcher.maxFeaIdx, suggPartExamples=posBatcher.suggPartExamples,
      suggPartFeas=posBatcher.suggPartFeas,
      batches=negsRDD, name=s"NegativeBatcher_${posBatcher.getIdx}")

    // accumulator
    val lossesAndGradients = ArrayBuffer[(F, F, Option[DSV[F]])]()
    // accumulate losses & gradients
    while(!negBatcher.oneEpochCompleted) {
      val (_, negDistFeas, negDistTgts) = negBatcher.next(negBatcherDepth)

      val negLabels = SigmoidPairedRanking.applyNegative(negDistTgts, indexer=indexer,
        partitioner=qp)
      negLabels.self.setName(s"negLabels_${negBatcher.getIdx}")

      val negScores = FactorizationMachine.apply(negDistFeas, repDSV)
      negScores.self.setName(s"negScores_${negBatcher.getIdx}")

      logDebug(s"Caching ${negScores.self.name}")
      negScores.self.cache.count

      // compute Losses
      val distributedLosses = computeLosses(posDistFeas,
        posDistTgts, negDistFeas, negDistTgts, posScores, negScores,
        indexer=indexer, linkPartitioner=linkPartitioner)

      distributedLosses.setName(s"distributedLosses_${negBatcher.getIdx}")
      logDebug(s"Caching ${distributedLosses.name}")
      distributedLosses.cache.count

      // collect loss & weight on the batch
      val (batchLoss, batchWeight) = distributedLosses.map(_._2._1).treeAggregate((fieldImpl.zero,
        fieldImpl.zero))(
        seqOp = (x, y) => (fieldImpl.opAdd(x._1, y.loss), fieldImpl.opAdd(x._2 , y.weight)),
          combOp = (x,y) =>
        (fieldImpl.opAdd(x._1 , y._1), fieldImpl.opAdd(x._2 , y._2)))

      // compute the gradient via a reduction step
      val distributedGrad = if(autograd) {
        // for mystical reasons classTags & typeTags had to be resupplied :0
        implicit val classTagE = classTag[E]
        implicit val classTagL = classTag[L]
        implicit val classTagF = classTag[F]
        implicit val typeTagF = typeTag[F]
        Some(computeGradients(distributedLosses,
          posScores, negScores, posDistFeas,  negDistFeas, rowDim, batchWeight)(
          classTagE, classTagL, classTagF, typeTagF, fieldImpl
        )
        )
      }
      else None

      // if the gradient is not empty we cache it and checkpoint
      if(autograd) {
        logDebug("Autograd is on, checkpointing & caching the gradient")
        distributedGrad.get.self.checkpoint
        distributedGrad.get.self.persist.count
      }

      lossesAndGradients.append((batchLoss, batchWeight, distributedGrad))

      logDebug("Freeing resources in loop of negative batches")
      negScores.self.unpersist()
      distributedLosses.unpersist()

    }

    // free resources of negativeBatcher
    logDebug(s"Freeing resources of ${negBatcher.name}")
    negBatcher.freeResources

    val totalWeight = lossesAndGradients.map(_._2).foldLeft(fieldImpl.zero)((x,y) =>
      fieldImpl.opAdd(x, y))

    val totalLoss = lossesAndGradients.map(_._1).foldLeft(fieldImpl.zero)((x,y) =>
      fieldImpl.opAdd(x,y))


    // for the sum of gradients we must correct by rescaling with totalWeight
    val totalGrad = if(autograd) {
      // initialize zero accumulator
      logDebug(s"Autogrd is on, summing all the gradients")
      val typeTagF = typeTag[F]
      val zeros = {
        if(typeTagF.tpe.toString == "Float")
          lossesAndGradients(0)._3.get.asInstanceOf[DSV[Float]].mapBlocks(v => v * 0.0f)
        else if (typeTagF.tpe.toString == "Double")
          lossesAndGradients(0)._3.get.asInstanceOf[DSV[Double]].mapBlocks(v => v * 0.0)
        else new Error("unimplemented")
      }.asInstanceOf[DSV[F]]


      val gradSum = {

        if(typeTagF.tpe.toString == "Float")
          lossesAndGradients.map(x=> (x._3.get.asInstanceOf[DSV[Float]],
            x._2.asInstanceOf[Float])).
            foldLeft(zeros.asInstanceOf[DSV[Float]])((x,y) =>
              (x + y._1.mapBlocks(_ * y._2/totalWeight.asInstanceOf[Float])))
        else if(typeTagF.tpe.toString == "Double")
          lossesAndGradients.map(x=> (x._3.get.asInstanceOf[DSV[Double]],
            x._2.asInstanceOf[Double])).
            foldLeft(zeros.asInstanceOf[DSV[Double]])((x,y) =>
              (x + y._1.mapBlocks(_ * y._2/totalWeight.asInstanceOf[Double])))
        else new Error("unimplemented")

      }.asInstanceOf[DSV[F]]

      Some(gradSum)

    } else None

    
    if(autograd) {
      logDebug("Autograd is on; checkpointing & caching total gradient")
      totalGrad.get.self.checkpoint
      totalGrad.get.persist
      totalGrad.get.self.count
      logDebug("Autograd is on; freeing gradients on the batches")
      lossesAndGradients.map(x => x._3.get.unpersist())
    }

    (fieldImpl.opDiv(totalLoss, totalWeight), totalGrad)
  }

}

/** Models are just ways of combining the previous ingredients
 * so that one returns a loss function
 */

object Models extends Logging with Serializable {

  import distfom.{DistributedDenseVector => DDV,
    FomDistDenseVec => FDDV, DistributedDiffFunction => DDF,
    FomDistVec => FBase,
    FomDistStackedDenseVec => FDSV}

  def linearL2Loss[E, @specialized (Float, Double) F: ClassTag : TypeTag](
    @transient batcher: Batcher[E, F, F], depth: Int = 2)
    (implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
      : DDF[FDDV[F], F]  =  new DDF[FDDV[F], F] {

      def compute(x: distfom.FomDistDenseVec[F]): (F, distfom.FomDistDenseVec[F]) = {
        val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

        // distribute weights for scoring
        val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
        // score
        val scores = LinearScorer(distFeas, repDDV)
        // losses
        val losses = L2Loss(scores, distTgts)
        // forward pass
        logDebug("Computing total loss")
        val totalLoss = L2Loss.forward(losses, totalWeight=totalWeight,
          depth = depth)
        // backward pass
        val ddvGrad = L2Loss.backward(losses, distFeas, totalWeight=totalWeight,
          depth=depth)
        // need to create the distfom
        val grad = new FDDV[F](ddvGrad, dotDepth=x.dotDepth)

        totalLoss -> grad
      }

    override def computeGrad(x: distfom.FomDistDenseVec[F]): distfom.FomDistDenseVec[F] = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

        // distribute weights for scoring
        val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
        // score
        val scores = LinearScorer(distFeas, repDDV)
        // losses
        val losses = L2Loss(scores, distTgts)
        // backward pass
        val ddvGrad = L2Loss.backward(losses, distFeas, totalWeight=totalWeight,
          depth=depth)
        // need to create the distfom
        val grad = new FDDV[F](ddvGrad, dotDepth=x.dotDepth)

        grad
      }

    override def computeValue(x: distfom.FomDistDenseVec[F]): F = {
        val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

        // distribute weights for scoring
        val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
        // score
        val scores = LinearScorer(distFeas, repDDV)
        // losses
        val losses = L2Loss(scores, distTgts)
        // forward pass
        logDebug("Computing total loss")
        val totalLoss = L2Loss.forward(losses, totalWeight=totalWeight,
          depth = depth)
        totalLoss 
      }

      override def holdBatch(): Unit = batcher.holdBatch()
      override def stopHoldingBatch(): Unit = batcher.stopHoldingBatch()

  }

  def linearBinLogLoss[E, @specialized (Float, Double) F: ClassTag : TypeTag](
                                                                           @transient batcher: Batcher[E, Boolean, F], depth: Int = 2)
                                                                             (implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
  : DDF[FDDV[F], F]  =  new DDF[FDDV[F], F] {

    def compute(x: distfom.FomDistDenseVec[F]): (F, distfom.FomDistDenseVec[F]) = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDDV)
      // losses
      val losses = BinLogLoss(scores, distTgts)
      // forward pass
      logDebug("Computing total loss")
      val totalLoss = BinLogLoss.forward(losses, totalWeight=totalWeight,
        depth = depth)
      // backward pass
      val ddvGrad = BinLogLoss.backward(losses, distFeas, totalWeight=totalWeight,
        depth=depth)
      // need to create the distfom
      val grad = new FDDV[F](ddvGrad, dotDepth=x.dotDepth)

      totalLoss -> grad
    }

    override def computeGrad(x: distfom.FomDistDenseVec[F]): distfom.FomDistDenseVec[F] = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDDV)
      // losses
      val losses = BinLogLoss(scores, distTgts)
      // backward pass
      val ddvGrad = BinLogLoss.backward(losses, distFeas, totalWeight=totalWeight,
        depth=depth)
      // need to create the distfom
      val grad = new FDDV[F](ddvGrad, dotDepth=x.dotDepth)

      grad
    }

    override def computeValue(x: distfom.FomDistDenseVec[F]): F = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDDV)
      // losses
      val losses = BinLogLoss(scores, distTgts)
      // forward pass
      logDebug("Computing total loss")
      val totalLoss = BinLogLoss.forward(losses, totalWeight=totalWeight,
        depth = depth)
      totalLoss
    }

    override def holdBatch(): Unit = batcher.holdBatch()
    override def stopHoldingBatch(): Unit = batcher.stopHoldingBatch()

  }

  def linearQuantileLoss[E, @specialized (Float, Double) F: ClassTag : TypeTag](
                                                                                 @transient batcher: Batcher[E, F, F], quantile: F, depth: Int = 2)
                                                                               (implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
  : DDF[FDDV[F], F]  =  new DDF[FDDV[F], F] {

    def compute(x: distfom.FomDistDenseVec[F]): (F, distfom.FomDistDenseVec[F]) = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDDV)
      // losses
      val losses = QuantileLoss(scores, distTgts, quantile)
      // forward pass
      logDebug("Computing total loss")
      val totalLoss = QuantileLoss.forward(losses, totalWeight=totalWeight,
        depth = depth)
      // backward pass
      val ddvGrad = QuantileLoss.backward(losses, distFeas, totalWeight=totalWeight,
        depth=depth)
      // need to create the distfom
      val grad = new FDDV[F](ddvGrad, dotDepth=x.dotDepth)

      totalLoss -> grad
    }

    override def computeGrad(x: distfom.FomDistDenseVec[F]): distfom.FomDistDenseVec[F] = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDDV)
      // losses
      val losses = QuantileLoss(scores, distTgts, quantile)
      // backward pass
      val ddvGrad = QuantileLoss.backward(losses, distFeas, totalWeight=totalWeight,
        depth=depth)
      // need to create the distfom
      val grad = new FDDV[F](ddvGrad, dotDepth=x.dotDepth)

      grad
    }

    override def computeValue(x: distfom.FomDistDenseVec[F]): F = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDDV = RepDDV(x.ddvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDDV)
      // losses
      val losses = QuantileLoss(scores, distTgts, quantile)
      // forward pass
      logDebug("Computing total loss")
      val totalLoss = QuantileLoss.forward(losses, totalWeight=totalWeight,
        depth = depth)
      totalLoss
    }

    override def holdBatch(): Unit = batcher.holdBatch()
    override def stopHoldingBatch(): Unit = batcher.stopHoldingBatch()

  }


  import MultiLabelSoftMaxLogLoss.MultiLabelTarget

  def linearLogLoss[E, @specialized (Float, Double) F: ClassTag : TypeTag](
    @transient batcher: Batcher[E, MultiLabelTarget[F], F], rowDim: Int,
    depth: Int = 2)
    (implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
      : DDF[FDSV[F], F]  =  new DDF[FDSV[F], F] {

    def compute(x: distfom.FomDistStackedDenseVec[F]):
        (F, distfom.FomDistStackedDenseVec[F]) = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDSV = RepSDDV(x.dsvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDSV)
      // losses
      val losses = MultiLabelSoftMaxLogLoss(scores, distTgts)
      // forward pass
      logDebug("Computing total loss")

      val totalLoss = MultiLabelSoftMaxLogLoss
        .forward(losses, totalWeight=totalWeight, depth = depth)

      // backward pass
      val dsdvGrad = MultiLabelSoftMaxLogLoss
        .backward(losses, distFeas, totalWeight=totalWeight, rowDim=rowDim,
          depth=depth)
      // need to create the distfom
      val grad = new FDSV[F](dsdvGrad, dotDepth=x.dotDepth)

      totalLoss -> grad
    }

    override def computeGrad(x: distfom.FomDistStackedDenseVec[F]):
        distfom.FomDistStackedDenseVec[F] = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDSV = RepSDDV(x.dsvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDSV)
      // losses
      val losses = MultiLabelSoftMaxLogLoss(scores, distTgts)
      // backward pass
      val dsvGrad = MultiLabelSoftMaxLogLoss
        .backward(losses, distFeas, totalWeight=totalWeight,
          rowDim=rowDim, depth=depth)
      // need to create the distfom
      val grad = new FDSV[F](dsvGrad, dotDepth=x.dotDepth)

      grad
    }

    override def computeValue(x: distfom.FomDistStackedDenseVec[F]): F = {
      val (totalWeight, distFeas, distTgts) = batcher.next(depth=depth)

      // distribute weights for scoring
      val repDSV = RepSDDV(x.dsvec, distFeas.pp.leftPartitioner)
      // score
      val scores = LinearScorer(distFeas, repDSV)
      // losses
      val losses = MultiLabelSoftMaxLogLoss(scores, distTgts)
      // forward pass
      logDebug("Computing total loss")

      val totalLoss = MultiLabelSoftMaxLogLoss
        .forward(losses, totalWeight=totalWeight,
          depth = depth)
      totalLoss
    }

    override def holdBatch(): Unit = batcher.holdBatch()
    override def stopHoldingBatch(): Unit = batcher.stopHoldingBatch()

  }

  import SigmoidPairedRanking.{FeedBackType, computeOnNegBatches}
  def SigmoidPairedRankingLoss[E: ClassTag, L: ClassTag, @specialized (Float, Double) F: ClassTag : TypeTag](
    @transient posBatcher: ArrayBatcher[LinkedExLabel[E, L, F], FeedBackType, F],
    @transient negSampler: NegativeSampler[E, L, FeedBackType, F],
    indexer: LinkedExLabel[E, L, F]=>Long,
    linkPartitioner: Partitioner,
    batcherDepth: Int = 2)
    (implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F],
    zeroValue: breeze.storage.Zero[F])
      : DDF[FDSV[F], F]  =  new DDF[FDSV[F], F] {

    def compute(x: distfom.FomDistStackedDenseVec[F]):
        (F, distfom.FomDistStackedDenseVec[F]) = {

      val (posTotalWeight, posDistFeas, posDistTgts) = posBatcher.next(depth=batcherDepth)

      // idx of the current positive batch
      val posIdx = posBatcher.getIdx
      logDebug(s"Instantiate negative RDD from the positive batch # $posIdx")
      val negsRDD = negSampler.sample(posBatcher.batches(posIdx))

      val (distLoss, totalGrad) = computeOnNegBatches(
        posDistFeas, posDistTgts, negsRDD,
        x.dsvec, posBatcher, indexer=indexer, linkPartitioner=linkPartitioner,
        rowDim=x.dsvec.rowDim, autograd=true, negBatcherDepth=batcherDepth)

      distLoss -> new FDSV[F](totalGrad.get, dotDepth=x.dotDepth)
      
    }

    override def computeValue(x: distfom.FomDistStackedDenseVec[F]): F = {
      
      val (posTotalWeight, posDistFeas, posDistTgts) = posBatcher.next(depth=batcherDepth)

      // idx of the current positive batch
      val posIdx = posBatcher.getIdx
      logDebug(s"Instantiate negative RDD from the positive batch # $posIdx")
      val negsRDD = negSampler.sample(posBatcher.batches(posIdx))

      val (distLoss, _) = computeOnNegBatches(
        posDistFeas, posDistTgts, negsRDD,
        x.dsvec, posBatcher, indexer=indexer, linkPartitioner=linkPartitioner,
        rowDim=x.dsvec.rowDim, autograd=false, negBatcherDepth=batcherDepth)

      distLoss
    }
    
    override def holdBatch(): Unit = posBatcher.holdBatch()
    override def stopHoldingBatch(): Unit = posBatcher.stopHoldingBatch()
  }

}

