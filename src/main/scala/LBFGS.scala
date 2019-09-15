/**
 * Implementation of LBFGS
 * Inspired By: https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/optimize/LBFGS.scala
 *
 * @commit c2193a11cc117940ec2de71b724c65de8a09a62c
 *
 *
 * @author Andrea Schioppa
 */

package distfom

import scala.collection.mutable.ListBuffer

import scala.language.implicitConversions

import scala.reflect.ClassTag

import CommonImplicitGenericFields._

/**
 * LBFGS
 * @param m memory parameter
 * @param maxIter
 * @param tolerance
 * @param lowerFunTolerance
 * @param suppliedLineSearch optionally supplied Line Search (Otherwise a Strong Wolfe gets used)
 * @param maxLineSearchIter
 * @param maxZoomIter
 * @param baseLSearch step to pass to the first line search
 * @param interruptLinSteps
 * @param holdBatch
 * @param fieldImpl
 * @tparam B generic used by vector type
 * @tparam F numeric field
 */
class LBFGS[B, @specialized(Float, Double) F: ClassTag](m: Int, override val maxIter: Int = -1,
  override val tolerance: Double = 1E-6,
  override val lowerFunTolerance: Double = 0.0,
  val suppliedLineSearch: Option[MinimizingLineSearch[F]] = None,
  maxLineSearchIter: Int = 10,
  maxZoomIter: Int = 10,
  val baseLSearch: Double = 1.0,
  override val interruptLinSteps: Int = 4,
  override val holdBatch : Boolean = true)(implicit fieldImpl: GenericField[F])
    extends FirstOrderMinimizer[B, F, FomDistVec[B, F]](maxIter, tolerance, lowerFunTolerance,
      interruptLinSteps, holdBatch)
    with Serializable {

  import scala.math.abs

  type T = FomDistVec[B, F]

  type History = LBFGS.ApproximateInverseHessian[B, F]

  protected def takeStep(state: State, dir: T, stepSize: F): T = {

    val newX = dir * stepSize + state.x
    newX

  }

  /** initial history is empty (will default on identity for the first
   *  application of Hessian matrix)
   *  Note DF here does not take type parameters as it inherits from
   * FirstOrderMinimizer
   */
  protected def initialHistory(f: DF, x: T): History = {

    val hist = new LBFGS.ApproximateInverseHessian[B, F](m)
    hist
    
  }

  protected def chooseDescentDirection(state: State, fn: DF): T = {
    val dir = state.history * state.grad
    dir
  }

  protected def updateHistory(newX: T, newGrad: T, newVal: F,
    f: DF, oldState: State): History = {

    val updateX = newX - oldState.x
    updateX.setName(s"X_${oldState.iter}")

    val updateG = newGrad - oldState.grad
    updateG.setName(s"G_${oldState.iter}")
    
    oldState.history.updated(updateX, updateG)

  }

  protected def determineStepSize(state: State, f: DF, dir: T) = {
    val x = state.x
    val grad = state.grad

    val ff = LineSearch.functionFromSearchDirection[B, F, T](f, x, dir)

    // either take the supplied line search or by default the StrongWolfeLineSearch
    val search = suppliedLineSearch.getOrElse(
      new StrongWolfeLineSearch[F](maxZoomIter = maxZoomIter,
        maxLineSearchIter = maxLineSearchIter)).
      asInstanceOf[MinimizingLineSearch[F]]
      
    val alpha = search.minimize(ff, if (state.iter == 0.0) baseLSearch else 1.0)

    if (fieldImpl.opToDouble(
      fieldImpl.opMul( alpha , grad.norm())) < 1E-10)
      throw StepSizeUnderflow

    alpha

  }

}

object LBFGS {

  type FV[B, F] = FomDistVec[B, F]

  /** here we need to explicitly supply type parameters for
   * DF
   */
  type DF[B, F] = DistributedDiffFunction[FV[B, F], F]

  /** FomDistVec[B] will in general contain RDDs so
   *  we mark the field as transient to avoid getting stuck
   * in serialization
   */

  case class ApproximateInverseHessian[B, @specialized (Float, Double) F : ClassTag](
    m: Int,
    @transient val memStep: IndexedSeq[FV[B, F]] = IndexedSeq.empty,
    @transient val memGradDelta: IndexedSeq[FV[B, F]] = IndexedSeq.empty
  )(implicit fieldImpl: GenericField[F]) extends Logging {

    type T = FV[B, F]

    def updated(step: T, gradDelta: T) = {

      // check if we need to free resources
      if(this.memStep.size == m) {
        logDebug(s"History length $m; we will unpersist ${this.memStep(m-1).name}")
        this.memStep(m-1).unpersist()
      }

      // we keep the steps copied as
      // FirstOrderMinimizer will free resources during the cycles

      val copy_step = step.copy()
      copy_step.setName(s"HCopy_${step.name}")

      logDebug(s"Start persisting ${copy_step.name}")
      copy_step.interruptLineage()
      copy_step.persist()
      copy_step.count()
      logDebug(s"End persisting ${copy_step.name}")

      // append to mem steps
      val memStep = (copy_step +: this.memStep).take(m)

      // free resources if possible
      if(this.memGradDelta.size == m) {
        logDebug(s"History length $m; we will unpersist ${this.memGradDelta(m-1).name}")
        this.memGradDelta(m-1).unpersist()
      }

      // we keep the gradients copied as
      // FirstOrderMinimizer will free resources during the cycles
      val copy_gradDelta = gradDelta.copy()
      copy_gradDelta.setName(s"HCopy_${gradDelta.name}")
      copy_gradDelta.interruptLineage()
      logDebug(s"Start persisting ${gradDelta.name}")
      copy_gradDelta.persist()
      copy_gradDelta.count()
      logDebug(s"End persisting ${gradDelta.name}")

      // append to history
      val memGradDelta = (copy_gradDelta +: this.memGradDelta).take(m)

      new ApproximateInverseHessian(m, memStep, memGradDelta)
    }

    def historyLength = memStep.length
    // This is applying the inverse Hessian to a vector
    def *(grad: T) : T = {
      val diag = if (historyLength > 0) {

        val prevStep = memStep.head
        val prevGradStep = memGradDelta.head

        val sy = prevStep.dot(prevGradStep)
        val yy = prevGradStep.dot(prevGradStep)

        if (fieldImpl.opToDouble(sy) < 0 || fieldImpl.opToDouble(sy).isNaN) throw NaNHistory

        fieldImpl.opDiv(sy , yy)

      } else {
        fieldImpl.opFromDouble(1.0)
      }

      val dir = ListBuffer[T](grad)

      val as = new Array[F](m)
      val rho = new Array[F](m)

      // The first loop in computing the
      // descent direction
      for (i <- 0 until historyLength) {
        rho(i) = memStep(i).dot(memGradDelta(i))
        as(i) = fieldImpl.opDiv(memStep(i).dot(dir.last) , rho(i))
        if (fieldImpl.opToDouble(as(i)).isNaN) {
          throw NaNHistory
        }

        dir.append(dir.last - (memGradDelta(i)*as(i)))

        // persist the last to speed up computation
        dir.last.persist
        logDebug(s"Applying Approximate Hessian: start persisting dir${i} loop1")
        dir.last.count()
        logDebug(s"Applying Approximate Hessian: end persisting dir${i} loop1")
        // free resources to the previous to the last
        logDebug(s"Applying Approximate Hessian: unpersist dir${dir.size-2} loop1")
        dir(dir.size-2).unpersist()


      }

      dir.append(dir.last * diag)
      // persist the last to speed up computation
      dir.last.persist
      logDebug("Applying Approximate Hessian: start persisting dir * diag")
      dir.last.count()
      logDebug("Applying Approximate Hessian: end persisting dir * diag")
      // free resources to the previous to the last
      logDebug(s"Applying Approximate Hessian: unpersisting dir${dir.size-2}")
      dir(dir.size-2).unpersist()

      // the second loop in computing the descent direction
      for (i <- (historyLength - 1) to 0 by (-1)) {

        val beta = fieldImpl.opDiv(memGradDelta(i).dot(dir.last), rho(i))
        dir.append(dir.last + memStep(i)*(fieldImpl.opSub(as(i), beta)))

        dir.last.persist
        logDebug(s"Applying Approximate Hessian: start persisting dir${i} loop2")
        dir.last.count()
        logDebug(s"Applying Approximate Hessian: end persisting dir${i} loop2")

        // free resources to the previous to the last
        logDebug(s"Applying Approximate Hessian: unpersist dir${dir.size-2} loop2")
        dir(dir.size-2).unpersist()
      }

      // invert sign
      dir.append(dir.last * fieldImpl.opFromDouble(-1.0))
      // persist the last to speed up computation
      dir.last.persist
      logDebug("Applying Approximate Hessian: start persisting -dir")
      dir.last.count()
      logDebug("Applying Approximate Hessian: end persisting -dir")

      // free resources to the previous to the last
      logDebug(s"Applying Approximate Hessian: free dir${dir.size-2} final")
      dir(dir.size-2).unpersist()

      dir.last

    }

  }
}
