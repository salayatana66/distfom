/**
 * Implementations of vanilla SGD &
 * AdaGrad
 * @author Andrea Schioppa
 */
package distfom

import scala.language.implicitConversions

import scala.reflect.ClassTag

import CommonImplicitGenericFields._
import CommonImplicitGenericSmoothFields._

/**
 * Vanilla SGD
 * @param defaultStepSize
 * @param powDec
 * @param decayFactor
 * @param maxIter
 * @param tolerance
 * @param lowerFunTolerance
 * @param interruptLinSteps
 * @param holdBatch
 * @param fieldImpl
 * @tparam B generic used by vector type
 * @tparam F numeric field
 */
class StochasticGradientDescent[B, @specialized(Float, Double) F: ClassTag](val defaultStepSize: Double,
  val powDec: Double,
  val decayFactor: Double,
  override val maxIter: Int = -1,
  override val tolerance: Double = 1E-6,
  override val lowerFunTolerance: Double = 0.0,
  override val interruptLinSteps: Int = 4,
  override val holdBatch : Boolean = true)(implicit fieldImpl: GenericField[F])
    extends FirstOrderMinimizer[B, F, FomDistVec[B, F]](maxIter, tolerance, lowerFunTolerance,
      interruptLinSteps, holdBatch)
    with Serializable {

  import scala.math.abs

  type T = FomDistVec[B, F]

  // For Stochastic Gradient Descent we don't use the History
  type History = Unit

  protected def takeStep(state: State, dir: T, stepSize: F) = state.x + dir * stepSize
  protected def chooseDescentDirection(state: State, fn: DF) = state.grad * fieldImpl.opFromDouble(-1.0)
  protected def initialHistory(f: DF, x: T): History = ()
  protected def updateHistory(newX: T, newGrad: T, newVal: F,
    f: DF, oldState: State) = ()

  /**
   * Choose a step size scale for this iteration.
   *
   * Default is stepSize / math.pow(decayFactor(state.iter + 1),powDec)
   */
  def determineStepSize(state: State, f: DF, dir: T) = {
    fieldImpl.opFromDouble(
      defaultStepSize / math.pow(decayFactor*(state.iter)+ 1, powDec))
  }

}

/**
 * AdaGrad algorithms
 *
 * @see Adaptive Subgradient Methods for
 *      Online Learning and Stochastic Optimization
 *      Duchi et. ali.
 *
 * @param defaultStepSize
 * @param l2reg
 * @param l1reg
 * @param maxAge
 * @param delta
 * @param maxIter
 * @param tolerance
 * @param lowerFunTolerance
 * @param interruptLinSteps
 * @param holdBatch
 * @param fieldImpl
 * @param smoothImpl
 * @tparam B generic used by vector type
 * @tparam F numeric field
 */

class AdaptativeGradientDescent[B, @specialized(Float, Double) F: ClassTag](
  val defaultStepSize: Double,  val l2reg : Double = 0.0,    val l1reg: Double = 0.0,
  val maxAge: Int = 5,
  val delta: Double = 1e-4,
  override val maxIter: Int = -1,  override val tolerance: Double = 1E-6,
  override val lowerFunTolerance: Double = 0.0,
  override val interruptLinSteps: Int = 4,
override val holdBatch : Boolean = true)(implicit fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
    extends FirstOrderMinimizer[B, F, FomDistVec[B, F]](maxIter, tolerance, lowerFunTolerance, interruptLinSteps, holdBatch)
    with Serializable {

type T = FomDistVec[B, F]
  override type DF = DistributedDiffFunction[T, F]

  case class History(@transient sumOfSquaredGradients: T)

  override def initialHistory(f: DF, init: T) = History(init * fieldImpl.zero)

  override def updateHistory(newX: T, newGrad: T, newValue: F,
    f: DF, oldState: State) = {
      val oldHistory = oldState.history
      val newSquare = (oldState.grad * oldState.grad)

      val newG = if (oldState.iter > this.maxAge) {
        newSquare * fieldImpl.opFromDouble(1.0 / this.maxAge) +
        oldHistory.sumOfSquaredGradients * fieldImpl.opFromDouble(1.0-1.0/this.maxAge)
      } else {
        newSquare + oldHistory.sumOfSquaredGradients
      }

    newG.setName(s"newG_${oldState.iter}")
    new History(newG)
  }

  
  protected def takeStep(state: State, dir: T, stepSize: F): T = {

    import scala.math

    /* adjustement factor
    * s = sqrt(hist+grad^2+delta)
    */
    val s = (state.history.sumOfSquaredGradients + (state.grad*state.grad).
      applyElementWise(x => fieldImpl.opAdd(x , fieldImpl.opFromDouble(this.delta))).
      applyElementWise(smoothImpl.opAbsPow(_,.5)))

    /* step whether l2 regularization is present or not
     * x => (x*s+dir*learn)/(s+reg2*learn)
     */
    val l2reg_F = fieldImpl.opFromDouble(this.l2reg)
    val reg2X = if(this.l2reg > 0) (state.x*s + dir*stepSize)/(
      s.applyElementWise(fieldImpl.opAdd(_ , fieldImpl.opMul(l2reg_F,stepSize))))
    else state.x + (dir*stepSize)/s

    // adjustements if l1 reg is present
    val tlambda = fieldImpl.opMul(fieldImpl.opFromDouble(this.l1reg), stepSize)

    val l1Trunc = (xentry: F, sentry: F) => {
      if(fieldImpl.lessThan(smoothImpl.opAbsPow(xentry,1.0),
        fieldImpl.opDiv(tlambda, sentry))) {
        fieldImpl.zero
      } else {
        fieldImpl.opSub(xentry , fieldImpl.opMul(smoothImpl.opSignum(xentry) ,
          fieldImpl.opDiv(tlambda , sentry)))
      }
    }
    val reg1X = if(l1reg > 0) reg2X.pairWiseFun(s)(l1Trunc) else reg2X

    reg1X
  }

  
  override def determineStepSize(state: State, f: DF, dir: T) = {
      fieldImpl.opFromDouble(defaultStepSize)
    }
 
  
  protected def chooseDescentDirection(state: State, fn: DF) = state.grad * fieldImpl.opFromDouble(-1.0)

  override protected def adjust(newX: T, newGrad: T, newVal: F) = {
    import scala.math

    // more convenient to adjust using doubles
    val newVal_double = fieldImpl.opToDouble(newVal)

    val avl2 = if(this.l2reg > 0)  newVal_double +
    math.pow(fieldImpl.opToDouble(newX.norm(2.0)),2.0) * this.l2reg / 2.0
    else newVal_double

    val avl1 = if(this.l1reg > 0) avl2 + fieldImpl.opToDouble(newX.norm(1.0))*this.l1reg else avl2

    val agl2 = if(this.l2reg > 0) newGrad + newX * fieldImpl.opFromDouble(this.l2reg) else newGrad

    val l1reg_F = fieldImpl.opFromDouble(this.l1reg)
    val agl1 = if(this.l1reg > 0) newGrad + newX.applyElementWise(x =>
      fieldImpl.opMul(smoothImpl.opSignum(x) , l1reg_F)) else agl2

    (fieldImpl.opFromDouble(avl1) -> agl1)
    }
}

