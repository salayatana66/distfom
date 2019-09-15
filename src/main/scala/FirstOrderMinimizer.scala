// @todo ideally make code more customizable / slim using
// the Context design pattern
/* Inspired by
 https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/optimize/FirstOrderMinimizer.scala
 @commit c2193a11cc117940ec2de71b724c65de8a09a62c
 */

package distfom

import scala.language.implicitConversions
import scala.specialized

// TODO: remove distlbfgs. when moving up
import RichIterator._

/******************************************************
 Persistable vector allowing to take dot product
 and setting name for logging in the SparkUI
 ******************************************************/

trait FomDistVec[B, @specialized(Float, Double) F] extends Persistable with InterruptableLineage {

  def dot(other: FomDistVec[B, F]) : F

  // p-norm size
  def norm(p:  Double = 2.0) : F

  // for the moment we add count to force materialization
  def count() : Long

  // we also add setName for debugging
  def setName(name: String): Unit

  def *(d: F) : FomDistVec[B, F]


  /* Cast down to a B
   *  usefulness comes up in trying
   *  to evaluate in LineSearch
   *  maybe FomDistVec should be replaced by
   *  implicit class or its method put in distributed
   *  vectors
   */
  //def castDown: B

  // get a copy for independent persisting
  // might be achievable with a dummy map
  def copy(): FomDistVec[B, F]

  // to get the name
  def name : String

  // Element-wise sum of this and b. 
  def +(b: FomDistVec[B, F]) : FomDistVec[B, F]

  // Element-wise product of this and b. 
  def *(b: FomDistVec[B, F]) : FomDistVec[B, F]

  // Element-wise difference of this and b. 
  def -(b: FomDistVec[B, F]) : FomDistVec[B, F]

  // Element-wise ratio of this and b
  def /(b: FomDistVec[B, F]) : FomDistVec[B, F]

  /* pairwise map elementwise
   * mainly used for L1-regularization
   */
  def pairWiseFun(b: FomDistVec[B, F])(f: (F, F) => F) : FomDistVec[B, F]

  def applyElementWise(f: F => F) : FomDistVec[B, F]

  /*
   *  pairwise map with a map accepting a template key
   *  type (to be type checked)
   */

  import scala.reflect.runtime.universe._
  def keyedPairWiseFun[K](b: FomDistVec[B, F])
    (f : K)(implicit tagK: TypeTag[K], tagF: TypeTag[F]) : FomDistVec[B, F]
}


/******************************************************
 Trait representing a distributed differentiable 
 function; the function can be made to hold the batch,
 e.g. to compute a gradient
 ******************************************************/
trait DistributedDiffFunction[T, @specialized(Float, Double) F] {

  def computeValue(x: T): F = this.compute(x)._1

  def computeGrad(x: T): T = this.compute(x)._2

  // should return value & gradient
  def compute(x: T) : (F, T)

  // hold current batch
  def holdBatch() : Unit = ()

  // stop holding batch
  def stopHoldingBatch() : Unit = ()
}

import CommonImplicitGenericFields._

object DistributedDiffFunction {

  type DDF[T, F] = DistributedDiffFunction[T, F]

  type FV[B, F] = FomDistVec[B, F]

  /* distributed diff function as a linear combination of two
   */

  def linCombF[B, @specialized(Float, Double) F](a1: F, f1: DDF[FV[B, F], F],
    a2: F, f2: DDF[FV[B, F], F])(implicit fieldImp: GenericField[F])
      : DDF[FV[B, F], F] =
    new DistributedDiffFunction[FomDistVec[B, F], F] {

      /* the scalar field F can only be Float or Double
       * we use conversions to get around the fact that the scala compiler
       * cannot get by default the linear combination a1*b1+a2*b2 when all
       * ai, bi are of type F
       */
      override def computeValue(x: FV[B, F]) = fieldImp.opLinComb(f1.computeValue(x),
          a1, f2.computeValue(x), a2)

      override def computeGrad(x: FV[B, F]) = f1.computeGrad(x)*a1 + f2.computeGrad(x)*a2

      // should return value & gradient
      def compute(x: FV[B, F])  = {
        val Tuple2(v1,g1) = f1.compute(x)
        val Tuple2(v2,g2) = f2.compute(x)

        val outg = g1*a1 + g2*a2
        val outv = fieldImp.opLinComb(v1, a1, v2, a2)

        (outv, outg)
      }


      override def holdBatch() = {
        f1.holdBatch
        f2.holdBatch
      }

      override def stopHoldingBatch = {
        f1.stopHoldingBatch
        f2.stopHoldingBatch
      }
    }

  def regularizationL2[B, @specialized(Float, Double) F](implicit fieldImpl: GenericField[F])
  : DDF[FV[B,F], F] =
    new DistributedDiffFunction[FomDistVec[B, F], F] {

      override def computeValue(x: FV[B, F]) = fieldImpl.opMul(x.dot(x), fieldImpl.opFromDouble(0.5))

      override def computeGrad(x: FV[B, F]) = x

      // should return value & gradient
      def compute(x: FV[B, F])  = {
        val funValue = fieldImpl.opMul(x.dot(x),
          fieldImpl.opFromDouble(0.5))
        val funGrad = x

        funValue -> funGrad
      }


      override def holdBatch() = {

      }

      override def stopHoldingBatch = {

      }
    }

  def regularizationL1[B, @specialized(Float, Double) F](implicit fieldImpl: GenericField[F],
                                                         smoothImpl: GenericSmoothField[F])
  : DDF[FV[B,F], F] =
    new DistributedDiffFunction[FomDistVec[B, F], F] {

      override def computeValue(x: FV[B, F]) = x.dot(x.applyElementWise(v => smoothImpl.opSignum(v)))

      override def computeGrad(x: FV[B, F]) = x.applyElementWise(v => smoothImpl.opSignum(v))

      // should return value & gradient
      def compute(x: FV[B, F])  = {
        val funValue = x.dot(x.applyElementWise(v => smoothImpl.opSignum(v)))
        val funGrad = x.applyElementWise(v => smoothImpl.opSignum(v))

        funValue -> funGrad
      }


      override def holdBatch() = {

      }

      override def stopHoldingBatch = {

      }
    }
}

/******************************************************
 Exceptions are used, in some cases,
 to handle termination
 ******************************************************/

abstract class FirstOrderException(val msg: String = "")
    extends RuntimeException(msg)
 //
case object NaNHistory extends FirstOrderException(msg = "NaNHistory")

case object StepSizeUnderflow extends FirstOrderException(
  msg = "StepSizeUnderflow")

case object StepSizeOverflow extends FirstOrderException(
  msg = "StepSizeOverflow")

case class LineSearchFailed(motivation: String)

extends FirstOrderException(msg = s"LineSearchFailed: ${motivation}")

/******************************************************
 Prototype of first order methods
 ******************************************************/

abstract class FirstOrderMinimizer[B, F,
                                   T <: FomDistVec[B, F]]
  (val maxIter: Int = -1, val tolerance: Double = 1E-6,
    val lowerFunTolerance: Double = 0.0,
    val interruptLinSteps : Int = 4,
    val holdBatch : Boolean = true)(implicit fieldImpl: GenericField[F])
    extends  Logging with Serializable {

  type DF = DistributedDiffFunction[T, F]
  type History
  type State = FirstOrderMinimizer.State[T, F, History]
  type ConvergenceInfo = Option[FirstOrderMinimizer.ConvergenceReason]

  protected def initialHistory(f: DF, init: T): History
  protected def adjustFunction(f: DF): DF = f

  // this is useful for methods which adaptatively change the gradient
  protected def adjust(newX: T, newGrad: T, newVal: F): (F, T) = (newVal, newGrad)

  // this is the place where the state is used to adjust the gradient direction
  protected def chooseDescentDirection(state: State, f: DF): T

  // this is where e.g. a line search gets triggered
  protected def determineStepSize(state: State, f: DF, direction: T): F
  protected def takeStep(state: State, dir: T, stepSize: F): T
  protected def updateHistory(newX: T, newGrad: T, newVal: F, f: DF, oldState: State): History

  // convergence check
  // TODO: need to add rules about Gradient Failure
  protected def convergenceCheck(x : T, grad : T, value: F,
    state: State, convergenceInfo: ConvergenceInfo): ConvergenceInfo = {


    convergenceInfo match {
      case Some(convergence) => Some(convergence)

      case None => {
        // previous function value
        val prev_value = fieldImpl.opToDouble(
          state.value)
        // current function value
        val curr_value = fieldImpl.opToDouble(
          value)

        val gradNorm = fieldImpl.opToDouble(grad.norm())
        logDebug(s"""
             checking convergence
             f(x_prev) = $prev_value
             f(x_curr) = $curr_value
             |f'(x_curr)| = $gradNorm
             """)
        /* For the function value to converge we require
         * prev_value - value < tolerance if prev_value >= value;
         *  if value > prev_value we check that
         * value < prev_value + funLowerTolerance
         */
        val value_improved = curr_value <= prev_value
        if( ( (value_improved) & (prev_value - curr_value < tolerance))
          | ( (!value_improved) & (curr_value - prev_value > lowerFunTolerance)))
          Some(FirstOrderMinimizer.FunctionValuesConverged)

        else if (state.iter +1 >= maxIter)
          Some(FirstOrderMinimizer.MaxIterations)

        else if (gradNorm < tolerance) {
          Some(FirstOrderMinimizer.GradientConverged)
        }

        else  None
      }
    }

  }

  /******************************************************
   First state when starting minimization
   note that init controls the first point
   ******************************************************/

  protected def initialState(f: DF, init: T,
    initial_state: Option[State] = None): State = initial_state match {
    case Some(state) => state
    case _ => {
    val x = init
    val history = initialHistory(f, init)
    val (value, grad) = calculateObjective(f, x, history)
    val (adjValue, adjGrad) = adjust(x, grad, value)
    new State(x, value, grad, adjValue, adjGrad, 0, adjValue, history)
    }
  }

  /******************************************************
   Compute loss & gradient
   ******************************************************/

  protected def calculateObjective(f: DF, x: T, history: History): (F, T) = {
    f.compute(x)
  }

  // Infinite loop of iterations
  def infiniteIterations(f: DF, state: State): Iterator[State] = {

    // initialization
    var failedOnce = false
    val adjustedFun = adjustFunction(f)
    var numSteps = 0

    Iterator.iterate(state) { state =>

      // we use try - catch to handle convergence
      try {

        // choose and persist a descent direction

        logDebug(s"Iterative Step: ${numSteps}")
        val dir = chooseDescentDirection(state, adjustedFun)
        dir.persist()
        dir.setName(s"Dir_${numSteps}")
        logDebug(s"Start persisting dir_${numSteps}")
        if ( (numSteps > 0) & (numSteps % interruptLinSteps == 0)) {
          logDebug(s"Interrupting lineage for dir_${numSteps}")
          dir.interruptLineage
        }
        dir.count()
        logDebug(s"End persisting dir_${numSteps}")

        /* determine the step size
         * if needed hold the Batch
         */
        if(holdBatch) adjustedFun.holdBatch()
        logDebug(s"Determining step size at iteration: ${numSteps}")
        val stepSize = determineStepSize(state, adjustedFun, dir)
        logDebug(f"Step Size determined: ${fieldImpl.opToDouble(stepSize)}%.4g")
        if(holdBatch) adjustedFun.stopHoldingBatch()

        // taking the step
        logDebug(s"Taking the step at iteration: ${numSteps}")
        val x = takeStep(state, dir, stepSize)
        x.persist()
        x.setName(s"x_${numSteps}")
        logDebug(s"Start persisting ${x.name}")
        if ( (numSteps > 0) & (numSteps % interruptLinSteps == 0)) {
          logDebug(s"Interrupting lineage for ${x.name}")
          x.interruptLineage
        }
        x.count()
        logDebug(s"End persisting ${x.name}")
        logDebug(s"Unpersisting ${dir.name}")
        dir.unpersist()

        // We now compute the new gradient
        logDebug(s"Computing the new gradient at iteration: ${numSteps}")
        val (value, grad) = calculateObjective(adjustedFun, x, state.history)

        // increment step
        logDebug(s"Increasing Iterative Step (grad will have new step number)")
        numSteps += 1

        // persist new gradient
        grad.persist()
        grad.setName(s"grad_${numSteps}")
        logDebug(s"Start persisting ${grad.name}")
        if ( (numSteps > 0) & (numSteps % interruptLinSteps == 0)) {
          logDebug(s"Interrupting lineage for ${grad.name}")
          grad.interruptLineage
        }

        grad.count()
        logDebug(s"End persisting ${grad.name}")

        // adjusting objective & gradient
        val (adjValue, adjGrad) = adjust(x, grad, value)
        adjGrad.persist()
        adjGrad.setName(s"adjGrad_${numSteps}")
        logDebug(s"Start persisting ${adjGrad.name}")
        if ( (numSteps > 0) & (numSteps % interruptLinSteps == 0)) {
          logDebug(s"Interrupting lineage for ${adjGrad.name}")
          adjGrad.interruptLineage
        }
        adjGrad.count()
        logDebug(s"End persisting ${adjGrad.name}")

        // measure improvement of objective
        val oneOffImprovement = fieldImpl.opToDouble(
          fieldImpl.opSub(state.adjustedValue, adjValue)) /
        (fieldImpl.opToDouble(state.adjustedValue).abs
          .max(fieldImpl.opToDouble(adjValue).abs)
          .max(1E-6 * fieldImpl.opToDouble(state.initialAdjVal).abs))

        logDebug(f"""Iteration ${numSteps}%d, adjusted objective and one off improvement: 
                  ${fieldImpl.opToDouble(adjValue)}%.6g (rel: $oneOffImprovement%.3g)""")

        // update history and check convergence
        val history = updateHistory(x, grad, value, adjustedFun, state)
        val newCInfo = convergenceCheck(x, grad, value, state, state.convergenceInfo)
        failedOnce = false

        // free resources from old state
        logDebug(s"Iteration ${numSteps}; freeing resources")
        state.x.unpersist()
        logDebug(s"Unpersisted ${state.x.name}")
        state.grad.unpersist()
        logDebug(s"Unpersisted ${state.grad.name}")
        state.adjustedGradient.unpersist()
        logDebug(s"Unpersisted ${state.adjustedGradient.name}")

        // create new state
        new State(
          x,
          value,
          grad,
          adjValue,
          adjGrad,
          state.iter + 1,
          state.initialAdjVal,
          history,
          false,
          newCInfo)

      } catch {

        case x: FirstOrderException if !failedOnce =>
          failedOnce = true
          logDebug("Failure! First failure so resetting history: ")
          state.copy(history = initialHistory(adjustedFun, state.x))

        case x: FirstOrderException =>
          logDebug(s"""Failure again! FirstOrderException: ${x.msg} 
                       Giving up and returning. Maybe the objective 
                       is just poorly behaved?""")
          state.copy(searchFailed = true, convergenceInfo = Some(
            FirstOrderMinimizer.SearchFailed))

      }
    }
  }

  // iterate till we hit convergence or fail
  def iterations(f: DF, init: T, initial_state: Option[State] = None): Iterator[State] = {

    // adjust objective
    val adjustedFun = adjustFunction(f)

    // stop with convergence result
    infiniteIterations(f, initialState(adjustedFun, init, initial_state)).takeUpToWhere { s =>

      s.convergenceInfo match  {

        case Some(converged) =>
          logDebug(s"Converged because ${converged.reason}")
          true

        case None =>
          false
      }
    }
  }

  def minimize(f: DF, init: T, initial_state: Option[State] = None): T = {
    minimizeAndReturnState(f, init, initial_state).x
  }

  def minimizeAndReturnState(f: DF, init: T, initial_state: Option[State] = None): State = {
    iterations(f, init, initial_state).last
  }

}

object FirstOrderMinimizer {

 /******************************************************
   State is a way to encode imperative actions
   in a functional setting (see Chiusano's book)
  ******************************************************/

  case class State[+T, +F, +History](
    x: T,
    value: F,
    grad: T,
    adjustedValue: F,
    adjustedGradient: T,
    iter: Int,
    initialAdjVal: F,
    history: History,
    searchFailed: Boolean = false,
    var convergenceInfo: Option[ConvergenceReason] = None) {

    def resetIters(): State[T, F, History] = State(
      this.x,
      this.value,
      this.grad,
      this.adjustedValue,
      this.adjustedGradient,
      0,
      this.initialAdjVal,
      this.history,
      false,
      None)

  }


  /******************************************************
   Reasons why a first order Method can converge
   ******************************************************/

  trait ConvergenceReason {

    def reason: String

  }

  case object MaxIterations extends ConvergenceReason {

    override def reason: String = "max iterations reached"

  }

  case object FunctionValuesConverged extends ConvergenceReason {

    override def reason: String = "function values converged"

  }

  case object GradientConverged extends ConvergenceReason {

    override def reason: String = "gradient converged"

  }

  case object SearchFailed extends ConvergenceReason {

    override def reason: String = "line search failed!"

  }


  case object ProjectedStepConverged extends ConvergenceReason {

    override def reason: String = "projected step converged"

  }
}
