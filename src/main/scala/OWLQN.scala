/**
 * Implementation of OWLQN
 * Built on top of
 * https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/optimize/OWLQN.scala
 *
 * @author Andrea Schioppa
 *
 * See the paper Scalable Training of L1-Regularized Log-Linear Models by Andrew & Gao
 */


package distfom

import scala.language.implicitConversions

import scala.reflect.runtime.universe._

import scala.reflect.ClassTag

import CommonImplicitGenericFields._

import CommonImplicitGenericSmoothFields._

/**
 * OWLQN extending LBFGS
 * @param m memory parameter
 * @param l1reg this function allow to fine-tune the l1-regularization applied to different components of
 *              the state.x
 * @param maxIter
 * @param tolerance
 * @param lowerFunTolerance
 * @param suppliedLineSearch optionally supplied Line Search (Otherwise a Strong Wolfe gets used)
 * @param maxLineSearchIter
 * @param baseLSearch step to pass to the first line search
 * @param interruptLinSteps
 * @param holdBatch
 * @param tagK
 * @param functionalTag
 * @param tagF
 * @param fieldImpl
 * @param smoothImpl
 * @tparam K implicit tag for the key lookup in grouping elements for l1-regularization
 * @tparam B generic used by vector type
 * @tparam F numeric field
 */
class OWLQN[K, B, F: ClassTag](m: Int,
  val l1reg : K => F,
  override val maxIter: Int = -1,
  override val tolerance: Double = 1E-6,
  override val lowerFunTolerance: Double = 0.0,
  override val suppliedLineSearch: Option[MinimizingLineSearch[F]] = None,
  maxLineSearchIter: Int = 10,
  override val baseLSearch: Double = 1.0,
  override val interruptLinSteps: Int = 4,
  override val holdBatch : Boolean = true)(implicit tagK: TypeTag[K],
    functionalTag: TypeTag[(K, F, F) => F],
    tagF: TypeTag[F],
    fieldImpl: GenericField[F], smoothImpl: GenericSmoothField[F])
    extends LBFGS[B, F](m = m, maxIter = maxIter,
      tolerance = tolerance,
      lowerFunTolerance = lowerFunTolerance,
      suppliedLineSearch = suppliedLineSearch,
      maxLineSearchIter = maxLineSearchIter,
      baseLSearch = baseLSearch,
      interruptLinSteps = interruptLinSteps,
    holdBatch = holdBatch) {

  import scala.math

  private def sigma(x : F): F = if(x != fieldImpl.zero) smoothImpl.opSignum(x)
  else fieldImpl.zero

  // proj is pi(x,y) in the paper
  private def proj(x: T, y: T) = {

    val pfun = (a: F, b: F) => if(sigma(a) == sigma(b)) a else fieldImpl.zero

    x.pairWiseFun(y)(pfun)
  }

  // function that represents equation 4 in the paper
  private val regGrad = (idx: K, x: F, lg: F)  => {
    // retrieve regularization for feature index
    val l1regValue = l1reg(idx)
    require(fieldImpl.lessOrEqual(fieldImpl.zero , l1regValue))

    val outGrad = if(l1regValue == fieldImpl.zero) lg else {
      val dplus = if(x == fieldImpl.zero)
        fieldImpl.opAdd(lg , l1regValue)
      else
        fieldImpl.opAdd(lg , fieldImpl.opMul(l1regValue, sigma(x)))

      val dminus = if(x == fieldImpl.zero)
        fieldImpl.opSub(lg, l1regValue)
      else
        fieldImpl.opAdd(lg, fieldImpl.opMul(l1regValue, sigma(x)))

      if (fieldImpl.lessThan(fieldImpl.zero, dminus)) dminus
      else if (fieldImpl.lessThan(fieldImpl.zero, dplus)) dplus
      else fieldImpl.zero
    }

    outGrad
  }

  /** function that multiplies gradient weights by
   *  regularization constant for indices of features
   */
  private val regWeight = (idx: K, x: F, lg: F)  => {

    // retrieve regularization for feature index
    val l1regValue = this.l1reg(idx)
    require(fieldImpl.lessOrEqual(fieldImpl.zero, l1regValue))

    fieldImpl.opMul(l1regValue, lg)

  }


  // Adjust function and gradient
  override protected def adjust(newX: T, newGrad: T, newVal: F): (F, T) = {

    // adjust the gradient
    val adjGrad = newX.keyedPairWiseFun(newGrad)(regGrad)

    // l1 correction (newX is not really used ;))
    val l1Corr = newX.keyedPairWiseFun(newGrad)(regWeight).norm(1.0)

    (fieldImpl.opAdd(newVal, l1Corr), adjGrad)
  }

  override protected def chooseDescentDirection(state: State, fn: DF) = {
    val descentDir = super.chooseDescentDirection(state.copy(grad = state.adjustedGradient), fn)

  /** Note of the authors of breeze
    * The original paper requires that the descent direction be corrected to be
    * in the same directional (within the same hypercube) as the adjusted gradient for proof.
    * Although this doesn't seem to affect the outcome that much in most of cases, there are some cases
    *where the algorithm won't converge (confirmed with the author, Galen Andrew).

    */
    val correctedDir = proj(descentDir, state.adjustedGradient *
      fieldImpl.opFromDouble(-1.0))

    correctedDir
  }

  // orthant construction
  def getOrthant(x: T, g: T): T = {
    val selector = (a: F, b: F) => if (a != fieldImpl.zero) a
    // represent -b
    else fieldImpl.opSub(fieldImpl.zero, b)

    x.pairWiseFun(g)(selector)

  }

  override protected def takeStep(state: State, dir: T, stepSize: F) = {
    val baseStep = state.x + dir*stepSize

    // project on the adjusted Gradient
    proj(baseStep, getOrthant(state.x, state.adjustedGradient))
  }

  /** the step alpha is taken on
   * f \circ proj(x + alpha*descent_dir, adjGrad)
   * See Sec 3.2 in the paper
   */
  override protected def determineStepSize(state: State, f: DF, dir: T) = {
    val iter = state.iter

    // we construct an adjusted function that is the composition

    val fcompProj = new DistributedDiffFunction[F, F] {

      def compute(alpha: F) = {

        /** the value and grad are first computed at the new
         *  position
         */
        
        val newX = takeStep(state, dir, alpha)
        val (v, newG) = f.compute(newX)

        /* we need to adjust for the value via l1 regularization
         * the new gradient is adjuste via the projection
         */

        val (adjv, adjgrad) = adjust(newX, newG, v)
        
        (adjv, adjgrad.dot(dir))
      }

      override def computeGrad(alpha: F) = compute(alpha)._2

      override def computeValue(alpha: F) = compute(alpha)._1

      override def holdBatch = f.holdBatch

      override def stopHoldingBatch = f.stopHoldingBatch

    }

    // either take the supplied line search or by default the BackTrackingLineSearch
    val search = suppliedLineSearch match{
      case None => new BackTrackingLineSearch[F](
        shrinkStep = if (iter < 1) 0.1 else 0.5,
        maxIterations = maxLineSearchIter)
      case Some(q) => q
    }// TODO: Need good default values here.

    val alpha = search.minimize(fcompProj, if (iter == 1) baseLSearch else 1.0)

    alpha
  }

}


