/**
 * Implementations of Line Searches
 * Built on top of 
 * https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/optimize/FirstOrderMinimizer.scala
 *
 * @commit c2193a11cc117940ec2de71b724c65de8a09a62c
 * @author Andrea Schioppa
 *
 * @todo Am not completely satisfied in the way Line Searches are implemented in breeze
 * a more direct class - hierarchy and more flexibility in the steps and supplying steps for the first
 * few iterations would be great
 */


package distfom

import scala.language.implicitConversions

// TODO: remove distlbfgs. when moving up
import RichIterator._

// abstract line search
trait MinimizingLineSearch[@specialized(Float, Double) F] {

  /** to import DD as a distributed diff function
   *  of a double
   */

  type DD[R] = DistributedDiffFunction[R, R]

  def minimize(f: DD[F], init: Double): F
}

object LineSearch {

  type DD[R] = DistributedDiffFunction[R, R]
  type FV[B, F] = FomDistVec[B, F]
  type DF[B, F] = DistributedDiffFunction[B, F]

  /** Given f, x and dir construct
   *  alpha => f(x+dir*alpha)
   */
  def functionFromSearchDirection[B, F, S <: FV[B, F]](
    f: DF[FV[B, F], F], x: S, direction: S): DistributedDiffFunction[F, F] =
    new DistributedDiffFunction[F, F] with Serializable {

      // calculates the value at a point
      override def computeValue(alpha: F): F = {

        val newX = direction * alpha

        f.computeValue(newX + x)

      }

      // calculates the value and the gradient
      override def compute(alpha: F): (F, F) = {

        val newX = direction * alpha
        val (ff, grad) = f.compute(newX+x)

        (ff, grad.dot(direction) )

      }

      override def computeGrad(alpha: F): F = {

        val newX = direction*alpha
        val grad = f.computeGrad(newX+x)

        grad.dot(direction)
      }

      override def holdBatch() = f.holdBatch()
      override def stopHoldingBatch() = f.stopHoldingBatch()

    }
}

case class LineSearchException(val msg: String = "") extends RuntimeException(msg)

import CommonImplicitGenericSmoothFields._

class StrongWolfeLineSearch[@specialized(Float, Double) F](maxZoomIter: Int,
  maxLineSearchIter: Int, growStep: Double = 1.5)(implicit fieldImpl: GenericField[F]) extends MinimizingLineSearch[F] with
    Logging with Serializable {

  import scala.math._

  // values from nocedal & wright
  val c1 = 1e-4
  val c2 = 0.9

  // the bracket is just (t, phi'(t), phi(t))
  case class Bracket(
    t: Double, // 1d line search parameter
    dd: Double, // Directional Derivative at t
    fval: Double // Function value at t
  )

  // cubit interpolator
  def interp(l: Bracket, r: Bracket) = {

    /** See N&W p57 actual for an explanation of the math
    * This essentially uses 3 values of phi & its derivative
     to fit a cubit and find a minimizer for the cubic
     */
    val d1 = l.dd + r.dd - 3 * (l.fval - r.fval) / (l.t - r.t)
    val d2 = sqrt(d1 * d1 - l.dd * r.dd)
    val multiplier = r.t - l.t
    val t = r.t - multiplier * (r.dd + d2 - d1) / (r.dd - l.dd + 2 * d2)
    logDebug(s"New minimizer of cubic interpolant for interval [$l.t, $r.t]: $t")

    // If t is too close to either end bracket, move it closer to the middle
    val lwBound = l.t + 0.1 * (r.t - l.t)
    val upBound = l.t + 0.9 * (r.t - l.t)

    t match {
      case _ if t < lwBound =>
        logDebug(s"Minimizer $t < $lwBound, moving it to $lwBound")
        lwBound
      case _ if t > upBound =>
        logDebug(s"Minimizer $t > $upBound, moving it to $upBound")
        upBound
      case _ => t
    }
  }

  def minimize(f: DD[F], init: Double): F = {
    minimizeWithBound(f, init, bound = Double.PositiveInfinity)
  }

  
  def minimizeWithBound(f: DD[F], init: Double, bound: Double): F = {

    require(init <= bound, s"init value $init should <= bound $bound")

    def phi(t: Double): Bracket = {
      val (pval, pdd) = f.compute(fieldImpl.opFromDouble(t))
      Bracket(t = t, dd = fieldImpl.opToDouble(pdd),
        fval = fieldImpl.opToDouble(pval))
    }

    var t = init
    var low = phi(0.0)
    val fval = low.fval
    val dd = low.dd
    logDebug(s"phi(0) = $fval, phi'(0) = $dd")

    if (dd > 0) {
      throw new LineSearchException(s"Line search invoked with non-descent direction: phi'(0) = $dd")
    }

    def zoom(linit: Bracket, rinit: Bracket): Double = {
      var low = linit
      var hi = rinit

      for (i <- 0 until maxZoomIter) {
        // Interpolant assumes left less than right in t value, so flip if needed
        val t = if (low.t > hi.t) interp(hi, low) else interp(low, hi)

        // Evaluate objective at t, and build bracket
        logDebug(s"Zoom: evaluating bracket for phi @t=$t")
        val c = phi(t)

        val armLin = fval + c1*c.t*dd
        logDebug(s"Zoom: phi(t) = ${c.fval}, phi'(t)=${c.dd}, Armijo linearization = $armLin")

        ///////////////
        /// Update left or right bracket, or both

        if (c.fval > fval + c1 * c.t * dd || c.fval >= low.fval) {
          // "Sufficient decrease" condition not satisfied by c. Shrink interval at right
          hi = c
          logDebug(s"Zoom: decrease not satisfied: hi=c($c)")

        } else {

          // Zoom exit condition is the "curvature" condition
          // Essentially that the directional derivative is large enough
          if (abs(c.dd) <= c2 * abs(dd)) {
            logDebug(s"Zoom: curvature condition satisfied, returning ${c.t}")
            return c.t
          }

          // If the signs don't coincide, flip left to right before updating l to c

          if (c.dd * (hi.t - low.t) >= 0) {
            logDebug(s"Zoom: flipping hi($hi) & low($low)")
            hi = low
          }

          // If curvature condition not satisfied, move the left hand side of the
          // interval further away from t=0.
          logDebug(s"Zoom: move low($low) to ($c)")
          low = c

        }

      }

      throw LineSearchException(s"Line search zoom failed")
      
    }


    for (i <- 0 until maxLineSearchIter) {
      val c = phi(t)

      // If phi has a bounded domain, inf or nan usually indicates we took
      // too large a step.
      if (java.lang.Double.isInfinite(c.fval) || java.lang.Double.isNaN(c.fval)) {
        t /= 2.0
        logDebug("Encountered bad values in function evaluation. Decreasing step size to " + t)
      } else {

        // Zoom if "sufficient decrease" condition is not satisfied
        val armijioLinearization = fval + c1 * t * dd
        if ((c.fval > armijioLinearization) ||
          (c.fval >= low.fval && i > 0)) {

          logDebug(s"""Either
          phi(t) = ${c.fval} > Armijo linearization = $armijioLinearization
           Or phi(t) = ${c.fval} >= value ${low.fval} at t' = ${low.t}
           """)
          logDebug(s"Applying zoom in interval [${low.t}, ${c.t}]")
          return fieldImpl.opFromDouble(zoom(low, c))

        }  // We don't need to zoom at all
           // if the strong wolfe condition is satisfied already.

        // Corrected wrt to breeze code, Nocedal pg.59
        if (abs(c.dd) <= -c2 * dd) {

          logDebug(
            s"""
               |Strong Wolfe condition is satisfied:
               ||phi'(t)| = ${abs(c.dd)} <= c2|phi'(0)| = ${c2 * abs(dd)}
             """.stripMargin)
          return fieldImpl.opFromDouble(c.t)
        }

        // If c.dd is positive, we zoom on the inverted interval.
        // Occurs if we skipped over the nearest local minimum
        // over to the next one.
        if (c.dd >= 0) {

         logDebug( s"""
             ||phi'(t)| = ${c.dd} >= 0
             |We can zoom in [${c.t}, $low]
             |to decrease phi further
           """.stripMargin
         )

           return fieldImpl.opFromDouble(zoom(c, low))
        }

        low = c
        if (t == bound) {
          logDebug(
            s"t = $t reached bound=$bound; it satisfies Armijio sufficient decrease condition but not the Wolfe curvature condition")

          return fieldImpl.opFromDouble(bound)
        } else {
          t *= growStep

          if (t > bound) {
            t = bound
          }

          logDebug(s"""
          Sufficient Decrease condition but not curvature condition satisfied.
            |phi'(t)| = ${abs(c.dd)} > c2|phi'(0)| = ${c2 * abs(dd)}
           Increased t to: $t
            """)
        }
      }
    }

    throw LineSearchException("Line search failed")
  }
  
}

/**
  * A line search optimizes a function of one variable without
  * analytic gradient information. It's often used approximately (e.g. in
  * backtracking line search), where there is no intrinsic termination criterion, only extrinsic
  * @author dlwh
  */
trait ApproximateLineSearch[@specialized(Float, Double) F]
    extends MinimizingLineSearch[F] with Logging {

  // state during search
  case class State(alpha: F, value: F, deriv: F)

  def iterations(f: DD[F], init: Double = 1.0): Iterator[State]

  def minimize(f: DD[F], init: Double = 1.0): F = iterations(f, init).last.alpha

}

class BackTrackingLineSearch[F](maxIterations: Int = 20,
    shrinkStep: Double = 0.5,
  growStep: Double = 2.0,
    cArmijo: Double = 1E-4,
    cWolfe: Double = 0.9,
    minAlpha: Double = 1E-10,
    maxAlpha: Double = 1E10,
    enforceWolfeConditions: Boolean = true,
    enforceStrongWolfeConditions: Boolean = true)(implicit fieldImpl: GenericField[F])
    extends ApproximateLineSearch[F] with Serializable {

  // TODO can we take this out?
  //require(shrinkStep * growStep != 1.0, "Can't do a line search with growStep * shrinkStep != 1.0")
  require(cArmijo < 0.5)
  require(cArmijo > 0.0)
  require(cWolfe > cArmijo)
  require(cWolfe < 1.0)

  def iterations(f: DD[F], init: Double = 1.0): Iterator[State] = {
    // value & derivative at the current point
    val Tuple2(f0, df0) = f.compute(fieldImpl.zero)
    logDebug(s"phi(0) = $f0, phi'(0) = $df0")

    val (initfval, initfderiv) = f.compute(fieldImpl.opFromDouble(init))

    Iterator
      .iterate((State(fieldImpl.opFromDouble(init),
        initfval, initfderiv), false, 0)){
        case (state @ State(alpha, fval, fderiv), _, iter) => {

          /* convert the state variable to doubles for easy 
           * arithmetic comparisons
           */
          val fval_double = fieldImpl.opToDouble(fval)
          val alpha_double = fieldImpl.opToDouble(alpha)
          val fderiv_double = fieldImpl.opToDouble(fderiv)
          val f0_double = fieldImpl.opToDouble(f0)
          val df0_double = fieldImpl.opToDouble(df0)

          val armijoLin = f0_double + alpha_double * df0_double * cArmijo
          logDebug(s"phi($alpha_double) = $fval_double, Armijo Linearitzation = $armijoLin")
          val multiplier = if (fval_double > armijoLin) {
            logDebug(s"Armijo Sufficient Decrease Condition is violated, shrinking step by $shrinkStep")
            shrinkStep
          }
          else if (enforceWolfeConditions && (fderiv_double < cWolfe * df0_double)) {
            logDebug(
              s"""
                 |Curvature condition is violated: phi'(alpha) = $fderiv_double <
                 |cWolfe * phi'(0) = ${cWolfe * df0_double}
                 |Dilating step by $growStep
               """.stripMargin)
            growStep
          }
          else if (enforceStrongWolfeConditions && (fderiv_double > -cWolfe * df0_double)) {
            logDebug(
              s"""
                 |Strong Wolfe Curvature Condition is violated: phi'(alpha) = $fderiv_double >
                 |-cWolfe*phi'(0) =  ${-cWolfe * df0_double}
                 |Shrinking step by $shrinkStep
               """.stripMargin)
            shrinkStep
          }
          else 1.0

          if (multiplier == 1.0) {
            (state, true, iter)
          } else {

            val newAlpha = fieldImpl.opFromDouble(alpha_double * multiplier)

            val newAlpha_double = fieldImpl.opToDouble(newAlpha)

            logDebug(s"New alpha: $alpha => ${newAlpha}")

            if (iter >= maxIterations) {
              throw new LineSearchFailed(s"Too many iterations $iter")
            } else if (newAlpha_double < minAlpha) {
              throw StepSizeUnderflow
            } else if (newAlpha_double > maxAlpha) {
              throw StepSizeOverflow
            }

             val (fvalnew, fderivnew) = f.compute(newAlpha)
            logDebug(s"phi(alpha) = $fvalnew, phi'(alpha)=$fderivnew")
            (State(newAlpha, fvalnew, fderivnew), false, iter + 1)
        } 

        
        }
      }
       .takeWhile(triple => !triple._2 && (triple._3 < maxIterations))
      .map(_._1)

  }

}
 
