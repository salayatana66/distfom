package distfom

import breeze.linalg.{DenseVector => DV}

/* generic field operations
 * for comparisons we also allow conversion to a Double
 */
abstract class GenericField[A] extends Serializable {

  def opAdd(x: A, y: A): A
  def opSub(x: A, y: A): A
  def opMul(x: A, y: A): A
  def opDiv(x: A, y: A): A
  // a * x + b * y
  def opLinComb(a: A, x: A, b: A, y: A): A
  def opInv(x: A): A
  def opToDouble(x: A): Double

  def opFromDouble(d: Double) :A

  // comparisons
  def lessThan(a: A, b: A) : Boolean
  def lessOrEqual(a: A, b: A) : Boolean
  // zero Element
  val zero: A
}

object CommonImplicitGenericFields {

  implicit val floatField : GenericField[Float] = new GenericField[Float] {

    def opAdd(x: Float, y: Float) = x + y
    def opSub(x: Float, y: Float) = x - y
    def opMul(x: Float, y: Float) = x * y
    def opDiv(x: Float, y: Float) = x / y
    def opLinComb(a: Float, x: Float, b: Float, y: Float) = a*x+b*y
    def opInv(x: Float) = 1.0f/x
    def opToDouble(x: Float) = x.toDouble
    def opFromDouble(x: Double) = x.toFloat
    def lessThan(x: Float, y: Float)= x < y
    def lessOrEqual(x: Float, y: Float)= x <= y
    val zero = 0.0f
  }

  implicit val doubleField : GenericField[Double] = new GenericField[Double] {

    def opAdd(x: Double, y: Double) = x + y
    def opSub(x: Double, y: Double) = x - y
    def opMul(x: Double, y: Double) = x * y
    def opDiv(x: Double, y: Double) = x / y
    def opLinComb(a: Double, x: Double, b: Double, y: Double) = a*x+b*y
    def opInv(x: Double) = 1.0/x
    def opToDouble(x: Double) = x.toDouble
    def opFromDouble(x: Double) = x
    def lessThan(x: Double, y: Double)= x < y
    def lessOrEqual(x: Double, y: Double)= x <= y
    val zero = 0.0
  }

}

/* generic smooth field operations
 * e.g. powers
 */
abstract class GenericSmoothField[A] extends Serializable {

  // |x|^p
  def opAbsPow(x: A, p: Double): A
  def opDot(x: DV[A], y: DV[A]): A
  def opSignum(x: A): A
}

object CommonImplicitGenericSmoothFields {

  implicit val floatSmoothField : GenericSmoothField[Float] = new GenericSmoothField[Float] {

    def opAbsPow(x: Float, p: Double) : Float = p match {

      case 2.0f => x*x
      case 0.5f => math.sqrt(x).toFloat
      case 1.0f => math.abs(x).toFloat
      case -1.0f => (1.0f/math.abs(x)).toFloat
      case _ => math.pow(math.abs(x), p).toFloat
    }

    def opDot(x: DV[Float], y: DV[Float]) = x.dot(y)

    def opSignum(x: Float) = math.signum(x)
  }

  implicit val doubleSmoothField : GenericSmoothField[Double] = new GenericSmoothField[Double] {

    def opAbsPow(x: Double, p: Double): Double = p match {
      case 2.0 => x*x
      case 0.5 => math.sqrt(x)
      case 1.0 => math.abs(x)
      case -1.0 => (1.0f/math.abs(x))
     
      case _ => math.pow(math.abs(x), p)
    }
    def opDot(x: DV[Double], y: DV[Double]) = x.dot(y)
    def opSignum(x: Double) = math.signum(x)
  }

}

 
