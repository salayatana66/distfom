/******************************************************
 *  Traits used to define a distributed vector
 *****************************************************/

package distfom

trait VectorOps[B] {

  /** Element-wise sum of this and b. */
  def +(b: B) : B

  /** Element-wise product of this and b. */
  def *(b: B) : B

  /** Element-wise quotient of this and b.*/
  def /(b: B) :B

  /** Element-wise difference of this and b. */
  def -(b: B) : B

}

trait Persistable {

  def persist(): this.type

  def unpersist(blocking: Boolean = true): this.type

}

// basic distributed vector
trait DistVec[B] extends VectorOps[B] with Persistable

/* trait to add if we want to interrupt lineage
 * of spark computations; this is
 * very important in iterative algorithms
 */

trait InterruptableLineage {

  def interruptLineage() : Unit
}




