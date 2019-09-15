package distfom

import scala.language.implicitConversions

class RichIterator[T](iter: Iterator[T]) {
    def takeUpToWhere(f: T => Boolean): Iterator[T] = new Iterator[T] {

      var done = false

      def next = {
        if (done) throw new NoSuchElementException()
        val n = iter.next;
        done = f(n)
        n
      }
    

      def hasNext = {
        !done && iter.hasNext;
      }
    }

    def last = {
      var x = iter.next()
      while (iter.hasNext) {
        x = iter.next()
      }
         x
    }
      
    
}

object RichIterator {

  implicit def scEnrichIterator[T](iter: Iterator[T]) : RichIterator[T]
  = new RichIterator[T](iter)
}
