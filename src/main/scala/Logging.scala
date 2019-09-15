/**********************************************************
 Based on the corresponding spark class; internal.Logging
 **********************************************************/

package distfom

import org.apache.logging.log4j.{LogManager, Logger}

trait Logging {

  /* to allow objects inheriting from Logging to be
   serializable
   */
  @transient private var log_ : Logger = null
  
  // Method to get the logger name for this object
  protected def logName = {
    // Ignore trailing $'s in the class names for Scala objects
    this.getClass.getName.stripSuffix("$")
  }

  // Method to get or create the logger for this object
  protected def log: Logger = {
    if (log_ == null) {
      log_ = LogManager.getLogger(logName)
    }
    log_
  }

  // lazy eval of debug message
  protected def logDebug(msg: => String) {
    log.debug(msg)
  }

}
