# Distributed Function Minimization in Apache Spark

GitHub code for the paper [http://arxiv.org/abs/1909.07922](http://arxiv.org/abs/1909.07922)

## Using the Library

* In `src/test/scala` you can find tests
that show how to use different library components.
The Scalability tests mentioned in the paper
are covered in `src/test/scala/ScalabilityExperiment_*`

* Note that the code in the tests was written to
be run in the spark-shell; cluster is set to local
mode but in reality was run on top of YARN.
In the future I plan to separate tests that
can be run in a "suite" without needing an actual
cluster, and "tests" to run in interactive mode,
e.g. those concerning scalability / benchmarks.

* `run_sparkshell_example` is an example of cluster setup
to run spark on. `project/log4j2.xml` is an example of
customizing logging.

* depending on the java libraries installed on your cluster
you might need to install jars for Apache Log4j **2**.