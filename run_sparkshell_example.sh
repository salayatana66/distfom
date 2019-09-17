#!/usr/bin/env bash

# Jars for log4j2 can be retrieved from mvn & might
# be needed depending on your spark / java installation
# If you use auto-paste-from-script just compile & add the jar
/opt/spark/bin/spark-shell \
    --conf spark.speculation=true \
    --conf spark.driver.memory=20g \
  --conf spark.dynamicAllocation.maxExecutors=400 \
    --conf spark.executor.memory=8g \
    --conf spark.executor.cores=2 \
    --conf spark.network.timeout=800 \
    --jars /home/$USER/distfom_2.11-0.1.jar,/home/$USER/log4j-core-2.11.1.jar,/home/$USER/log4j-api-2.11.1.jar\
    --driver-java-options -Dlog4j.configurationFile=project/log4j2.xml -Dsun.io.serialization.extendedDebugInfo=true \
    --conf spark.executor.extraJavaOptions="-Dsun.io.serialization.extendedDebugInfo=true" \
    --conf spark.speculation=true \
    --conf spark.rdd.compress=true \
    --conf spark.executor.memoryOverhead=1g
 
    



