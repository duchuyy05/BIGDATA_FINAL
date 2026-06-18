#!/bin/bash
echo 'Waiting for namenode to leave safe mode...'
hdfs dfsadmin -safemode wait

echo 'Creating directories...'
hdfs dfs -mkdir -p /spark/checkpoints /mlflow/artifacts

echo 'Upload completed.'
exit 0
