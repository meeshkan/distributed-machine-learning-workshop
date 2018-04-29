# TensorFlow

There are two popular solutions to distribute TensorFlow accross multiple nodes: Distributed TensorFlow and TensorFlowOnSpark.

This tutorial uses Distributed TensorFlow because there are several other spark-based solutions in this repository with which you may familiarize yourself with the Spark ecosystem. Distributed TensorFlow and TensorFlowOnSpark acheive similar performance, with the main difference being usability (comfort level with TF vs Spark) and a company's current infrastructure (use TensorFlowOnSpark if you already have a spark cluster).

This tutorial uses Imanol Schlag's [Distributed TensorFlow](https://github.com/ischlag/distributed-tensorflow-example) tutorial on the mNIST dataset with minor modifications for the purposes of this presentation.

```
ps$ python example.py --job_name="ps" --task_index=0 --ps_ip=10.0.0.0:3222 --worker1_ip=10.0.0.1:3222 --worker2_ip=10.0.0.2:3222 --worker3_ip=10.0.0.3:3222
worker-1$ python example.py --job_name="worker" --task_index=0 --ps_ip=10.0.0.0:3222 --worker1_ip=10.0.0.1:3222 --worker2_ip=10.0.0.2:3222 --worker3_ip=10.0.0.3:3222
worker-2$ python example.py --job_name="worker" --task_index=1 --ps_ip=10.0.0.0:3222 --worker1_ip=10.0.0.1:3222 --worker2_ip=10.0.0.2:3222 --worker3_ip=10.0.0.3:3222
worker-3$ python example.py --job_name="worker" --task_index=2 --ps_ip=10.0.0.0:3222 --worker1_ip=10.0.0.1:3222 --worker2_ip=10.0.0.2:3222 --worker3_ip=10.0.0.3:3222
```