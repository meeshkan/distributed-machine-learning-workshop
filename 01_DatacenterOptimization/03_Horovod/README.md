# Horovod

[Horovod](https://eng.uber.com/horovod/) is a tool developed by Uber that allows for the distribution of TensorFlow models at scale using an algorithm called ring-allreduce.  Uber's benchmarks show that it provides better results than TensorFlow lite when used on large datasets and has more robust documentation and examples.

To show how Horovod works, we will use the simple example given on their [GitHub](https://github.com/uber/horovod#why-not-traditional-distributed-tensorflow) page.

