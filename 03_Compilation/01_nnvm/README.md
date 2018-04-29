# tvm and nnvm

[`nnvm`](https://github.com/dmlc/nnvm/) is a machine learning compiler that focuses on compiling Machine Learning models down to custom infrastructures including Raspberry-Pi, browsers, Android, iOS and various other operating systems.  [`tvm`](http://tvmlang.org) is a compilation stack for distributed hardware.  The [original `nnvm` announcement from October 2017](http://tvmlang.org/2017/10/06/nnvm-compiler-announcement.html) contains a useful chart that shows how NNVM and TVM interact to compile high-level deep learning graphs onto various architectures.

![compiler stack](/03_Compilation/01_nnvm/nnvm_compiler_stack.png)

To explore NNVM, we can look at the [TinyFlow](https://github.com/tqchen/tinyflow) project, which implements parts of the TensorFlow API.