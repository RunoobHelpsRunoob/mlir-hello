// RUN: hello-opt %s | FileCheck %s

// CHECK: define void @main()
func.func @main() {
    %0 = "hello.constant"() {value = dense<[[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tensor<2x2xf64>} : () -> tensor<2x2xf64>
    %1 = "hello.constant"() {value = dense<[[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tensor<2x2xf64>} : () -> tensor<2x2xf64>
    %2 = "hello.mul"(%0, %1)  : (tensor<2x2xf64>, tensor<2x2xf64>) -> (tensor<2x2xf64>)
    "hello.print"(%2) : (tensor<2x2xf64>) -> ()
    return
}
