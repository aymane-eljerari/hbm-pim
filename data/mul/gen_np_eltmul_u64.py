import numpy as np

DIM_IN = 1024 * 1024

np.set_printoptions(precision=20)
np.random.seed(1113)
data_in1 = np.random.rand(DIM_IN).astype('uint64')
data_in2 = np.random.rand(DIM_IN).astype('uint64')
data_out = np.zeros(DIM_IN).astype('uint64')

data_out = data_in1 * data_in2

np.save("eltmul_input0_u64_" + str(DIM_IN), data_in1)
np.save("eltmul_input1_u64_" + str(DIM_IN), data_in2)
np.save("eltmul_output_u64_" + str(DIM_IN), data_out)

print(data_in1)
print(data_in2)
print(data_out)
