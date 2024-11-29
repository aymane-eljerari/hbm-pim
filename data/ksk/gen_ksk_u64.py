import numpy as np

# Size of A: 2^16 * 32 * 3 uint64s
# Size of B: 2^16 * 32 * 3 uint64s
# Size of C: 32 uint64s

LIMBS=32
COEFFS=2**16
DNUM=3

ELEM_LEN = 32
DATA_WORD_LEN = 2**16 * ELEM_LEN * 3

print((2*DATA_WORD_LEN + ELEM_LEN) / 2**20)

np.random.seed(1113)
data_in1 = np.random.randint(0, 2**64, size=(DNUM, ELEM_LEN, COEFFS), dtype=np.uint64)
data_in2 = np.random.randint(0, 2**64, size=(DNUM, ELEM_LEN, COEFFS), dtype=np.uint64)
data_in3 = np.random.randint(0, 2**64, size=LIMBS, dtype=np.uint64)
data_out = np.zeros((ELEM_LEN, COEFFS), dtype=np.int128)

for k in range(DNUM):
    for i in range(LIMBS):
        for j in range(COEFFS):
            # This encounters an overflow
            data_out[i, j] = ((data_in1[k, i, j] % data_in3[i]) + (data_in2[k, i, j] % data_in3[i])) % data_in3[i]

np.save("./data/ksk/ksk_inputA_u64", data_in1)
np.save("./data/ksk/ksk_inputB_u64", data_in2)
np.save("./data/ksk/ksk_inputC_u64", data_in3)
np.save("./data/ksk/ksk_output_u64", data_out)

print(data_in1)
print(data_in2)
print(data_in3)
print(data_out)
