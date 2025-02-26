/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#ifndef __TEST_CASES_H__
#define __TEST_CASES_H__

#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include "Burst.h"
#include "FP16.h"
#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;

class basicFixture : public testing::Test
{
  public:
    basicFixture() {}
    ~basicFixture() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class MemBandwidthFixture : public testing::Test
{
  public:
    MemBandwidthFixture() {}
    ~MemBandwidthFixture() {}
    virtual void SetUp()
    {
        cur_cycle = 0;
        mem = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                    "system_hbm.ini", ".", "example_app", 256 * 16);
        mem_size = (uint64_t)getConfigParam(UINT, "NUM_CHANS") * getConfigParam(UINT, "NUM_RANKS") *
                   getConfigParam(UINT, "NUM_BANK_GROUPS") * getConfigParam(UINT, "NUM_BANKS") *
                   getConfigParam(UINT, "NUM_ROWS") * getConfigParam(UINT, "NUM_COLS");
        basic_stride =
            (getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8);
    }
    virtual void TearDown()
    {
        printResult(cur_cycle);
    }

    uint64_t measureCycle(bool is_write)
    {
        printTestMessage();
        write_ = is_write;
        generateMemTraffic(is_write);

        while (mem->hasPendingTransactions())
        {
            cur_cycle++;
            mem->update();
        }

        return cur_cycle;
    }

    void printTestMessage()
    {
        cout << ">> Bandwidth Test" << endl;
        cout << "  Num. channels: " << getConfigParam(UINT, "NUM_CHANS") << endl;
        cout << "  Num. ranks: " << getConfigParam(UINT, "NUM_RANKS") << endl;
        cout << "  Num. banks: " << getConfigParam(UINT, "NUM_BANKS") << endl;
        cout << "  Data size (byte): " << data_size_in_byte << endl;
    }

    void printResult(uint64_t cycle)
    {
        uint64_t totalReads = 0;
        uint64_t totalWrites = 0;

        for (size_t i = 0; i < getConfigParam(UINT, "NUM_CHANS"); i++)
        {
            MemoryController* mem_ctrl = mem->channels[i]->memoryController;
            totalReads += mem_ctrl->totalReads;
            totalWrites += mem_ctrl->totalWrites;
        }
        uint32_t bw = (totalReads + totalWrites) * getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") *
                      getConfigParam(UINT, "BL") / 8 / (cycle * getConfigParam(FLOAT, "tCK"));
        cout << endl;
        cout << "> Test Result " << endl;
        cout << "> BW (GB/s): " << bw << endl;
    }

    void setDataSize(unsigned size)
    {
        data_size_in_byte = size;
    }

    void generateMemTraffic(bool is_write)
    {
        int num_trans = 0;
        BurstType nullBst;

        for (uint64_t i = 0; i < mem_size; ++i)
        {
            if (num_trans >= (data_size_in_byte / basic_stride))
            {
                break;
            }
            uint64_t addr = i * basic_stride;
            mem->addTransaction(is_write, addr, &nullBst);
            num_trans++;
        }
    }

  private:
    bool write_;
    uint64_t cur_cycle = 0;
    uint64_t mem_size;
    uint64_t data_size_in_byte;
    uint64_t basic_stride;
    shared_ptr<MultiChannelMemorySystem> mem;
};

class DataDim
{
  private:
    unsigned getPrecisionToByte()
    {
        switch (PIMConfiguration::getPIMPrecision())
        {
            case INT8:
                return 1;
            case FP16:
                return 2;
            case FP32:
                return 4;
            // ($add) 64 bit precision for regular dram execution
            case INT64:
                return 8;
            default:
                return 0;
        }
    }

    void loadData(KernelType kn_type)
    {
        string input_dim_str = to_string(input_dim_);

        switch (kn_type)
        {
            case KernelType::GEMV:
            case KernelType::GEMVTREE:
            {
                string output_dim_str = to_string(output_dim_);

                string in_out_dim_str = output_dim_str + "x" + input_dim_str;

                if (batch_size_ > 1)
                {
                    string batch_size_str = to_string(batch_size_);
                    string batch_in_out_dim_str = batch_size_str + "_" + in_out_dim_str;
                    input_npbst_.loadFp16("data/gemv/gemv_input_batch_" + batch_in_out_dim_str +
                                          ".npy");
                    weight_npbst_.loadFp16("data/gemv/gemv_weight_batch_" + batch_in_out_dim_str +
                                           ".npy");
                    output_npbst_.loadFp16("data/gemv/gemv_output_batch_" + batch_in_out_dim_str +
                                           ".npy");
                }
                else
                {
                    input_npbst_.loadFp16("data/gemv/gemv_input_" + in_out_dim_str + ".npy");
                    weight_npbst_.loadFp16("data/gemv/gemv_weight_" + in_out_dim_str + ".npy");
                    output_npbst_.loadFp16("data/gemv/gemv_output_" + in_out_dim_str + ".npy");
                }

                // output_dim_ = weight_npbst_.bShape[0];
                output_dim_ = bShape1ToDim(output_npbst_.bShape[1]);
                input_dim_ = bShape1ToDim(input_npbst_.bShape[1]);
                batch_size_ = input_npbst_.bShape[0];
                return;
            }
            case KernelType::ADD:
            {
                // input_npbst_.loadFp16("data/add/resadd_input0_" + input_dim_str + ".npy");
                // input1_npbst_.loadFp16("data/add/resadd_input1_" + input_dim_str + ".npy");
                // output_npbst_.loadFp16("data/add/resadd_output_" + input_dim_str + ".npy");

                input_npbst_.loadint64("data/add/resadd_input0_u64_" + input_dim_str + ".npy");
                input1_npbst_.loadint64("data/add/resadd_input1_u64_" + input_dim_str + ".npy");
                output_npbst_.loadint64("data/add/resadd_output_u64_" + input_dim_str + ".npy");

                output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
                input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());

                return;
            }
            case KernelType::MUL:
            {
                // input_npbst_.loadFp16("data/mul/eltmul_input0_" + input_dim_str + ".npy");
                // input1_npbst_.loadFp16("data/mul/eltmul_input1_" + input_dim_str + ".npy");
                // output_npbst_.loadFp16("data/mul/eltmul_output_" + input_dim_str + ".npy");

                input_npbst_.loadint64("data/mul/eltmul_output_u64_" + input_dim_str + ".npy");
                input1_npbst_.loadint64("data/mul/eltmul_output_u64_" + input_dim_str + ".npy");
                output_npbst_.loadint64("data/mul/eltmul_output_u64_" + input_dim_str + ".npy");


                output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
                input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());

                return;
            }
            case KernelType::RELU:
            {
                input_npbst_.loadFp16("data/relu/relu_input_" + input_dim_str + ".npy");
                output_npbst_.loadFp16("data/relu/relu_output_" + input_dim_str + ".npy");

                output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());

                return;
            }
            case KernelType::KSKIP:
            {
                input_npbst_.loadint64("data/ksk/ksk_inputA_u64.npy");
                input1_npbst_.loadint64("data/ksk/ksk_inputB_u64.npy");
                input2_npbst_.loadint64("data/ksk/ksk_inputC_u64.npy");
                output_npbst_.loadint64("data/ksk/ksk_output_u64.npy");

                input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
                input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());
                input2_dim_ = bShape1ToDim(input2_npbst_.getTotalDim());
                output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());

                // ERROR("== Error - KSKIP KernelType not supported yet");
                // exit(2);
                return;
            }
            default:
            {
                ERROR("== Error - Unknown KernelType trying to load data");
                exit(-1);
                return;
            }
        }
    }

    void loadDummyData(KernelType kn_type)
    {
        switch (kn_type)
        {
            case KernelType::GEMV:
            case KernelType::GEMVTREE:
            {
                weight_npbst_.shape.push_back(output_dim_);
                weight_npbst_.shape.push_back(input_dim_);
                weight_npbst_.loadTobShape(16);

                input_npbst_.shape.push_back(batch_size_);
                input_npbst_.shape.push_back(input_dim_);
                input_npbst_.loadTobShape(16);

                for (int i = 0; i < input_npbst_.bShape[1]; i++)
                {
                    BurstType null_bst;
                    null_bst.set((float)0);
                    input_npbst_.bData.push_back(null_bst);
                }

                return;
            }
            case KernelType::ADD:
            case KernelType::MUL:
            case KernelType::RELU:
            {
                input_npbst_.shape.push_back(batch_size_);
                input_npbst_.shape.push_back(input_dim_);
                // printf("shape[0]: %d", (int)input_npbst_.shape[0]);
                // printf("shape[1]: %d", (int)input_npbst_.shape[1]);
                // ? Why is the divisor set to 16?
                // ? How is the data organized physically in memory?
                // Hypothesis
                // The data loaded has shape (batch_size, input_dim)
                // We need to partition the data accross each PIM BLOCK
                // There are 16x FP16 ALUs per PIM BLOCK
                // If we use 4x INT64 ALUS per PIM BLOCK, then divisor should be 4?

                input_npbst_.loadTobShape(16);
                // input_npbst_.loadDummyint64();

                output_npbst_.shape.push_back(batch_size_);
                output_npbst_.shape.push_back(output_dim_);

                output_npbst_.loadTobShape(16);
                // input_npbst_.loadDummyint64();

                return;
            }
            default:
            {
                return;
            }
        }
    }

  public:
    /* data */
    NumpyBurstType input_npbst_;
    NumpyBurstType input1_npbst_;
    NumpyBurstType weight_npbst_;
    NumpyBurstType output_npbst_;

    /* dump */
    NumpyBurstType preloaded_npbst_;
    NumpyBurstType result_npbst_;
    NumpyBurstType reduced_result_npbst_;

    size_t burst_cnt_;
    BurstType* preloaded_bst_;
    BurstType* result_;
    BurstType* reduced_result_;

    /* dim */
    unsigned long output_dim_;
    int input_dim_;
    int input1_dim_;
    int batch_size_;
    bool used_data_;


    // ($add) support for a third input for KSKIP (vector C) 
    int input2_dim_;
    NumpyBurstType input2_npbst_;

    // Overload the DataDim constructor for KSKIP kernel
    DataDim(KernelType kn_type, uint64_t batch_size, uint64_t output_dim, uint64_t input_dim,
            uint64_t vec_dim, bool used_data) 
    {
        batch_size  = batch_size;
        output_dim  = output_dim;
        input_dim_  = input_dim;
        // ! current implementation assumes data is all stored inside the same variable.
        // might be easier to have 3 variables (A, B and C)
        used_data_  = used_data;
        input2_dim_ = vec_dim;

        input1_dim_ = input2_dim_;

        // TODO implment the data loading inside loadData or loadDummydata (whichever is easier)
        if (used_data_){
            loadData(kn_type);
        }
        else{
            loadDummyData(kn_type);
        }
    }

    DataDim(KernelType kn_type, uint32_t batch_size, uint32_t output_dim, uint32_t input_dim,
            bool used_data)
    {
        batch_size_ = batch_size;
        output_dim_ = output_dim;
        input_dim_ = input_dim;
        used_data_ = used_data;

        switch (kn_type)
        {
            case KernelType::MUL:
            case KernelType::ADD:
            {
                input1_dim_ = input_dim;
                break;
            }
            default:
            {
                break;
            }
        }

        // load data from files
        if (used_data_)
            loadData(kn_type);
        else
            loadDummyData(kn_type);
    }

    uint32_t getDataSize(uint32_t dim1, uint32_t dim2 = 1, uint32_t dim3 = 1)
    {
        return dim1 * dim2 * dim3 * getPrecisionToByte();
    }

    void printDim(KernelType kn_type)
    {
        switch (kn_type)
        {
            case KernelType::GEMV:
            case KernelType::GEMVTREE:
            {
                cout << "  Weight data dimension : " << output_dim_ << "x" << input_dim_ << endl;
                if (batch_size_ > 1)
                {
                    cout << "  Input data dimension : " << input_dim_ << "x" << batch_size_ << endl;
                    cout << "  Output data dimension : " << output_dim_ << "x" << batch_size_
                         << endl;
                }
                else
                {
                    cout << "  Input data dimension : " << input_dim_ << endl;
                    cout << "  Output data dimension : " << output_dim_ << endl;
                }
                break;
            }
            case KernelType::MUL:
            case KernelType::ADD:
            case KernelType::RELU:
            {
                cout << "  Input/output data dimension : " << output_dim_ << endl;
                break;
            }
            case KernelType::KSKIP:
            case KernelType::HEMUL:
            {
                cout << "  InputA data dimension : " << input_dim_ << endl;
                cout << "  InputB data dimension : " << input1_dim_ << endl;
                cout << "  InputC data dimension : " << input2_dim_ << endl;
                cout << "  Output data dimension : " << output_dim_ << endl;
                break;
            }
            default:
            {
                ERROR("== Error - Unknown KernelType trying to load data");
                exit(-1);
                break;
            }
        }
    }
    uint32_t getNumElementsPerBlocks()
    {
        return ((getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8) /
                getPrecisionToByte());
    }

    uint32_t dimTobShape(int in_dim)
    {
        return ceil(in_dim / getNumElementsPerBlocks());
    }

    uint32_t bShape1ToDim(int bSahpe1)
    {
        return bSahpe1 * getNumElementsPerBlocks();
    }
};

#endif
