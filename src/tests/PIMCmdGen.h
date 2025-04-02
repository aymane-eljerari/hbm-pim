/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 *distributed, transmitted, transcribed, stored in a retrieval system, or
 *translated into any human or computer language in any form by any
 *means,electronic, mechanical, manual or otherwise, or disclosed to third
 *parties without the express written permission of Samsung Electronics. (Use of
 *the Software is restricted to non-commercial, personal or academic, research
 *purpose only)
 **************************************************************************************************/

#ifndef __PIM_KERNEL_GEN_H__
#define __PIM_KERNEL_GEN_H__

#include "fhe_params.h"
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "SystemConfiguration.h"
#include "tests/KernelAddrGen.h"

using namespace std;
using namespace DRAMSim;

class IPIMCmd
{
public:
  IPIMCmd(KernelType ktype) : kernelType(ktype) {}
  virtual vector<PIMCmd> generateKernel(int num_jump_to_be_taken,
                                        int num_jump_to_be_taken_odd_bank,
                                        int num_jump_to_be_taken_even_bank) = 0;

protected:
  KernelType kernelType;
};

class EltwisePIMKernel : public IPIMCmd
{
public:
  EltwisePIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd>
  generateKernel(int num_jump_to_be_taken,
                 int num_jump_to_be_taken_odd_bank = 0,
                 int num_jump_to_be_taken_even_bank = 0) override
  {
    vector<PIMCmd> pim_cmds;
    PIMCmdType pimType = getPIMCmdType();
    vector<PIMCmd> tmp_cmds{
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        PIMCmd(pimType, PIMOpdType::GRF_A, PIMOpdType::GRF_A,
               PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_B, PIMOpdType::ODD_BANK),
        PIMCmd(pimType, PIMOpdType::GRF_B, PIMOpdType::GRF_B,
               PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7)};
    pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }

private:
  PIMCmdType getPIMCmdType()
  {
    if (kernelType == KernelType::ADD)
      return PIMCmdType::ADD;
    else if (kernelType == KernelType::MUL)
      return PIMCmdType::MUL;
    else
      throw invalid_argument("Not supported element-wise operation");
  }
};

class KSKIPPIMKernel : public IPIMCmd
{
public:
  KSKIPPIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd>
  generateKernel(int num_jump_to_be_taken,
                 int num_jump_to_be_taken_odd_bank = 0,
                 int num_jump_to_be_taken_even_bank = 0) override
  {
    vector<PIMCmd> pim_cmds;
    vector<PIMCmd> tmp_cmds{
        PIMCmd(PIMCmdType::NOP, 1),
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        // MUL+MOD
        PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_A, PIMOpdType::GRF_A,
               PIMOpdType::EVEN_BANK, 1),
        // ADD+MOD
        // accumulation in GRF_B
        PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_A, PIMOpdType::GRF_A,
               PIMOpdType::GRF_B, 1),
        // dnum loop
        PIMCmd(PIMCmdType::JUMP, FHE_DNUM - 1, 4),

    };
    pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }
};

class TPRODPIMKernel : public IPIMCmd
{
public:
  TPRODPIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd>
  generateKernel(int num_jump_to_be_taken,
                 int num_jump_to_be_taken_odd_bank = 0,
                 int num_jump_to_be_taken_even_bank = 0) override
  {
    vector<PIMCmd> pim_cmds;
    vector<PIMCmd> tmp_cmds{
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        // MUL+MOD
        PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_A, PIMOpdType::GRF_A,
               PIMOpdType::EVEN_BANK, 1),
        // ADD+MOD
        PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_A, PIMOpdType::GRF_A,
               PIMOpdType::GRF_A, 1),
        // 1st iter a0 b0
        // 2nd iter a1 b1
        // 3rd - 4th iters intermediate results for a1b0 + a0b1
        PIMCmd(PIMCmdType::JUMP, 3, 4),

        // accumulation a1b0 + a0b1
        PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::GRF_A)

    };
    pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }
};

class ActPIMKernel : public IPIMCmd
{
public:
  ActPIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd>
  generateKernel(int num_jump_to_be_taken,
                 int num_jump_to_be_taken_odd_bank = 0,
                 int num_jump_to_be_taken_even_bank = 0) override
  {
    vector<PIMCmd> pim_cmds;
    if (kernelType == KernelType::RELU)
    {
      vector<PIMCmd> tmp_cmds{PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A,
                                     PIMOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
                              PIMCmd(PIMCmdType::NOP, 7),
                              PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_B,
                                     PIMOpdType::ODD_BANK, 1, 0, 0, 0, 1),
                              PIMCmd(PIMCmdType::NOP, 7)};
      pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    }
    else
    {
      throw invalid_argument("Not supported activation");
    }
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }
};

class GemvPIMKernel : public IPIMCmd
{
public:
  GemvPIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd>
  generateKernel(int num_jump_to_be_taken, int num_jump_to_be_taken_odd_bank,
                 int num_jump_to_be_taken_even_bank) override
  {
    vector<PIMCmd> pim_cmds;
    if (kernelType == KernelType::GEMV)
    {
      vector<PIMCmd> tmp_cmds{
          PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                 PIMOpdType::EVEN_BANK, 1, 0, 0, 0),
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken_even_bank, 2),
          PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                 PIMOpdType::ODD_BANK, 1, 0, 0, 0),
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken_odd_bank, 2),
          PIMCmd(PIMCmdType::NOP, 7),
      };
      pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    }
    else if (kernelType == KernelType::GEMVTREE)
    {
      vector<PIMCmd> tmp_cmds{
          PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                 PIMOpdType::EVEN_BANK, 1, 0, 0, 0),
          // FIXME: hard coding
          PIMCmd(PIMCmdType::JUMP, 7, 2), PIMCmd(PIMCmdType::NOP, 7),
          PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_B, PIMOpdType::GRF_B,
                 PIMOpdType::EVEN_BANK, 1),
          PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                 PIMOpdType::ODD_BANK, 1, 0, 0, 0),
          PIMCmd(PIMCmdType::JUMP, 7, 2), PIMCmd(PIMCmdType::NOP, 7),
          PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_B, PIMOpdType::GRF_B,
                 PIMOpdType::EVEN_BANK, 1),
          // PIMCmd(PIMCmdType::JUMP, num_jump, 7), /*it used that tile is 2*/
      };
      pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    }
    else
    {
      throw invalid_argument("Not supported gemv operation");
    }
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }
};

// H: PTMul
class PTMulPIMKernel : public IPIMCmd
{
public:
  PTMulPIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd> generateKernel(int num_jump_to_be_taken,
                                        int num_jump_to_be_taken_odd_bank = 0,
                                        int num_jump_to_be_taken_even_bank = 0) override
  {
    vector<PIMCmd> pim_cmds;

    // Load mod
    pim_cmds.push_back(
      PIMCmd(PIMCmdType::FILL, PIMOpdType::SRF_A, PIMOpdType::EVEN_BANK));

    vector<PIMCmd> tmp_cmds{
        // Load 4x 64-bit coefficients from B into GRF_A
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        // GRF_A = GRF_A * A[0] (in even bank)
        PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        // Load 2
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        // PIMCmd(PIMCmdType::JUMP, 64, 4)

    };

    pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }

private:
  PIMCmdType getPIMCmdType()
  {
    if (kernelType == KernelType::ADD)
      return PIMCmdType::ADD;
    else if (kernelType == KernelType::MUL)
      return PIMCmdType::MUL;
    else
      throw invalid_argument("Not supported element-wise operation");
  }
};

// H: HEAdd
class HEAddPIMKernel : public IPIMCmd
{
public:
  HEAddPIMKernel(KernelType ktype) : IPIMCmd(ktype) {}
  virtual vector<PIMCmd> generateKernel(int num_jump_to_be_taken,
                                        int num_jump_to_be_taken_odd_bank = 0,
                                        int num_jump_to_be_taken_even_bank = 0) override
  {
    vector<PIMCmd> pim_cmds;
    // PIMCmdType pimType = getPIMCmdType();

    // Load mod
    pim_cmds.push_back(
        PIMCmd(PIMCmdType::FILL, PIMOpdType::SRF_A, PIMOpdType::EVEN_BANK));

    vector<PIMCmd> tmp_cmds{
        // Load 4x 64-bit coefficients from A into GRF_A
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        // Add
        PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        // TODO: Do I need to store?
        // Add 4x 64-bit coefficients from B into GRF_A
        // Load 4x 64-bit coefficients from A into GRF_A
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        // Add
        PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),

    };

    pim_cmds.assign(tmp_cmds.begin(), tmp_cmds.end());
    if (num_jump_to_be_taken != 0)
    {
      pim_cmds.push_back(
          PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken, pim_cmds.size() + 1));
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));
    return pim_cmds;
  }

private:
  PIMCmdType getPIMCmdType()
  {
    if (kernelType == KernelType::ADD)
      return PIMCmdType::ADD;
    else if (kernelType == KernelType::MUL)
      return PIMCmdType::MUL;
    else
      throw invalid_argument("Not supported element-wise operation");
  }
};

class PIMCmdGen
{
public:
  static vector<PIMCmd> getPIMCmds(KernelType ktype, int num_jump_to_be_taken,
                                   int num_jump_to_be_taken_odd_bank,
                                   int num_jump_to_be_taken_even_bank);
};
#endif // __PIM_KERNEL_GEN_H__
