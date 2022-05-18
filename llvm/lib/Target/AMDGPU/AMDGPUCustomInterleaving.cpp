//===--- AMDGPUCustomInterleaving.cpp - AMDGPU Custom Interleaving  -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation for interleaving inside
///       a GEMM hot loop.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCustomInterleaving.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class CustomInterleaving : public ScheduleDAGMutation {
public:
  CustomInterleaving() {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

// Try recognize a GEMM hot loop.
// The 0th SUnit would be an inline asm.
// The last SUnit would be an S_CBRANCH_SCC1.
bool identifyGEMMHotLoop(ScheduleDAGInstrs *DAG) {
  bool gotBegin = false;
  bool gotEnd = false;

  const SUnit &SU = DAG->SUnits[0];
  if (SU.isInstr()) {
    const MachineInstr *MI = SU.getInstr();
    if (MI->isInlineAsm()) {
      gotBegin = true;
    }
  }

  if (gotBegin) {
    if (DAG->ExitSU.getInstr() != nullptr) {
      const MachineInstr *MI = DAG->ExitSU.getInstr();
      if (MI->getOpcode() == AMDGPU::S_CBRANCH_SCC1) {
        gotEnd = true;
      }
    }
  }

  return (gotBegin && gotEnd);
}

void CustomInterleaving::apply(ScheduleDAGInstrs *DAG) {
  if (!identifyGEMMHotLoop(DAG))
    return;

  llvm::errs() << "Inside a GEMM hot loop DAG.\n";

  llvm::errs() << "Before adding cluster edges.\n";
  for (SUnit &SU : DAG->SUnits) {
    DAG->dumpNodeAll(SU);
    llvm::errs() << "==========\n";
  }

  llvm::errs() << "Add some cluster edges.\n";

  // interleave MFMA with ds_loads.
  DAG->addEdge(&DAG->SUnits[28], SDep(&DAG->SUnits[26], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[29], SDep(&DAG->SUnits[35], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[30], SDep(&DAG->SUnits[36], SDep::Artificial));

  // interleave MFMA with ds_writes.
  DAG->addEdge(&DAG->SUnits[31], SDep(&DAG->SUnits[46], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[32], SDep(&DAG->SUnits[47], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[33], SDep(&DAG->SUnits[48], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[34], SDep(&DAG->SUnits[49], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[37], SDep(&DAG->SUnits[51], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[38], SDep(&DAG->SUnits[53], SDep::Artificial));

  // interleave MFMA with buffer_loads.
  DAG->addEdge(&DAG->SUnits[39], SDep(&DAG->SUnits[55], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[40], SDep(&DAG->SUnits[57], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[41], SDep(&DAG->SUnits[58], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[42], SDep(&DAG->SUnits[59], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[43], SDep(&DAG->SUnits[61], SDep::Artificial));

  DAG->addEdge(&DAG->SUnits[44], SDep(&DAG->SUnits[63], SDep::Artificial));

  llvm::errs() << "After adding cluster edges.\n";
  for (SUnit &SU : DAG->SUnits) {
    DAG->dumpNodeAll(SU);
    llvm::errs() << "==========\n";
  }
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUCustomInterleavingDAGMutation() {
  return std::make_unique<CustomInterleaving>();
}

} // end namespace llvm
