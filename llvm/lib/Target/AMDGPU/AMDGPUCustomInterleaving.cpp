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

  //llvm::errs() << "Before adding cluster edges.\n";
  //for (SUnit &SU : DAG->SUnits) {
  //  DAG->dumpNodeAll(SU);
  //  llvm::errs() << "==========\n";
  //}

  //llvm::errs() << "Add some cluster edges.\n";
  //DAG->addEdge(&DAG->SUnits[5], SDep(&DAG->SUnits[3], SDep::Cluster));
  //DAG->addEdge(&DAG->SUnits[5], SDep(&DAG->SUnits[3], SDep::Artificial));
  //DAG->addEdge(&DAG->SUnits[6], SDep(&DAG->SUnits[3], SDep::Cluster));
  //DAG->addEdge(&DAG->SUnits[6], SDep(&DAG->SUnits[3], SDep::Artificial));

  //llvm::errs() << "After adding cluster edges.\n";
  //for (SUnit &SU : DAG->SUnits) {
  //  DAG->dumpNodeAll(SU);
  //  llvm::errs() << "==========\n";
  //}
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUCustomInterleavingDAGMutation() {
  return std::make_unique<CustomInterleaving>();
}

} // end namespace llvm
