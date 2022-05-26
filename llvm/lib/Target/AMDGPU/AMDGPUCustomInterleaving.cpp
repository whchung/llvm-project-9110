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

static bool isDSRead(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isDS(*MI) && (MI->mayLoad()));
}

static bool isDSWrite(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isDS(*MI) && (MI->mayStore()));
}

static bool isMFMA(const SUnit &SU) {
  return SIInstrInfo::isMAI(*SU.getInstr());
}

static bool isVMEMLoad(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isVMEM(*MI) && (MI->mayLoad()));
}

static bool isVMEMStore(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return (SIInstrInfo::isVMEM(*MI) && (MI->mayStore()));
}

static bool isInlineAsm(const SUnit &SU) {
  MachineInstr *MI = SU.getInstr();
  return MI->isInlineAsm();
}

// Try recognize a GEMM hot loop.
// The 0th SUnit would be:
// - CK: an inline asm.
// - MS benchmark: a VMEM load.
// The last SUnit would be an S_CBRANCH_SCC1.
bool identifyGEMMHotLoop(ScheduleDAGInstrs *DAG) {
  bool gotBegin = false;
  bool gotEnd = false;

  const SUnit &SU = DAG->SUnits[0];
  if (SU.isInstr()) {
    if (isInlineAsm(SU) || isVMEMLoad(SU)) {
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
#if 1
  llvm::errs() << "Try identify a GEMM hot loop DAG.\n";
#endif

  if (!identifyGEMMHotLoop(DAG))
    return;

#if 1
  llvm::errs() << "Inside a GEMM hot loop DAG.\n";
#endif

  int64_t DSReadCount = 0;
  int64_t DSWriteCount = 0;
  int64_t VMEMLoadCount = 0;
  int64_t VMEMStoreCount = 0;
  int64_t MFMACount = 0;

  SmallVector<SUnit *, 8> DSReads;
  SmallVector<SUnit *, 8> DSWrites;
  SmallVector<SUnit *, 8> VMEMLoads;
  SmallVector<SUnit *, 8> VMEMStores;
  SmallVector<SUnit *, 32> MFMAs;

#if 0
  llvm::errs() << "Before adding artificial edges.\n";
#endif
  for (SUnit &SU : DAG->SUnits) {
#if 0
    DAG->dumpNodeAll(SU);
    llvm::errs() << "==========\n";
#endif

    if (isDSRead(SU)) {
      DSReadCount++;
      DSReads.push_back(&SU);
    } else if (isDSWrite(SU)) {
      DSWriteCount++;
      DSWrites.push_back(&SU);
    } else if (isMFMA(SU)) {
      MFMACount++;
      MFMAs.push_back(&SU);
    } else if (isVMEMLoad(SU)) {
      VMEMLoadCount++;
      VMEMLoads.push_back(&SU);
    } else if (isVMEMStore(SU)) {
      VMEMStoreCount++;
      VMEMStores.push_back(&SU);
    }
  }

#if 1
  llvm::errs() << "DSRead instruction count: " << DSReadCount << "\n";
  llvm::errs() << "DSWrite instruction count: " << DSWriteCount << "\n";
  llvm::errs() << "VMEMLoad instruction count: " << VMEMLoadCount << "\n";
  llvm::errs() << "VMEMStore instruction count: " << VMEMStoreCount << "\n";
  llvm::errs() << "MFMA instruction count: " << MFMACount << "\n";
#endif

  assert(VMEMStoreCount == 0);

  // Determine the order of interleaving.
  int64_t DSReadPriority, DSWritePriority, VMEMLoadPriority;
  DSReadPriority = DSWritePriority = VMEMLoadPriority = -1;
  auto NotAssignedPriority = [](int64_t prio) { return prio < 0; };

  int64_t CurrentPriority, TotalPriority;
  CurrentPriority = TotalPriority = 0;

  // Starting backward.
  int64_t SUIter = DAG->SUnits.size() - 1;
  while (SUIter >= 0) {
    SUnit &SU = DAG->SUnits[SUIter--];
    if (isDSRead(SU) && NotAssignedPriority(DSReadPriority)) {
      DSReadPriority = CurrentPriority++;
    } else if (isDSWrite(SU) && NotAssignedPriority(DSWritePriority)) {
      DSWritePriority = CurrentPriority++;
    } else if (isVMEMLoad(SU) && NotAssignedPriority(VMEMLoadPriority)) {
      VMEMLoadPriority = CurrentPriority++;
    }
  }
  TotalPriority = CurrentPriority;

#if 1
  llvm::errs() << "DSReadPriority: " << DSReadPriority << "\n";
  llvm::errs() << "DSWritePriority: " << DSWritePriority << "\n";
  llvm::errs() << "VMEMLoadPriority: " << VMEMLoadPriority << "\n";
#endif

#if 0
  llvm::errs() << "Add some artificial edges.\n";
#endif

  int64_t MFMAIter = MFMAs.size() - 1;

  // Reset CurrentPriority.
  CurrentPriority = 0;

  // Iterate through all different instruction groups to be interleaved with
  // MFMA.
  while (CurrentPriority < TotalPriority) {
    if (CurrentPriority == VMEMLoadPriority) {
      // Interleave MFMA with buffer_loads.
      int64_t VMEMLoadIter = VMEMLoads.size() - 1;
      while ((VMEMLoadIter >= 0) && (MFMAIter >= 0)) {
        SUnit *VMEMLoadSU = VMEMLoads[VMEMLoadIter--];
        SUnit *MFMASU = MFMAs[MFMAIter--];
        DAG->addEdge(MFMASU, SDep(VMEMLoadSU, SDep::Artificial));
      }
    } else if (CurrentPriority == DSWritePriority) {
      // Interleave MFMA with ds_writes.
      int64_t DSWriteIter = DSWrites.size() - 1;
      while ((DSWriteIter >= 0) && (MFMAIter >= 0)) {
        SUnit *DSWriteSU = DSWrites[DSWriteIter--];
        SUnit *MFMASU = MFMAs[MFMAIter--];
        DAG->addEdge(MFMASU, SDep(DSWriteSU, SDep::Artificial));
      }
    } else if (CurrentPriority == DSReadPriority) {
      // Interleave MFMA with ds_reads.
      int64_t DSReadIter = DSReads.size() - 1;
      while ((DSReadIter >= 0) && (MFMAIter >= 0)) {
        SUnit *DSReadSU = DSReads[DSReadIter--];
        SUnit *MFMASU = MFMAs[MFMAIter--];
        DAG->addEdge(MFMASU, SDep(DSReadSU, SDep::Artificial));
      }
    }

    // Move to the next instruction groups.
    ++CurrentPriority;
  }

#if 0
  llvm::errs() << "After adding artificial edges.\n";
  for (SUnit &SU : DAG->SUnits) {
    DAG->dumpNodeAll(SU);
    llvm::errs() << "==========\n";
  }
#endif
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUCustomInterleavingDAGMutation() {
  return std::make_unique<CustomInterleaving>();
}

} // end namespace llvm
