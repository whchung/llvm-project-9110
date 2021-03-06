//===--- AMDGPUDSReadClusting.cpp - AMDGPU DSRead Clustering  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to cluster LDS reads.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUDSReadClustering.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "amdgpu-dsread-clustering"

using namespace llvm;

namespace {

class DSReadClustering : public BaseMemOpClusterMutation {
public:
  DSReadClustering(const TargetInstrInfo *tii, const TargetRegisterInfo *tri)
      : BaseMemOpClusterMutation(tii, tri, true) {}

  void collectMemOpRecords(std::vector<SUnit> &SUnits,
                           SmallVectorImpl<MemOpInfo> &MemOpRecords, ScheduleDAGInstrs *DAG) override;
};

// Logic mostly copied from BaseMemOpClusterMutation::collectMemOpRecords.
void DSReadClustering::collectMemOpRecords(
    std::vector<SUnit> &SUnits, SmallVectorImpl<MemOpInfo> &MemOpRecords, ScheduleDAGInstrs *DAG) {
  llvm::errs() << "DSReadClustering::collectMemOpRecords IN\n";
  for (auto &SU : SUnits) {
    if (SU.getInstr()->mayLoad() || SU.getInstr()->mayStore()) {
      llvm::errs() << "Found one mem op:\n";
      DAG->dumpNodeAll(SU);
      llvm::errs() << "\n";
      if (IsLoad && !SU.getInstr()->mayLoad()) {
        llvm::errs() << "Ignore because it's not a load instruction.\n";
        continue;
      }
      if (!IsLoad && !SU.getInstr()->mayStore()) {
        llvm::errs() << "Ignore because it's not a store instruction.\n";
        continue;
      }
    }

    if ((IsLoad && !SU.getInstr()->mayLoad()) ||
        (!IsLoad && !SU.getInstr()->mayStore()))
      continue;

    // Only cluster LDS instructions.
    const MachineInstr &MI = *SU.getInstr();
    if (!SIInstrInfo::isDS(MI)) {
      llvm::errs() << "Ignore due to not a DS instruction.\n";
      continue;
    }

    SmallVector<const MachineOperand *, 4> BaseOps;
    int64_t Offset;
    bool OffsetIsScalable;
    unsigned Width;
    if (TII->getMemOperandsWithOffsetWidth(MI, BaseOps, Offset,
                                           OffsetIsScalable, Width, TRI)) {
      MemOpRecords.push_back(MemOpInfo(&SU, BaseOps, Offset, Width));

      errs() << "Num BaseOps: " << BaseOps.size() << ", Offset: "
                        << Offset << ", OffsetIsScalable: " << OffsetIsScalable
                        << ", Width: " << Width << "\n";
    }
#ifndef NDEBUG
    for (auto *Op : BaseOps)
      assert(Op);
#endif
  }
  llvm::errs() << "DSReadClustering::collectMemOpRecords OUT\n";
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUDSReadClusterDAGMutation(const TargetInstrInfo *TII,
                                     const TargetRegisterInfo *TRI) {
  return std::make_unique<DSReadClustering>(TII, TRI);
}

} // end namespace llvm
