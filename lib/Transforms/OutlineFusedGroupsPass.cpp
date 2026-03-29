//===- OutlineFusedGroupsPass.cpp - Outline fusible subgraphs ---*- C++ -*-===//
//
// Identifies connected subgraphs of fusible linalg ops using the cost model,
// outlines each subgraph into its own func.func, and replaces the original
// ops with func.call.  Each outlined function represents a "kernel" that the
// NPU executes with its three engines (DMA, matrix, DSP) pipelined.
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace npu {
#define GEN_PASS_DEF_NPUOUTLINEFUSEDGROUPS
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

//===----------------------------------------------------------------------===//
// Union-Find for grouping ops
//===----------------------------------------------------------------------===//

class UnionFind {
public:
  void makeSet(Operation *op) {
    if (parent_.count(op))
      return;
    parent_[op] = op;
    rank_[op] = 0;
  }

  Operation *find(Operation *op) {
    if (parent_[op] != op)
      parent_[op] = find(parent_[op]);
    return parent_[op];
  }

  void unite(Operation *a, Operation *b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;
    if (rank_[a] < rank_[b])
      std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b])
      rank_[a]++;
  }

private:
  llvm::DenseMap<Operation *, Operation *> parent_;
  llvm::DenseMap<Operation *, int> rank_;
};

//===----------------------------------------------------------------------===//
// Helper: get a short name for a linalg op
//===----------------------------------------------------------------------===//

static std::string getOpShortName(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return "matmul";
  if (isa<linalg::Conv2DNchwFchwOp>(op))
    return "conv2d";
  if (isa<linalg::FillOp>(op))
    return "fill";
  if (isa<linalg::GenericOp>(op))
    return "generic";
  // Fallback: use the op name, replacing dots with underscores.
  std::string name = op->getName().getStringRef().str();
  std::replace(name.begin(), name.end(), '.', '_');
  return name;
}

//===----------------------------------------------------------------------===//
// Helper: topological sort of ops within a group
//===----------------------------------------------------------------------===//

static SmallVector<Operation *> topoSort(ArrayRef<Operation *> ops) {
  // Use the original ordering in the block: ops earlier in the block come first.
  SmallVector<Operation *> sorted(ops.begin(), ops.end());
  llvm::sort(sorted, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return sorted;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct NPUOutlineFusedGroupsPass
    : public npu::impl::NPUOutlineFusedGroupsBase<NPUOutlineFusedGroupsPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Build cost model from module-level target attribute.
    CostModelAnalysis analysis(moduleOp);
    const CostModel &costModel = analysis.getCostModel();

    // Collect all func.func ops to process (avoid modifying while iterating).
    SmallVector<func::FuncOp> funcOps;
    moduleOp.walk([&](func::FuncOp fn) {
      // Skip outlined functions (already marked).
      if (!fn->hasAttr("npu.fused_kernel"))
        funcOps.push_back(fn);
    });

    for (auto funcOp : funcOps)
      processFunction(funcOp, costModel, moduleOp);
  }

  void processFunction(func::FuncOp funcOp, const CostModel &costModel,
                        ModuleOp moduleOp) {
    // Phase 1: Collect all linalg ops in the function.
    SmallVector<Operation *> linalgOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      linalgOps.push_back(op.getOperation());
    });

    if (linalgOps.empty())
      return;

    // Phase 2: Build union-find groups based on producer->consumer edges.
    UnionFind uf;
    llvm::DenseSet<Operation *> linalgSet(linalgOps.begin(), linalgOps.end());
    for (Operation *op : linalgOps)
      uf.makeSet(op);

    for (Operation *consumer : linalgOps) {
      for (Value operand : consumer->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || !linalgSet.contains(producer))
          continue;
        // Only merge if the producer has a single consumer (single use
        // for the connecting value).
        if (!producer->hasOneUse())
          continue;
        // Check cost model: is fusing this pair profitable?
        SmallVector<Operation *> candidateGroup = {producer, consumer};
        FusionDecision decision =
            costModel.isSubgraphFusionProfitable(candidateGroup);
        if (decision.shouldFuse)
          uf.unite(producer, consumer);
      }
    }

    // Phase 3: Collect groups.
    llvm::DenseMap<Operation *, SmallVector<Operation *>> groups;
    for (Operation *op : linalgOps) {
      Operation *root = uf.find(op);
      groups[root].push_back(op);
    }

    // Phase 4: For each group with >1 op, outline into a new func.func.
    int outlinedCount = 0;
    for (auto &[root, groupOps] : groups) {
      if (groupOps.size() <= 1)
        continue;

      // Topologically sort ops within the group.
      SmallVector<Operation *> sorted = topoSort(groupOps);

      outlineGroup(sorted, costModel, moduleOp, funcOp, outlinedCount);
      outlinedCount++;
    }
  }

  void outlineGroup(ArrayRef<Operation *> ops, const CostModel &costModel,
                     ModuleOp moduleOp, func::FuncOp parentFunc,
                     int groupIndex) {
    if (ops.empty())
      return;

    // Build the set of values produced by ops in this group.
    llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
    llvm::DenseSet<Value> producedValues;
    for (Operation *op : ops)
      for (Value result : op->getResults())
        producedValues.insert(result);

    // Identify external inputs: operands not produced within the group.
    SmallVector<Value> externalInputs;
    llvm::DenseSet<Value> seenInputs;
    for (Operation *op : ops) {
      for (Value operand : op->getOperands()) {
        if (!producedValues.contains(operand) &&
            !seenInputs.contains(operand)) {
          externalInputs.push_back(operand);
          seenInputs.insert(operand);
        }
      }
    }

    // Identify external outputs: produced values used outside the group,
    // or results of the last op.
    SmallVector<Value> externalOutputs;
    llvm::DenseSet<Value> seenOutputs;
    for (Operation *op : ops) {
      for (Value result : op->getResults()) {
        bool usedOutside = false;
        for (Operation *user : result.getUsers()) {
          if (!opSet.contains(user)) {
            usedOutside = true;
            break;
          }
        }
        if (usedOutside && !seenOutputs.contains(result)) {
          externalOutputs.push_back(result);
          seenOutputs.insert(result);
        }
      }
    }
    // Also ensure the last op's results are in the output set.
    Operation *lastOp = ops.back();
    for (Value result : lastOp->getResults()) {
      if (!seenOutputs.contains(result)) {
        externalOutputs.push_back(result);
        seenOutputs.insert(result);
      }
    }

    // Build the function name.
    std::string funcName = "fused";
    for (Operation *op : ops)
      funcName += "_" + getOpShortName(op);

    // Build function type.
    SmallVector<Type> inputTypes;
    for (Value v : externalInputs)
      inputTypes.push_back(v.getType());
    SmallVector<Type> outputTypes;
    for (Value v : externalOutputs)
      outputTypes.push_back(v.getType());

    auto funcType =
        FunctionType::get(moduleOp.getContext(), inputTypes, outputTypes);

    // Create the outlined function at module level, before the parent function.
    OpBuilder moduleBuilder(moduleOp.getContext());
    moduleBuilder.setInsertionPoint(parentFunc);
    auto outlinedFunc = func::FuncOp::create(
        moduleBuilder, ops.front()->getLoc(), funcName, funcType);
    outlinedFunc.setVisibility(SymbolTable::Visibility::Private);
    outlinedFunc->setAttr("npu.fused_kernel",
                          moduleBuilder.getUnitAttr());

    // Create the entry block with arguments matching external inputs.
    Block *entryBlock = outlinedFunc.addEntryBlock();
    OpBuilder bodyBuilder(entryBlock, entryBlock->end());

    // Build value mapping: original values -> new values in outlined func.
    IRMapping mapping;
    for (auto [origVal, blockArg] :
         llvm::zip(externalInputs, entryBlock->getArguments())) {
      mapping.map(origVal, blockArg);
    }

    // Clone ops into the outlined function.
    for (Operation *op : ops)
      bodyBuilder.clone(*op, mapping);

    // Build return values.
    SmallVector<Value> returnValues;
    for (Value origOutput : externalOutputs)
      returnValues.push_back(mapping.lookup(origOutput));
    func::ReturnOp::create(bodyBuilder, lastOp->getLoc(), returnValues);

    // Replace original ops with a func.call in the parent function.
    OpBuilder callBuilder(moduleOp.getContext());
    callBuilder.setInsertionPointAfter(lastOp);
    auto callOp = func::CallOp::create(
        callBuilder, lastOp->getLoc(), outlinedFunc, externalInputs);

    // Replace uses of external outputs with call results.
    for (auto [origOutput, callResult] :
         llvm::zip(externalOutputs, callOp.getResults())) {
      origOutput.replaceAllUsesExcept(callResult, callOp);
    }

    // Erase the original ops in reverse topological order.
    for (Operation *op : llvm::reverse(ops))
      op->erase();
  }
};

} // namespace
