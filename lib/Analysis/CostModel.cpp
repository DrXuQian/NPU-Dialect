//===- CostModel.cpp - Unified NPU cost model -----------------*- C++ -*-===//
//
// Single analytical cost model shared by all passes.
// Passes enumerate candidates; this model evaluates them.
//
//===----------------------------------------------------------------------===//

#include "npu/Analysis/CostModel.h"
#include "npu/Analysis/OperatorTilingSpec.h"
#include "npu/Dialect/NPU/NPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace mlir;
using namespace npu;

static constexpr int64_t kInfCycles = std::numeric_limits<int64_t>::max() / 2;

//===----------------------------------------------------------------------===//
// Level 0 — hardware primitive queries
//===----------------------------------------------------------------------===//

/// Safe product of shape dimensions, treating dynamic dims as 1.
static int64_t safeShapeProduct(ShapedType type) {
  if (!type.hasStaticShape()) {
    int64_t prod = 1;
    for (int64_t d : type.getShape())
      prod *= (d >= 0 ? d : 1); // kDynamic is negative
    return prod;
  }
  int64_t prod = 1;
  for (int64_t d : type.getShape())
    prod *= d;
  return prod;
}

static int64_t safeDim(ShapedType type, unsigned dim) {
  int64_t d = type.getDimSize(dim);
  return d >= 0 ? d : 1; // kDynamic → 1
}

int64_t CostModel::opMACs(Operation *op) const {
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
    auto outType = cast<ShapedType>(matmul.getResult(0).getType());
    auto lhsType = cast<ShapedType>(matmul.getInputs()[0].getType());
    int64_t M = safeDim(outType, 0);
    int64_t N = safeDim(outType, 1);
    int64_t K = safeDim(lhsType, 1);
    return M * N * K;
  }

  if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
    auto outType = cast<ShapedType>(conv.getResult(0).getType());
    auto filterType = cast<ShapedType>(conv.getInputs()[1].getType());
    int64_t N = safeDim(outType, 0);
    int64_t Co = safeDim(outType, 1);
    int64_t Ho = safeDim(outType, 2);
    int64_t Wo = safeDim(outType, 3);
    int64_t Ci = safeDim(filterType, 1);
    int64_t Kh = safeDim(filterType, 2);
    int64_t Kw = safeDim(filterType, 3);
    return N * Co * Ho * Wo * Ci * Kh * Kw;
  }

  // linalg.generic / elementwise: one op per output element
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (generic.getNumResults() > 0) {
      auto outType = cast<ShapedType>(generic.getResult(0).getType());
      int64_t elems = safeShapeProduct(outType);
      return elems;
    }
  }

  return 1024; // conservative default
}

int64_t CostModel::computeCycles(Operation *op) const {
  int64_t macs = opMACs(op);

  // Determine unit: matmul/conv → matrix, everything else → dsp
  bool isMatrix = isa<linalg::MatmulOp>(op) ||
                  isa<linalg::Conv2DNchwFchwOp>(op);
  int64_t throughput = isMatrix ? target_.matrixThroughput
                                : target_.dspThroughput;
  return std::max<int64_t>(1, macs / throughput);
}

int64_t CostModel::dmaCycles(int64_t bytes) const {
  return std::max<int64_t>(1, static_cast<int64_t>(bytes / target_.bytesPerCycle()));
}

int64_t CostModel::tensorBytes(ShapedType type) {
  int64_t elems = 1;
  for (int64_t d : type.getShape())
    elems *= (d >= 0 ? d : 1); // kDynamic → 1
  int64_t bitWidth = type.getElementTypeBitWidth();
  return elems * std::max<int64_t>(1, bitWidth / 8);
}

//===----------------------------------------------------------------------===//
// Roofline model
//===----------------------------------------------------------------------===//

RooflineCost CostModel::evaluateRooflineRaw(
    double flops, double memBytes, bool useMatrixUnit) const {
  RooflineCost cost;
  cost.flops = flops;
  cost.memBytes = memBytes;

  // Peak hardware capabilities
  double peakFLOPS = useMatrixUnit
      ? static_cast<double>(target_.matrixThroughput) * target_.clockGHz * 1e9
      : static_cast<double>(target_.dspThroughput) * target_.clockGHz * 1e9;
  double peakBW = static_cast<double>(target_.dramBandwidth) * 1e9; // bytes/sec

  // Arithmetic intensity = FLOPs / Bytes
  cost.arithmeticIntensity = (memBytes > 0) ? flops / memBytes : 1e12;

  // Roofline ridge point
  double ridgePoint = peakFLOPS / peakBW;

  // Attainable performance
  cost.peakAttainable = std::min(peakFLOPS,
                                  cost.arithmeticIntensity * peakBW);

  // Time = max(compute_time, memory_time)
  double computeTime = (peakFLOPS > 0) ? flops / peakFLOPS : 0;
  double memoryTime = (peakBW > 0) ? memBytes / peakBW : 0;
  cost.timeSec = std::max(computeTime, memoryTime);
  cost.isMemoryBound = (cost.arithmeticIntensity < ridgePoint);
  cost.efficiency = (peakFLOPS > 0 && cost.timeSec > 0)
      ? (flops / cost.timeSec) / peakFLOPS : 0;

  return cost;
}

RooflineCost CostModel::evaluateRoofline(Operation *op) const {
  double flops = static_cast<double>(opMACs(op)) * 2.0; // MAC = 2 FLOPs

  // Memory: all input + output bytes (each goes to/from DRAM once)
  double memBytes = 0;
  for (auto operand : op->getOperands()) {
    if (auto type = dyn_cast<ShapedType>(operand.getType()))
      memBytes += tensorBytes(type);
  }
  for (auto result : op->getResults()) {
    if (auto type = dyn_cast<ShapedType>(result.getType()))
      memBytes += tensorBytes(type);
  }

  bool isMatrix = isa<linalg::MatmulOp>(op) ||
                  isa<linalg::Conv2DNchwFchwOp>(op);
  return evaluateRooflineRaw(flops, memBytes, isMatrix);
}

RooflineCost CostModel::evaluateRoofline(
    llvm::ArrayRef<Operation *> ops) const {
  if (ops.empty())
    return {};

  // Aggregate FLOPs
  double totalFlops = 0;
  for (auto *op : ops)
    totalFlops += static_cast<double>(opMACs(op)) * 2.0;

  // For memory: only count EXTERNAL data (not intermediates).
  // Intermediates stay in SRAM → no DRAM access.
  llvm::DenseSet<Value> producedValues;
  llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
  for (auto *op : ops)
    for (auto result : op->getResults())
      producedValues.insert(result);

  double externalMemBytes = 0;

  // External inputs
  for (auto *op : ops) {
    for (auto operand : op->getOperands()) {
      if (!producedValues.contains(operand)) {
        if (auto type = dyn_cast<ShapedType>(operand.getType()))
          externalMemBytes += tensorBytes(type);
      }
    }
  }
  // External outputs (used outside subgraph or last op's results)
  for (auto *op : ops) {
    for (auto result : op->getResults()) {
      bool usedOutside = false;
      for (auto *user : result.getUsers()) {
        if (!opSet.contains(user)) {
          usedOutside = true;
          break;
        }
      }
      if (usedOutside || result.getUses().empty()) {
        if (auto type = dyn_cast<ShapedType>(result.getType()))
          externalMemBytes += tensorBytes(type);
      }
    }
  }

  // Determine dominant unit
  bool hasMatrix = false;
  for (auto *op : ops) {
    if (isa<linalg::MatmulOp>(op) || isa<linalg::Conv2DNchwFchwOp>(op))
      hasMatrix = true;
  }

  return evaluateRooflineRaw(totalFlops, externalMemBytes, hasMatrix);
}

//===----------------------------------------------------------------------===//
// Level 1 — tile configuration evaluation
//===----------------------------------------------------------------------===//

ScheduleCost CostModel::evaluateMatmulTile(
    int64_t tm, int64_t tn, int64_t tk,
    int64_t M, int64_t N, int64_t K,
    int64_t dtypeBytes, int64_t extraIntermediateBytes) const {

  int64_t aBytes = tm * tk * dtypeBytes;
  int64_t bBytes = tk * tn * dtypeBytes;
  int64_t cBytes = tm * tn * dtypeBytes;

  // Double-buffer inputs, single-buffer output + intermediates
  int64_t sramPeak = aBytes * 2 + bBytes * 2 + cBytes + extraIntermediateBytes;

  if (sramPeak > target_.sramPerCore) {
    return {0, 0, kInfCycles, sramPeak, 0, 0};
  }

  int64_t numM = (M + tm - 1) / tm;
  int64_t numN = (N + tn - 1) / tn;
  int64_t numK = (K + tk - 1) / tk;
  int64_t numTiles = numM * numN * numK;

  int64_t tileCompute = std::max<int64_t>(1, (tm * tn * tk) / target_.matrixThroughput);
  int64_t tileDmaIn = dmaCycles(aBytes + bBytes);
  int64_t tileDmaOut = dmaCycles(cBytes);
  int64_t tileDmaBytes = aBytes + bBytes + cBytes;

  int64_t total;
  if (numTiles <= 1) {
    total = tileDmaIn + tileCompute + tileDmaOut;
  } else {
    int64_t steady = std::max(tileDmaIn, tileCompute);
    total = tileDmaIn + (numTiles - 1) * steady + tileCompute + tileDmaOut;
  }

  return {
    numTiles * tileCompute,
    numTiles * (tileDmaIn + tileDmaOut),
    total,
    sramPeak,
    numTiles,
    numTiles * tileDmaBytes,
  };
}

ScheduleCost CostModel::evaluateConv2dTile(
    int64_t tn, int64_t tco, int64_t th, int64_t tw,
    int64_t N, int64_t Co, int64_t Ho, int64_t Wo,
    int64_t Ci, int64_t Kh, int64_t Kw,
    llvm::ArrayRef<int64_t> stride,
    int64_t dtypeBytes, int64_t extraOutputShaped) const {

  int64_t sH = stride.size() > 0 ? stride[0] : 1;
  int64_t sW = stride.size() > 1 ? stride[1] : 1;
  int64_t inH = std::min((th - 1) * sH + Kh, Ho * sH);
  int64_t inW = std::min((tw - 1) * sW + Kw, Wo * sW);

  int64_t inputBytes = tn * Ci * inH * inW * dtypeBytes;
  int64_t weightBytes = tco * Ci * Kh * Kw * dtypeBytes;
  int64_t outputBytes = tn * tco * th * tw * dtypeBytes;
  int64_t extraBytes = extraOutputShaped * outputBytes;

  int64_t sramPeak = inputBytes * 2 + weightBytes + outputBytes + extraBytes;

  if (sramPeak > target_.sramPerCore) {
    return {0, 0, kInfCycles, sramPeak, 0, 0};
  }

  int64_t numTiles = ((N + tn - 1) / tn) * ((Co + tco - 1) / tco) *
                     ((Ho + th - 1) / th) * ((Wo + tw - 1) / tw);

  int64_t macsPerTile = tn * tco * th * tw * Ci * Kh * Kw;
  int64_t tileCompute = std::max<int64_t>(1, macsPerTile / target_.matrixThroughput);
  int64_t tileDmaIn = dmaCycles(inputBytes + weightBytes);
  int64_t tileDmaOut = dmaCycles(outputBytes);
  int64_t tileDmaBytes = inputBytes + weightBytes + outputBytes;

  int64_t total;
  if (numTiles <= 1) {
    total = tileDmaIn + tileCompute + tileDmaOut;
  } else {
    int64_t steady = std::max(tileDmaIn, tileCompute);
    total = tileDmaIn + (numTiles - 1) * steady + tileCompute + tileDmaOut;
  }

  return {
    numTiles * tileCompute,
    numTiles * (tileDmaIn + tileDmaOut),
    total,
    sramPeak,
    numTiles,
    numTiles * tileDmaBytes,
  };
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

llvm::SmallVector<int64_t> CostModel::tileCandidates(int64_t dim) {
  llvm::SmallVector<int64_t> sizes;
  int64_t s = dim;
  while (s >= 16) {
    sizes.push_back(s);
    s /= 2;
  }
  if (sizes.empty() || sizes.back() > 16)
    sizes.push_back(std::min<int64_t>(16, dim));
  return sizes;
}

//===----------------------------------------------------------------------===//
// Level 2 — pass decision APIs
//===----------------------------------------------------------------------===//

TileConfig CostModel::bestTileConfig(Operation *op) const {
  TileConfig best;
  best.cost = {0, 0, kInfCycles, 0, 0, 0};

  if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
    auto outType = cast<ShapedType>(matmul.getResult(0).getType());
    auto lhsType = cast<ShapedType>(matmul.getInputs()[0].getType());
    int64_t M = outType.getDimSize(0);
    int64_t N = outType.getDimSize(1);
    int64_t K = lhsType.getDimSize(1);
    int64_t dtypeBytes = std::max<int64_t>(1, outType.getElementTypeBitWidth() / 8);

    for (int64_t tm : tileCandidates(M))
      for (int64_t tn : tileCandidates(N))
        for (int64_t tk : tileCandidates(K)) {
          auto cost = evaluateMatmulTile(tm, tn, tk, M, N, K, dtypeBytes);
          if (cost.totalCycles < best.cost.totalCycles) {
            best.tileSizes = {tm, tn, tk};
            best.cost = cost;
          }
        }
    return best;
  }

  if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
    auto outType = cast<ShapedType>(conv.getResult(0).getType());
    auto filterType = cast<ShapedType>(conv.getInputs()[1].getType());
    auto inputType = cast<ShapedType>(conv.getInputs()[0].getType());
    int64_t N_ = outType.getDimSize(0);
    int64_t Co = outType.getDimSize(1);
    int64_t Ho = outType.getDimSize(2);
    int64_t Wo = outType.getDimSize(3);
    int64_t Ci = inputType.getDimSize(1);
    int64_t Kh = filterType.getDimSize(2);
    int64_t Kw = filterType.getDimSize(3);
    int64_t dtypeBytes = std::max<int64_t>(1, outType.getElementTypeBitWidth() / 8);

    // TODO: extract strides from op attributes
    int64_t stride[] = {1, 1};

    auto tnCands = N_ > 1 ? tileCandidates(N_)
                           : llvm::SmallVector<int64_t>{N_};
    for (int64_t tn : tnCands)
      for (int64_t tco : tileCandidates(Co))
        for (int64_t th : tileCandidates(Ho))
          for (int64_t tw : tileCandidates(Wo)) {
            auto cost = evaluateConv2dTile(tn, tco, th, tw, N_, Co, Ho, Wo,
                                           Ci, Kh, Kw, stride, dtypeBytes);
            if (cost.totalCycles < best.cost.totalCycles) {
              best.tileSizes = {tn, tco, th, tw};
              best.cost = cost;
            }
          }
    return best;
  }

  // Elementwise / generic: tile along first dimension
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (generic.getNumResults() > 0) {
      auto outType = cast<ShapedType>(generic.getResult(0).getType());
      int64_t dim0 = outType.getDimSize(0);
      int64_t dtypeBytes = std::max<int64_t>(1, outType.getElementTypeBitWidth() / 8);
      int64_t elemsPerRow = 1;
      for (unsigned i = 1; i < outType.getRank(); ++i)
        elemsPerRow *= outType.getDimSize(i);
      int64_t bytesPerRow = elemsPerRow * dtypeBytes * 3; // in + out + intermediate

      for (int64_t td : tileCandidates(dim0)) {
        int64_t sram = td * bytesPerRow * 2; // double buffer
        if (sram > target_.sramPerCore)
          continue;
        int64_t numTiles = (dim0 + td - 1) / td;
        int64_t tileCompute = std::max<int64_t>(1, (td * elemsPerRow) / target_.dspThroughput);
        int64_t tileDma = dmaCycles(td * bytesPerRow);
        int64_t total;
        if (numTiles <= 1)
          total = tileDma + tileCompute + tileDma;
        else {
          int64_t steady = std::max(tileDma, tileCompute);
          total = tileDma + (numTiles - 1) * steady + tileCompute + tileDma;
        }
        if (total < best.cost.totalCycles) {
          best.tileSizes = {td};
          best.cost = {numTiles * tileCompute, numTiles * tileDma * 2,
                       total, sram, numTiles, numTiles * td * bytesPerRow * 2};
        }
      }
    }
    return best;
  }

  return best;
}

FusionDecision CostModel::isFusionProfitable(
    Operation *producer, Operation *consumer) const {

  // Hard constraint: never fuse INTO a matrix-unit consumer
  bool consumerIsMatrix = isa<linalg::MatmulOp>(consumer) ||
                          isa<linalg::Conv2DNchwFchwOp>(consumer);
  if (consumerIsMatrix)
    return {false, 0};

  // Find the intermediate tensor (producer output = consumer input)
  Value intermediate;
  for (auto result : producer->getResults()) {
    for (auto &use : result.getUses()) {
      if (use.getOwner() == consumer) {
        intermediate = result;
        break;
      }
    }
    if (intermediate)
      break;
  }

  if (!intermediate)
    return {false, 0};

  auto intermediateType = dyn_cast<ShapedType>(intermediate.getType());
  if (!intermediateType)
    return {false, 0};

  // Benefit: skip DRAM roundtrip of intermediate
  int64_t intermediateBytes = tensorBytes(intermediateType);
  int64_t roundtripDMA = 2 * dmaCycles(intermediateBytes);

  // Cost: compare downstream tiling costs
  auto costUnfusedProducer = bestTileConfig(producer).cost;
  auto costUnfusedConsumer = bestTileConfig(consumer).cost;
  int64_t costUnfused = costUnfusedProducer.totalCycles +
                        costUnfusedConsumer.totalCycles + roundtripDMA;

  // For fused cost, we use the producer's cost as approximation
  // (fused group is dominated by the heavier op, but intermediate
  // stays in SRAM → no extra DMA, just possibly larger working set).
  // A more accurate estimate would build a temporary fused group and
  // run bestTileConfig on it, but this is a good first-order model.
  int64_t costFused = costUnfusedProducer.totalCycles +
                      costUnfusedConsumer.totalCycles;
  // If intermediate doesn't fit in SRAM alongside tile data, penalty
  if (costUnfusedProducer.sramPeakBytes + intermediateBytes > target_.sramPerCore) {
    // Larger working set → need smaller tiles → more iterations
    double overflowRatio =
        static_cast<double>(costUnfusedProducer.sramPeakBytes + intermediateBytes) /
        target_.sramPerCore;
    costFused += static_cast<int64_t>(roundtripDMA * (overflowRatio - 1.0));
  }

  int64_t benefit = costUnfused - costFused;
  return {benefit > 0, benefit};
}

ScheduleCost CostModel::evaluateSpatialSplit(
    Operation *op, unsigned splitDim, int64_t numChunks) const {

  if (op->getNumResults() == 0)
    return {0, 0, kInfCycles, 0, 0, 0};

  auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!outType || !outType.hasStaticShape())
    return {0, 0, kInfCycles, 0, 0, 0};

  if (splitDim >= static_cast<unsigned>(outType.getRank()))
    return {0, 0, kInfCycles, 0, 0, 0};

  int64_t dimSize = outType.getDimSize(splitDim);
  int64_t activeCores = std::min(numChunks, dimSize);
  if (activeCores <= 0)
    return {0, 0, kInfCycles, 0, 0, 0};

  // Per-core share (ceiling)
  int64_t perCoreDim = (dimSize + activeCores - 1) / activeCores;

  // Estimate per-core compute = full compute / activeCores (ceiling)
  int64_t fullCycles = computeCycles(op);
  int64_t perCoreCompute = (fullCycles + activeCores - 1) / activeCores;

  // Estimate per-core DMA: input bytes scale roughly with split dim
  int64_t totalInputBytes = 0;
  for (auto input : op->getOperands()) {
    if (auto type = dyn_cast<ShapedType>(input.getType()))
      totalInputBytes += tensorBytes(type);
  }
  // Rough: DMA per core ≈ total / activeCores for data that splits,
  // plus full weight data for data that doesn't split
  int64_t perCoreDmaBytes = totalInputBytes / activeCores;
  // Weights (second operand of matmul/conv) are shared → full copy per core
  if (op->getNumOperands() >= 2) {
    auto weightType = dyn_cast<ShapedType>(op->getOperand(1).getType());
    if (weightType)
      perCoreDmaBytes += tensorBytes(weightType);
  }
  int64_t perCoreDma = dmaCycles(perCoreDmaBytes);

  // Idle core penalty
  int64_t idlePenalty = 0;
  if (activeCores < numChunks)
    idlePenalty = (numChunks - activeCores) * target_.interCoreSyncCost;

  int64_t total = std::max(perCoreCompute, perCoreDma) +
                  idlePenalty + target_.interCoreSyncCost;

  // Also factor in downstream temporal tiling cost for one core
  auto tileCost = bestTileConfig(op).cost;
  // Scale tile cost by per-core fraction
  int64_t scaledTileCost = tileCost.isValid()
      ? tileCost.totalCycles / std::max<int64_t>(1, activeCores)
      : total;

  total = std::max(total, scaledTileCost);

  return {
    perCoreCompute,
    perCoreDma,
    total,
    tileCost.sramPeakBytes,
    tileCost.numTiles,
    perCoreDmaBytes * activeCores,
  };
}

CostModel::TileAlignmentResult CostModel::checkTileAlignment(
    llvm::ArrayRef<Operation *> ops) const {
  TileAlignmentResult result;
  result.aligned = true;
  result.alignmentOverheadBytes = 0;

  if (ops.size() <= 1)
    return result;

  // Check each adjacent producer->consumer pair.
  for (size_t i = 0; i + 1 < ops.size(); ++i) {
    Operation *producer = ops[i];
    Operation *consumer = ops[i + 1];

    // Find the connecting value: a producer result used by the consumer.
    Value connecting;
    for (Value res : producer->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (user == consumer) {
          connecting = res;
          break;
        }
      }
      if (connecting)
        break;
    }

    // If there's no direct connection, the pair is trivially compatible
    // (they may be connected through an intermediate op not in our list).
    if (!connecting)
      continue;

    // Get tiling specs for both ops.
    const OperatorTilingSpec *producerSpec = getTilingSpec(producer);
    const OperatorTilingSpec *consumerSpec = getTilingSpec(consumer);

    // If either spec is unavailable, assume aligned (be conservative and
    // allow fusion to proceed).
    if (!producerSpec || !consumerSpec)
      continue;

    // Get the type of the connecting tensor.
    auto connectingType = dyn_cast<ShapedType>(connecting.getType());
    if (!connectingType || !connectingType.hasStaticShape())
      continue;

    // Check: do the producer's output shape and the consumer's input shape
    // match in type and rank? The connecting value IS the producer's output
    // and the consumer's input, so check that the consumer uses it as-is.
    // Find which consumer operand this corresponds to.
    ShapedType consumerInputType = connectingType; // same value
    ShapedType producerOutputType = connectingType;

    // If shapes are the same type and rank, they're aligned for this pair.
    // (The connecting value is literally the same SSA value, so shapes match
    // by construction. But the tiling specs may want different tile
    // granularities on different dims.)

    // Check preferred temporal dim compatibility: if both specs tile a
    // dimension of the connecting tensor differently, there's overhead.
    // For example, if the producer tiles output dim 0 and the consumer
    // tiles input dim 0 at different granularities.

    // Compare the rank and shape of the producer output with what the
    // consumer expects. For the connecting tensor, these are identical,
    // but we check if the overall output of consumer differs in rank/shape
    // (e.g., pooling halves spatial dims).
    if (consumer->getNumResults() == 0)
      continue;

    auto consumerOutType =
        dyn_cast<ShapedType>(consumer->getResult(0).getType());
    if (!consumerOutType || !consumerOutType.hasStaticShape())
      continue;

    // If producer output rank != consumer output rank, there's a shape
    // transformation happening → potential alignment issue.
    if (producerOutputType.getRank() != consumerOutType.getRank()) {
      // Rank mismatch: compute overhead as the full connecting tensor size.
      result.aligned = false;
      result.alignmentOverheadBytes += tensorBytes(connectingType);
      continue;
    }

    // Check each dimension: if consumer output is smaller than producer
    // output on any dim, there's a reduction (e.g., pooling).
    for (int64_t d = 0; d < producerOutputType.getRank(); ++d) {
      int64_t prodDim = producerOutputType.getDimSize(d);
      int64_t consDim = consumerOutType.getDimSize(d);
      if (prodDim != consDim && prodDim > 0 && consDim > 0) {
        // Dimension mismatch: the tile boundaries won't align perfectly.
        // Overhead = extra bytes that need to be buffered for the mismatch.
        int64_t elemBytes =
            std::max<int64_t>(1, connectingType.getElementTypeBitWidth() / 8);
        int64_t totalElems = 1;
        for (int64_t dd = 0; dd < connectingType.getRank(); ++dd)
          totalElems *= connectingType.getDimSize(dd);
        // Overhead is proportional to the dimension ratio difference.
        double ratio = static_cast<double>(std::max(prodDim, consDim)) /
                       std::min(prodDim, consDim);
        int64_t overhead =
            static_cast<int64_t>((ratio - 1.0) * totalElems * elemBytes);
        result.alignmentOverheadBytes += overhead;
        result.aligned = false;
      }
    }
  }

  return result;
}

CostModel::SpillStrategy CostModel::evaluateSpillStrategy(
    int64_t tileWorkingSet, int64_t intermediateBytes) const {
  int64_t spillDmaCost = 2 * dmaCycles(intermediateBytes);
  int64_t retileOverhead = dmaCycles(tileWorkingSet); // rough: one extra DMA pass
  return spillDmaCost < retileOverhead ? SpillStrategy::Spill
                                       : SpillStrategy::Retile;
}

CostModel::SpatialSplitCost CostModel::evaluateSpatialSplitDetailed(
    Operation *op, const SplitDimSpec &dimSpec, int64_t numChunks) const {

  SpatialSplitCost result;
  result.needsReduce = !dimSpec.isParallel;

  if (op->getNumResults() == 0)
    return result;

  auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!outType || !outType.hasStaticShape())
    return result;

  // Number of active cores
  int64_t dimSize = 1;
  if (dimSpec.outputDim < static_cast<unsigned>(outType.getRank()))
    dimSize = outType.getDimSize(dimSpec.outputDim);
  int64_t activeCores = std::min(numChunks, dimSize);
  if (activeCores <= 0)
    return result;

  int64_t perCoreDim = (dimSize + activeCores - 1) / activeCores;

  // Per-core compute
  int64_t fullCycles = computeCycles(op);
  result.perCoreComputeCycles = (fullCycles + activeCores - 1) / activeCores;

  // Per-core DMA: precisely account for each input operand
  unsigned numInputs = 0;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    numInputs = linalgOp.getNumDpsInputs();
  else
    numInputs = op->getNumOperands() > 0 ? op->getNumOperands() - 1 : 0;

  for (const auto &rule : dimSpec.inputRules) {
    if (rule.operandIdx >= op->getNumOperands())
      continue;
    auto operandType = dyn_cast<ShapedType>(
        op->getOperand(rule.operandIdx).getType());
    if (!operandType)
      continue;

    int64_t fullBytes = tensorBytes(operandType);

    switch (rule.kind) {
    case SliceKind::Shared:
      // Each core loads the full operand — expensive!
      result.sharedDataBytes += fullBytes;
      result.perCoreDmaInBytes += fullBytes;
      break;

    case SliceKind::Split: {
      // Operand is sliced along rule.sliceDim, size = perCoreDim
      int64_t slicedBytes = fullBytes;
      if (rule.sliceDim < static_cast<unsigned>(operandType.getRank())) {
        int64_t origDim = operandType.getDimSize(rule.sliceDim);
        if (origDim > 0)
          slicedBytes = fullBytes * perCoreDim / origDim;
      }
      result.splitDataBytes += slicedBytes;
      result.perCoreDmaInBytes += slicedBytes;
      break;
    }

    case SliceKind::Halo: {
      // Sliced with extra halo elements
      int64_t slicedBytes = fullBytes;
      if (rule.sliceDim < static_cast<unsigned>(operandType.getRank())) {
        int64_t origDim = operandType.getDimSize(rule.sliceDim);
        int64_t haloDim = perCoreDim + rule.haloLow + rule.haloHigh;
        haloDim = std::min(haloDim, origDim); // clamp
        if (origDim > 0) {
          slicedBytes = fullBytes * haloDim / origDim;
          result.haloOverheadBytes +=
              fullBytes * (rule.haloLow + rule.haloHigh) / origDim;
        }
      }
      result.splitDataBytes += slicedBytes;
      result.perCoreDmaInBytes += slicedBytes;
      break;
    }

    case SliceKind::Reduce:
      // Reduction output: each core produces a partial sum
      result.perCoreDmaInBytes += fullBytes; // full output buffer needed
      break;
    }
  }

  // DMA cycles
  int64_t perCoreDmaCycles = dmaCycles(result.perCoreDmaInBytes);

  // Reduction penalty: need to accumulate partial results across cores
  int64_t reducePenalty = 0;
  if (result.needsReduce) {
    int64_t outBytes = tensorBytes(outType);
    // Each core writes partial result, then reduce = (numCores-1) reads+adds
    reducePenalty = (activeCores - 1) * dmaCycles(outBytes) +
                    (activeCores - 1) * (outBytes / 2); // rough reduce compute
  }

  // Idle core penalty
  int64_t idlePenalty = 0;
  if (activeCores < numChunks)
    idlePenalty = (numChunks - activeCores) * target_.interCoreSyncCost;

  result.totalCycles = std::max(result.perCoreComputeCycles, perCoreDmaCycles) +
                       reducePenalty + idlePenalty + target_.interCoreSyncCost;

  return result;
}

//===----------------------------------------------------------------------===//
// Level 3 — subgraph (fused kernel) pipeline evaluation
//===----------------------------------------------------------------------===//

CostModel::SubgraphCost CostModel::evaluateSubgraph(
    llvm::ArrayRef<Operation *> ops) const {
  SubgraphCost cost;
  cost.numOps = ops.size();

  if (ops.empty())
    return cost;

  // Build the set of all values produced by ops in this subgraph.
  llvm::DenseSet<Value> producedValues;
  llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
  for (Operation *op : ops) {
    for (Value result : op->getResults())
      producedValues.insert(result);
  }

  // Classify each op's compute cycles into matrix or DSP.
  for (Operation *op : ops) {
    bool isMatrix = isa<linalg::MatmulOp>(op) ||
                    isa<linalg::Conv2DNchwFchwOp>(op);
    int64_t cycles = computeCycles(op);
    if (isMatrix)
      cost.matrixCycles += cycles;
    else
      cost.dspCycles += cycles;
  }

  // Identify external inputs: operands not produced by another op in the
  // subgraph and not block arguments of inner regions.
  llvm::DenseSet<Value> externalInputs;
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      if (!producedValues.contains(operand))
        externalInputs.insert(operand);
    }
  }

  // Identify intermediates and external outputs.
  // Intermediate: produced value consumed ONLY within the subgraph.
  // External output: produced value used outside the subgraph, or result of
  // the last op.
  llvm::DenseSet<Value> externalOutputs;
  llvm::DenseSet<Value> intermediates;
  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      bool usedOutside = false;
      bool usedInside = false;
      for (Operation *user : result.getUsers()) {
        if (opSet.contains(user))
          usedInside = true;
        else
          usedOutside = true;
      }
      if (usedOutside || result.getUses().empty()) {
        externalOutputs.insert(result);
      } else if (usedInside && !usedOutside) {
        intermediates.insert(result);
      }
    }
  }

  // Also: if an op is the last in the list and has no external uses,
  // its results are still external outputs (they are the subgraph output).
  // The logic above handles this via the result.getUses().empty() check,
  // but let's also ensure the last op's results are always outputs.
  if (!ops.empty()) {
    Operation *lastOp = ops.back();
    for (Value result : lastOp->getResults()) {
      if (!externalOutputs.contains(result)) {
        intermediates.erase(result);
        externalOutputs.insert(result);
      }
    }
  }

  // DMA costs for external inputs.
  for (Value input : externalInputs) {
    if (auto shapedType = dyn_cast<ShapedType>(input.getType()))
      cost.dmaInCycles += dmaCycles(tensorBytes(shapedType));
  }

  // DMA costs for external outputs.
  for (Value output : externalOutputs) {
    if (auto shapedType = dyn_cast<ShapedType>(output.getType()))
      cost.dmaOutCycles += dmaCycles(tensorBytes(shapedType));
  }

  // DMA saved by keeping intermediates in SRAM (no DRAM roundtrip).
  for (Value inter : intermediates) {
    if (auto shapedType = dyn_cast<ShapedType>(inter.getType()))
      cost.intermediateDMA += 2 * dmaCycles(tensorBytes(shapedType));
  }

  // Pipeline model: three engines overlap in steady state.
  cost.unpipelinedCycles = cost.dmaInCycles + cost.matrixCycles +
                           cost.dspCycles + cost.dmaOutCycles;
  // With pipelining: DMA_in for next tile overlaps with matrix+dsp of current.
  // Steady state bottleneck = max(dmaIn, matrix, dsp), then drain with dmaOut.
  cost.pipelinedCycles = std::max({cost.dmaInCycles, cost.matrixCycles,
                                   cost.dspCycles}) + cost.dmaOutCycles;

  return cost;
}

FusionDecision CostModel::isSubgraphFusionProfitable(
    llvm::ArrayRef<Operation *> ops) const {
  if (ops.size() <= 1)
    return {false, 0};

  // Fused cost: evaluate the entire subgraph as one pipeline.
  int64_t fusedCost = evaluateSubgraph(ops).pipelinedCycles;

  // Unfused cost: each op runs independently with its own DMA in/out.
  int64_t unfusedCost = 0;
  for (Operation *op : ops) {
    SmallVector<Operation *> singleOp = {op};
    unfusedCost += evaluateSubgraph(singleOp).pipelinedCycles;
  }

  int64_t benefit = unfusedCost - fusedCost;
  return {benefit > 0, benefit};
}

//===----------------------------------------------------------------------===//
// Post-hoc estimation
//===----------------------------------------------------------------------===//

int64_t CostModel::estimateDMAOp(Operation *op) const {
  if (auto dmaCopy = dyn_cast<npu::DMACopyOp>(op)) {
    auto srcType = cast<ShapedType>(dmaCopy.getSrc().getType());
    return dmaCycles(tensorBytes(srcType));
  }
  if (auto dmaStart = dyn_cast<npu::DMAStartOp>(op)) {
    auto srcType = cast<ShapedType>(dmaStart.getSrc().getType());
    return dmaCycles(tensorBytes(srcType));
  }
  return 1;
}

int64_t CostModel::estimateComputeOp(Operation *op) const {
  if (auto computeOp = dyn_cast<npu::ComputeOp>(op)) {
    int64_t total = 0;
    computeOp.getBody().walk([&](Operation *inner) {
      if (isa<linalg::LinalgOp>(inner))
        total += computeCycles(inner);
    });
    return std::max<int64_t>(1, total);
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// MLIR Analysis wrapper
//===----------------------------------------------------------------------===//

CostModelAnalysis::CostModelAnalysis(Operation *op) {
  // Look for npu.target attribute on the module
  HWTarget target;
  if (auto module = dyn_cast<ModuleOp>(op)) {
    if (auto targetAttr = module->getAttrOfType<npu::TargetAttr>("npu.target")) {
      target.numCores = targetAttr.getNumCores();
      target.sramPerCore = targetAttr.getSramPerCore();
      target.dramBandwidth = targetAttr.getDramBandwidth();
      target.sramBandwidth = targetAttr.getSramBandwidth();
      target.matrixThroughput = targetAttr.getMatrixThroughput();
      target.dspThroughput = targetAttr.getDspThroughput();
      target.dmaChannels = targetAttr.getDmaChannels();
    }
  }
  costModel_ = std::make_unique<CostModel>(target);
}
