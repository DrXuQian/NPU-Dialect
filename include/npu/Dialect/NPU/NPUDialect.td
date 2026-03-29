//===- NPUDialect.td - NPU dialect definition ----------------*- tablegen -*-===//
//
// Defines the NPU dialect for multi-core NPU compilation.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_TD
#define NPU_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// NPU dialect definition
//===----------------------------------------------------------------------===//

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "Dialect for NPU multi-core compiler IR";
  let description = [{
    The NPU dialect provides operations for targeting a multi-core NPU with
    private SRAM, matrix compute units, DSP units, and DMA engines.

    Key abstractions:
    - Core execution regions (`npu.core_execute`)
    - Explicit DMA transfers (`npu.dma_copy`, `npu.dma_start`/`npu.dma_wait`)
    - SRAM buffer management (`npu.alloc_sram`, `npu.dealloc_sram`)
    - Compute unit binding (`npu.compute`)
    - Hardware target configuration (`npu.target`)
  }];
  let cppNamespace = "::npu";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
  let extraClassDeclaration = [{
    /// Materialize a constant value for NPU attributes.
    mlir::Operation *materializeConstant(mlir::OpBuilder &builder,
                                         mlir::Attribute value,
                                         mlir::Type type,
                                         mlir::Location loc) override;
  }];
}

//===----------------------------------------------------------------------===//
// NPU enums
//===----------------------------------------------------------------------===//

def NPU_ComputeUnit : I32EnumAttr<"ComputeUnit",
    "NPU compute unit type",
    [
      I32EnumAttrCase<"Matrix", 0, "matrix">,
      I32EnumAttrCase<"DSP",    1, "dsp">,
    ]> {
  let cppNamespace = "::npu";
}

def NPU_DMADirection : I32EnumAttr<"DMADirection",
    "DMA transfer direction",
    [
      I32EnumAttrCase<"D2S", 0, "dram_to_sram">,
      I32EnumAttrCase<"S2D", 1, "sram_to_dram">,
      I32EnumAttrCase<"S2S", 2, "sram_to_sram">,
    ]> {
  let cppNamespace = "::npu";
}

def NPU_MemorySpace : I32EnumAttr<"MemorySpace",
    "Memory space for buffers",
    [
      I32EnumAttrCase<"DRAM", 0, "dram">,
      I32EnumAttrCase<"SRAM", 1, "sram">,
    ]> {
  let cppNamespace = "::npu";
}

//===----------------------------------------------------------------------===//
// NPU Types
//===----------------------------------------------------------------------===//

class NPU_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<NPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def NPU_TokenType : NPU_Type<"Token", "token"> {
  let summary = "DMA completion token for async operations";
}

//===----------------------------------------------------------------------===//
// NPU Attributes
//===----------------------------------------------------------------------===//

def NPU_TargetAttr : AttrDef<NPU_Dialect, "Target"> {
  let mnemonic = "target";
  let summary = "NPU hardware target configuration";
  let parameters = (ins
    "int64_t":$num_cores,
    "int64_t":$sram_per_core,
    "int64_t":$dram_bandwidth,
    "int64_t":$sram_bandwidth,
    "int64_t":$matrix_throughput,
    "int64_t":$dsp_throughput,
    "int64_t":$dma_channels
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// NPU Operations
//===----------------------------------------------------------------------===//

class NPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPU_Dialect, mnemonic, traits>;

//--- Core execution ---

def NPU_CoreExecuteOp : NPU_Op<"core_execute",
    [SingleBlockImplicitTerminator<"CoreYieldOp">]> {
  let summary = "Execute a region on a specific NPU core";
  let arguments = (ins I64Attr:$core_id);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`core` $core_id $body attr-dict";
}

def NPU_CoreYieldOp : NPU_Op<"core_yield", [Terminator]> {
  let summary = "Terminator for npu.core_execute region";
  let assemblyFormat = "attr-dict";
}

//--- DMA operations ---

def NPU_DMACopyOp : NPU_Op<"dma_copy"> {
  let summary = "Synchronous DMA transfer between memory spaces";
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    NPU_DMADirection:$direction
  );
  let assemblyFormat = "$src `to` $dst `dir` $direction attr-dict `:` type($src) `to` type($dst)";
}

def NPU_DMAStartOp : NPU_Op<"dma_start"> {
  let summary = "Start an asynchronous DMA transfer";
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    NPU_DMADirection:$direction
  );
  let results = (outs NPU_TokenType:$token);
  let assemblyFormat = "$src `to` $dst `dir` $direction attr-dict `:` type($src) `to` type($dst) `->` type($token)";
}

def NPU_DMAWaitOp : NPU_Op<"dma_wait"> {
  let summary = "Wait for an asynchronous DMA transfer to complete";
  let arguments = (ins NPU_TokenType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

//--- SRAM allocation ---

def NPU_AllocSRAMOp : NPU_Op<"alloc_sram"> {
  let summary = "Allocate a buffer in core-local SRAM";
  let arguments = (ins
    OptionalAttr<I64Attr>:$sram_offset
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "(`offset` $sram_offset^)? attr-dict `:` type($result)";
}

def NPU_DeallocSRAMOp : NPU_Op<"dealloc_sram"> {
  let summary = "Deallocate an SRAM buffer";
  let arguments = (ins AnyMemRef:$buffer);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}

//--- Compute binding ---

def NPU_ComputeOp : NPU_Op<"compute",
    [SingleBlockImplicitTerminator<"ComputeYieldOp">]> {
  let summary = "Execute computation on a specific hardware unit";
  let arguments = (ins NPU_ComputeUnit:$unit);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`on` $unit $body attr-dict";
}

def NPU_ComputeYieldOp : NPU_Op<"compute_yield", [Terminator]> {
  let summary = "Terminator for npu.compute region";
  let assemblyFormat = "attr-dict";
}

//--- Synchronization ---

def NPU_BarrierOp : NPU_Op<"barrier"> {
  let summary = "Inter-core synchronization barrier";
  let assemblyFormat = "attr-dict";
}

#endif // NPU_DIALECT_TD
