# Vertical Fusion Algorithm Design

## 1. Memory Hierarchy Model (Generalized)

Different NPU designs have different inter-core interconnects:

```
MemoryHierarchy:
  Level 0: Core-private SRAM     (fastest, smallest)
  Level 1: NoC / Shared L1       (inter-core, medium)  ← NPU-specific
  Level 2: DRAM                   (slowest, largest)
```

| NPU Type     | Level 1           | Spatial cost            | Temporal cost      |
|--------------|-------------------|-------------------------|--------------------|
| NoC mesh     | NoC transfer      | NoC BW × bytes (cheap)  | SRAM↔DRAM DMA      |
| Shared cache | Hardware coherence | ≈ 0                     | L1↔DRAM            |
| DRAM-only    | None              | 2 × DMA (expensive)     | SRAM↔DRAM DMA      |
| Shared SRAM  | Direct access     | address-mapped (cheap)  | if fits: 0         |

**Key insight**: Spatial tiling cost depends on interconnect type.
Temporal tiling cost is always SRAM↔DRAM DMA.

## 2. Vertical Fusion Algorithm

### 2.1 Definitions

- **Fused Group (Kernel)**: A connected subgraph of ops that executes as one
  unit on a core, with intermediates staying in SRAM.
- **Temporal Tile**: One iteration of the innermost compute loop. The working
  set of one tile must fit in SRAM.
- **Tile Alignment**: All ops in a fused group must agree on compatible tile
  sizes such that:
  - The output tile of op_i matches the input tile of op_{i+1}
  - All tile working sets fit in SRAM simultaneously
  - For residual connections: the skip tensor also fits

### 2.2 Baseline Algorithm

```
VERTICAL_FUSION(graph, hw):
  1. Compute per-op cost: cycles[op] = computeCycles(op) for all ops
  2. Sort ops by cost descending → priority_queue
  3. Initialize: each op is its own group

  4. FOR EACH anchor_op in priority_queue:
     IF anchor_op already in a multi-op group: SKIP

     5. GREEDY EXPANSION:
        current_group = {anchor_op}
        best_group = {anchor_op}
        best_cost = evaluate_group_cost(best_group)

        FOR direction IN [forward (consumers), backward (producers)]:
          WHILE |current_group| < MAX_FUSION_LENGTH:
            candidates = get_neighbors(current_group, direction)
            candidates = filter(candidates, single_use_edge)

            FOR EACH candidate IN candidates:
              trial_group = current_group ∪ {candidate}

              // CHECK TILE ALIGNMENT
              aligned_tiles = find_aligned_tile_config(trial_group, hw)
              IF aligned_tiles == NONE:
                CONTINUE  // can't align → skip this candidate

              // CHECK SRAM BUDGET (including residual connections)
              working_set = compute_working_set(trial_group, aligned_tiles)
              IF working_set > hw.sram_per_core:
                // Try smaller tiles
                aligned_tiles = find_aligned_tile_config(trial_group, hw,
                                                          reduced_budget=True)
                IF aligned_tiles == NONE: CONTINUE

              // EVALUATE COST
              fused_cost = evaluate_fused_cost(trial_group, aligned_tiles)
              IF fused_cost < best_cost:
                best_group = trial_group
                best_cost = fused_cost
                current_group = trial_group
                BREAK  // greedy: accept first improvement, expand further

        6. IF |best_group| > 1:
             mark best_group as a fused kernel
```

### 2.3 Tile Alignment Algorithm

```
FIND_ALIGNED_TILE_CONFIG(group, hw):
  // The "anchor" op determines tile sizes, others must adapt.
  anchor = most_compute_intensive_op(group)
  anchor_spec = get_tiling_spec(anchor)

  // Enumerate candidate tile sizes for the anchor
  FOR tile_config IN enumerate_tile_candidates(anchor, hw.sram_per_core):
    // Propagate tile sizes to all other ops in the group
    aligned = True
    total_working_set = 0

    FOR op IN topological_order(group):
      IF op == anchor:
        op_tile_shape = tile_config.output_shape
        total_working_set += tile_working_set(anchor, tile_config)
        CONTINUE

      // Infer this op's tile shape from its producer's output tile
      producer_output_tile = get_producer_output_tile(op, group)
      op_input_tile = producer_output_tile  // same tensor, no transform needed
      op_output_tile = infer_output_from_input(op, op_input_tile)
      // For ops with different shapes (e.g. pooling), the indexing map
      // gives the output tile shape from the input tile shape.

      op_working_set = input_tile_bytes + output_tile_bytes
      total_working_set += op_working_set

      // Check: does this tile fit for this op?
      IF any_dim_of(op_output_tile) < 1:
        aligned = False; BREAK

    // Handle residual connections: skip tensors that persist across
    // the entire group must also fit in SRAM
    FOR skip_tensor IN find_residual_tensors(group):
      skip_tile_bytes = compute_skip_tile_bytes(skip_tensor, tile_config)
      total_working_set += skip_tile_bytes

    IF aligned AND total_working_set <= hw.sram_per_core:
      RETURN (tile_config, total_working_set)

  RETURN NONE  // no valid alignment found
```

### 2.4 Residual Connection Handling

```
ResNet block: input → conv1 → relu → conv2 → add(conv2_out, input)
                |                               ↑
                └──────── residual (skip) ───────┘
```

When fusing {conv1, relu, conv2, add}:
- Intermediates (conv1_out, relu_out) are ephemeral: produced and consumed
  within the group, don't need persistent SRAM.
- But `input` (the skip connection) must persist in SRAM from the beginning
  of conv1 until the end of add — the FULL duration of the group.
- SRAM budget for one temporal tile:
  ```
  conv1_input_tile + conv1_weight + conv1_output_tile  (conv1 phase)
  + relu_tile                                           (relu phase, reuses conv1_out)
  + conv2_weight + conv2_output_tile                    (conv2 phase, reuses relu_out)
  + input_skip_tile                                     (persists entire time!)
  + add_output_tile                                     (add phase)
  ```
- The skip tensor tile size must match the add's input tile size, which is
  determined by the tile alignment propagation.

### 2.5 Cost Model Integration

```
EVALUATE_FUSED_COST(group, tile_config):
  // Three-engine pipeline for one temporal tile:
  //   DMA_in → Matrix → DSP → DMA_out
  //
  // Intermediates stay in SRAM → no DMA between ops
  // Only external inputs need DMA_in, only final output needs DMA_out

  dma_in = Σ dmaCycles(external_input_tile_bytes)
  matrix_cycles = Σ computeCycles(matrix_op, tile) for matrix ops
  dsp_cycles = Σ computeCycles(dsp_op, tile) for dsp ops
  dma_out = Σ dmaCycles(external_output_tile_bytes)

  num_tiles = total_output_elements / tile_output_elements

  // Pipelined: DMA/matrix/DSP overlap in steady state
  IF num_tiles == 1:
    total = dma_in + matrix_cycles + dsp_cycles + dma_out
  ELSE:
    steady = max(dma_in, matrix_cycles + dsp_cycles)
    total = dma_in + (num_tiles - 1) * steady + matrix_cycles + dsp_cycles + dma_out

  RETURN total
```

## 3. Better-than-Baseline Algorithms (from literature)

### 3.1 WELDER (OSDI'23): Tile-Graph with Shape Propagation

**Improvement over baseline**: Instead of greedy expansion from one anchor,
WELDER builds a **tile-graph** over the entire computation graph and
simultaneously solves tile alignment for all ops using constraint propagation.

Key ideas:
- Each op is a node with a "tile config" variable
- Edges represent tile shape constraints (output shape of producer = input
  shape of consumer, via indexing map transformation)
- Solve for globally optimal tile shapes using propagation + search
- Hierarchical: start with the full graph, recursively partition into
  sub-graphs that fit in each memory level

**When it's better**: Complex graphs with many fusion candidates where greedy
order matters. Avoids "greedy lock-in" where fusing A-B prevents the better
fusion B-C.

### 3.2 XLA Priority Fusion: Global Priority Queue

**Improvement over baseline**: Uses a global cost model to prioritize fusions
by estimated runtime reduction, not by op cost.

Key ideas:
- Maintain a priority queue of all candidate fusions (producer-consumer pairs)
- Priority = estimated runtime reduction from fusing this pair
- Pop the highest-priority pair, fuse it, update all affected candidates
- Re-evaluate priorities after each fusion (fusion may enable/invalidate others)

**When it's better**: Avoids locally-optimal but globally-suboptimal fusions.
The priority re-evaluation catches cascading effects.

### 3.3 TileFlow (MICRO'23): Tree-Based Dataflow Analysis

**Improvement over baseline**: Precisely models data movement in the memory
hierarchy using a tree structure, enabling accurate cost estimation for
complex fusion patterns (not just linear chains).

Key ideas:
- Fusion dataflow = tree where leaves are ops, internal nodes are loop nests
- Tree structure encodes: compute ordering, resource binding, loop tiling
- Analytical model computes exact data movement volume at each memory level
- Can evaluate branching patterns (diamond, residual) not just linear chains

### 3.4 Recommended Approach

Combine:
1. **Baseline greedy expansion** (yours) for initial grouping
2. **WELDER-style tile alignment propagation** for solving tile shapes
3. **XLA-style priority queue** for fusion ordering
4. **TileFlow-style tree analysis** for cost evaluation of complex patterns

## 4. References

- WELDER (OSDI'23): https://www.usenix.org/conference/osdi23/presentation/shi
- TileFlow (MICRO'23): https://dl.acm.org/doi/10.1145/3613424.3623792
- XLA Priority Fusion: https://github.com/openxla/xla/discussions/6407
- XLA Cost Model: https://github.com/openxla/xla/discussions/10065
- Optimus: https://dl.acm.org/doi/full/10.1145/3520142
- eIQ Neutron NPU: https://arxiv.org/html/2509.14388v1
