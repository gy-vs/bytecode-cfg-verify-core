# Bytecode verifier core

Strict two-phase bytecode verifier that builds a control-flow graph only
**after** the whole code array has been proven structurally sound. It closes
the classic hole where a jump target is range-checked against the code array
but not against the instruction-boundary table, allowing a jump into a
constant operand whose bytes then get interpreted as opcodes.

Run `npm install`, then `npm test` and `npm run build`.

## Instruction set

All immediates are big-endian.

| opcode | mnemonic | operand |
|--------|----------|---------|
| `0x00` | NOP      | — |
| `0x01` | PUSH1    | 1-byte unsigned immediate |
| `0x02` | PUSH2    | 2-byte unsigned immediate |
| `0x03` | JUMP8    | 1-byte **signed** relative displacement |
| `0x04` | JUMP16   | 2-byte signed relative displacement |
| `0x05` | JUMP32   | 4-byte signed relative displacement |
| `0x06` | HALT     | — (terminator, no fall-through) |
| `0x07` | WIDE     | prefix: only legal immediately before `PUSH2`, widening it to `PUSH4` (4-byte immediate) |

Jump displacements are relative to the **start** of the jump instruction, so
a zero displacement jumps to the instruction itself.

## Verification pipeline

`buildControlFlowGraph(code, exceptionRanges)` fails closed (throws
`BytecodeVerifyError`) and never produces a partial CFG:

1. **Full linear decode** — every byte is swept, including code after HALT or
   otherwise unreachable code. Failure codes:
   `EMPTY_CODE`, `UNKNOWN_OPCODE`, `TRUNCATED_OPERAND`,
   `NON_CANONICAL_ENCODING` (WIDE at end / doubled / not followed by PUSH2).
2. **Boundary table** — the set of raw byte offsets where instructions
   legally start. Operand bytes and end-of-code are not boundaries.
3. **Relative jumps** — `pc + disp` is computed in the unsigned 32-bit
   address space and rejected on integer overflow (`JUMP_OVERFLOW`); the
   target must be inside the code array (`JUMP_OUT_OF_RANGE`, incl.
   end-of-code) and an instruction boundary (`JUMP_INTO_OPERAND`).
4. **Fall-through** — a non-terminal final instruction is
   `FALLTHROUGH_PAST_END`.
5. **Exception ranges** — `start`/`handler` must be instruction starts;
   `end` is exclusive and may equal code length; ranges must be non-empty
   (`EXCEPTION_RANGE_EMPTY`), ordered (`EXCEPTION_RANGE_BAD_ORDER`),
   in-bounds (`EXCEPTION_RANGE_OOB`), boundary-aligned
   (`EXCEPTION_RANGE_BAD_BOUNDARY`), and laminar: every pair is disjoint or
   properly nested, never partially overlapping (`EXCEPTION_RANGE_OVERLAP`).
6. **Reachability** is reported per instruction from roots (index 0 plus
   every handler). It is informational only — bad instructions are rejected
   in step 1 whether reachable or not.

Every `BytecodeVerifyError` carries the raw byte `offset`, the dense
instruction `index` (when known), and for bad jumps the computed
`targetOffset`. The returned graph likewise reports both `offset` and
`index` for every instruction, edge and exception range.

## API

```ts
import {
  decode,                 // strict full decode -> DecodedInstruction[]
  boundaries,             // Set<number> of instruction-start offsets
  buildControlFlowGraph,  // decode + verify + CFG/reachability
  resolveRelative,        // checked pc + disp in the u32 address space
  BytecodeVerifyError,
} from 'bytecode-cfg-verify-core';
```
