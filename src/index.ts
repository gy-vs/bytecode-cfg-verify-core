/**
 * Bytecode verifier core.
 *
 * Verification is deliberately staged so that an attacker cannot smuggle
 * executable bytes past the decoder:
 *
 *   1. Linear decode of the ENTIRE code array and construction of the
 *      instruction-boundary table. Unknown opcodes, truncated operands,
 *      integer overflow and non-canonical encodings fail immediately,
 *      even when they occur after an unconditional terminator.
 *   2. Relative-jump resolution: 32-bit overflow check, in-range check,
 *      and instruction-boundary check on every target.
 *   3. Exception-table validation: every start/end/handler offset must be
 *      an instruction boundary, ranges must be non-empty and may only be
 *      disjoint or properly nested (no partial overlap).
 *   4. Control-flow graph construction and reachability from entry 0.
 *
 * Every reported location carries both the raw byte offset and the
 * instruction index.
 *
 * Instruction set
 * ---------------
 *   0x00 NOP                         (1 byte)
 *   0x01 PUSH uleb128/u32 operand    (1 + N bytes, canonical minimal LEB128)
 *   0x02 JMP  rel8                   (2 bytes, target = byte after operand + rel)
 *   0x03 JMP  rel32 little-endian    (5 bytes, target = byte after operand + rel)
 *   0x04 JZ   rel8                   (2 bytes, conditional, falls through)
 *   0x05 HALT                        (1 byte)
 * Every other opcode is unknown and rejected.
 */

export const OP_NOP = 0x00;
export const OP_PUSH = 0x01;
export const OP_JMP = 0x02;
export const OP_JMP32 = 0x03;
export const OP_JZ = 0x04;
export const OP_HALT = 0x05;

const TERMINATORS: ReadonlySet<number> = new Set([OP_JMP, OP_JMP32, OP_HALT]);

export type VerifyErrorCode =
  | 'EMPTY_CODE'
  | 'UNKNOWN_OPCODE'
  | 'TRUNCATED_OPERAND'
  | 'INTEGER_OVERFLOW'
  | 'NONCANONICAL_ENCODING'
  | 'JUMP_TARGET_OUT_OF_RANGE'
  | 'JUMP_TARGET_NOT_BOUNDARY'
  | 'EXCEPTION_RANGE_INVALID'
  | 'EXCEPTION_TARGET_NOT_BOUNDARY'
  | 'EXCEPTION_RANGES_OVERLAP'
  | 'FALLTHROUGH_PAST_END';

export class BytecodeVerifyError extends Error {
  readonly code: VerifyErrorCode;
  /** Instruction index, when the error is attached to an instruction. */
  readonly index?: number;
  /** Raw byte offset of the offending construct. */
  readonly offset?: number;
  /** Raw byte offset of a jump/exception target. */
  readonly targetOffset?: number;
  /** Which field/construct is offending, or structured overlap info. */
  readonly detail?: unknown;

  constructor(
    code: VerifyErrorCode,
    message: string,
    loc?: { index?: number; offset?: number; targetOffset?: number; detail?: unknown },
  ) {
    super(message);
    this.name = 'BytecodeVerifyError';
    this.code = code;
    if (loc) {
      if (loc.index !== undefined) this.index = loc.index;
      if (loc.offset !== undefined) this.offset = loc.offset;
      if (loc.targetOffset !== undefined) this.targetOffset = loc.targetOffset;
      if (loc.detail !== undefined) this.detail = loc.detail;
    }
  }
}

export interface DecodedInstruction {
  /** Zero-based instruction index in decode order. */
  index: number;
  /** Raw byte offset of the opcode. */
  offset: number;
  /** Total length in bytes including opcode and operands. */
  length: number;
  opcode: number;
  /** Decoded operand: unsigned value for PUSH, signed rel for jumps. */
  operand?: number;
}

/** Backward-compatible alias. */
export type Op = DecodedInstruction;

function fail(
  code: VerifyErrorCode,
  message: string,
  loc?: { index?: number; offset?: number; targetOffset?: number; detail?: unknown },
): never {
  throw new BytecodeVerifyError(code, message, loc);
}

/**
 * Add a signed relative displacement to a base byte offset.
 *
 * Relative branches are encoded relative to the byte immediately following
 * the branch's operand (standard VM semantics), so callers pass the
 * post-operand PC as `pc`. The addition is performed in 32-bit unsigned
 * address space: a target below 0 or above 0xFFFFFFFF is an integer
 * overflow/underflow rather than a merely out-of-range target.
 */
export function computeRelativeTarget(pc: number, rel: number, insnIndex?: number): number {
  if (!Number.isInteger(pc) || !Number.isInteger(rel)) {
    fail('INTEGER_OVERFLOW', 'relative displacement arithmetic requires integers');
  }
  const sum = BigInt(pc) + BigInt(rel);
  if (sum < 0n || sum > 0xffffffffn) {
    fail(
      'INTEGER_OVERFLOW',
      `relative target ${sum.toString()} overflows 32-bit address space`,
      { index: insnIndex, detail: { postOperandPc: pc, displacement: rel } },
    );
  }
  return Number(sum);
}

/**
 * Read a canonical unsigned LEB128 value limited to 32 bits, mirrorring the
 * WebAssembly rules:
 *   - a stream ending while the continuation bit is set is truncated;
 *   - more than 5 bytes / continuation set on the 5th byte is too long;
 *   - non-zero unused high bits of the 5th byte overflow u32;
 *   - a multi-byte encoding that could have been shorter is non-canonical.
 */
function readULEB128U32(code: Uint8Array, at: number, insnOffset: number): { value: number; next: number } {
  let result = 0;
  let shift = 0;
  let count = 0;

  for (;;) {
    if (at >= code.length) {
      fail('TRUNCATED_OPERAND', `truncated ULEB128 operand at byte ${at}`, {
        offset: insnOffset,
        targetOffset: at,
      });
    }
    const byte = code[at++];
    count++;

    if (count === 5) {
      if ((byte & 0x80) !== 0) {
        fail('NONCANONICAL_ENCODING', 'ULEB128 representation too long (6th byte required)', {
          offset: insnOffset,
        });
      }
      if ((byte & 0xf0) !== 0) {
        fail('INTEGER_OVERFLOW', 'ULEB128 integer too large for u32', {
          offset: insnOffset,
        });
      }
    }

    result = (result | ((byte & 0x7f) << shift)) >>> 0;

    if ((byte & 0x80) === 0) {
      if (count > 1 && result < 2 ** (7 * (count - 1))) {
        fail('NONCANONICAL_ENCODING', 'ULEB128 operand is not minimally encoded', {
          offset: insnOffset,
          detail: { bytes: count, value: result },
        });
      }
      return { value: result, next: at };
    }

    shift += 7;
  }
}

function readFixedOperand(code: Uint8Array, at: number, width: number, insnOffset: number): number {
  if (at + width > code.length) {
    fail('TRUNCATED_OPERAND', `instruction at ${insnOffset} needs ${width} operand byte(s), ${code.length - at} available`, {
      offset: insnOffset,
      targetOffset: Math.min(at, code.length),
    });
  }
  if (width === 1) {
    const b = code[at];
    return b > 0x7f ? b - 0x100 : b;
  }
  // width === 4, signed little-endian
  const view = new DataView(code.buffer, code.byteOffset, code.byteLength);
  return view.getInt32(at, true);
}

/**
 * Stage 1: linearly decode every byte of `code` into instructions.
 * Throws on the first unknown opcode, truncated operand, overflow or
 * non-canonical encoding.
 */
export function decode(code: Uint8Array): DecodedInstruction[] {
  const out: DecodedInstruction[] = [];
  let at = 0;

  while (at < code.length) {
    const start = at;
    const opcode = code[at++];

    switch (opcode) {
      case OP_NOP:
      case OP_HALT:
        out.push({ index: out.length, offset: start, length: 1, opcode });
        break;

      case OP_PUSH: {
        const { value, next } = readULEB128U32(code, at, start);
        out.push({ index: out.length, offset: start, length: next - start, opcode, operand: value });
        at = next;
        break;
      }

      case OP_JMP:
      case OP_JZ: {
        const rel = readFixedOperand(code, at, 1, start);
        out.push({ index: out.length, offset: start, length: 2, opcode, operand: rel });
        at += 1;
        break;
      }

      case OP_JMP32: {
        const rel = readFixedOperand(code, at, 4, start);
        out.push({ index: out.length, offset: start, length: 5, opcode, operand: rel });
        at += 4;
        break;
      }

      default:
        fail('UNKNOWN_OPCODE', `unknown opcode 0x${opcode.toString(16).padStart(2, '0')} at byte ${start}`, {
          index: out.length,
          offset: start,
          detail: { opcode },
        });
    }
  }

  return out;
}

/** Set of raw byte offsets at which an instruction opcode begins. */
export function boundaries(code: Uint8Array): Set<number> {
  return new Set(decode(code).map((op) => op.offset));
}

export interface ExceptionRangeInput {
  startOffset: number;
  endOffset: number;
  handlerOffset: number;
}

export interface CfgSuccessor {
  index: number;
  offset: number;
  kind: 'fallthrough' | 'jump' | 'exception';
}

export interface VerifiedInstruction extends DecodedInstruction {
  reachable: boolean;
  terminator: boolean;
  successors: CfgSuccessor[];
}

export interface CfgEdge {
  kind: 'fallthrough' | 'jump' | 'exception';
  fromIndex: number;
  fromOffset: number;
  toIndex: number;
  toOffset: number;
}

export interface ExceptionRangeInfo {
  /** Index of the entry in the supplied exception table. */
  index: number;
  startIndex: number;
  startOffset: number;
  endIndex: number;
  endOffset: number;
  handlerIndex: number;
  handlerOffset: number;
}

export interface VerifyResult {
  codeLength: number;
  entry: { index: 0; offset: 0 };
  instructions: VerifiedInstruction[];
  edges: CfgEdge[];
  exceptionRanges: ExceptionRangeInfo[];
}

interface ResolvedRange {
  index: number;
  start: number;
  end: number;
  handler: number;
}

function isBranch(opcode: number): boolean {
  return opcode === OP_JMP || opcode === OP_JMP32 || opcode === OP_JZ;
}

/**
 * Fully verify `code` and return the decoded instructions, boundary-derived
 * CFG edges and resolved exception ranges. Throws BytecodeVerifyError on the
 * first violation.
 */
export function verify(code: Uint8Array, exceptionTable: readonly ExceptionRangeInput[] = []): VerifyResult {
  if (code.length === 0) {
    fail('EMPTY_CODE', 'code array is empty');
  }

  // ---- Stage 1: complete linear decode + boundary table ----------------
  const instructions = decode(code);

  const offsetToIndex = new Map<number, number>();
  for (const insn of instructions) {
    offsetToIndex.set(insn.offset, insn.index);
  }
  const isBoundary = (offset: number): boolean => offsetToIndex.has(offset);

  // ---- Stage 2: relative jumps -----------------------------------------
  // Validated for every instruction, reachable or not.
  const branchTargetOf = new Map<number, number>();
  for (const insn of instructions) {
    if (!isBranch(insn.opcode)) continue;
    const rel = insn.operand ?? 0;
    // Relative to the first byte after the branch's operand. Overflow errors
    // from the address-space check are re-attributed to the branch opcode.
    let target: number;
    try {
      target = computeRelativeTarget(insn.offset + insn.length, rel, insn.index);
    } catch (e) {
      if (e instanceof BytecodeVerifyError && e.code === 'INTEGER_OVERFLOW') {
        fail('INTEGER_OVERFLOW', e.message, {
          index: insn.index,
          offset: insn.offset,
          detail: { postOperandPc: insn.offset + insn.length, displacement: rel },
        });
      }
      throw e;
    }

    if (target < 0 || target >= code.length) {
      fail(
        'JUMP_TARGET_OUT_OF_RANGE',
        `instruction #${insn.index} at byte ${insn.offset} jumps to byte ${target}, code length is ${code.length}`,
        { index: insn.index, offset: insn.offset, targetOffset: target },
      );
    }
    if (!isBoundary(target)) {
      fail(
        'JUMP_TARGET_NOT_BOUNDARY',
        `instruction #${insn.index} at byte ${insn.offset} jumps to byte ${target}, which is not an instruction boundary`,
        { index: insn.index, offset: insn.offset, targetOffset: target },
      );
    }
    branchTargetOf.set(insn.index, target);
  }

  // ---- Stage 3: exception ranges ---------------------------------------
  const resolved: ResolvedRange[] = [];
  exceptionTable.forEach((entry, i) => {
    const { startOffset, endOffset, handlerOffset } = entry;
    for (const [which, value] of [
      ['startOffset', startOffset],
      ['endOffset', endOffset],
      ['handlerOffset', handlerOffset],
    ] as const) {
      if (!Number.isInteger(value) || value < 0 || value > code.length) {
        fail(
          'EXCEPTION_RANGE_INVALID',
          `exception entry #${i} has out-of-range ${which}=${String(value)}`,
          { index: i, detail: { field: which, value } },
        );
      }
    }
    if (startOffset >= endOffset) {
      fail(
        'EXCEPTION_RANGE_INVALID',
        `exception entry #${i} is empty or reversed: [${startOffset}, ${endOffset})`,
        {
          index: i,
          offset: startOffset,
          targetOffset: endOffset,
          detail: { startOffset, endOffset, handlerOffset },
        },
      );
    }
    for (const [which, value] of [
      ['start', startOffset],
      ['handler', handlerOffset],
    ] as const) {
      if (!isBoundary(value)) {
        fail(
          'EXCEPTION_TARGET_NOT_BOUNDARY',
          `exception entry #${i} ${which} offset ${value} is not an instruction boundary`,
          { index: i, offset: value, targetOffset: value, detail: { field: which } },
        );
      }
    }
    // The end is exclusive: it must be a boundary, or equal to the code
    // length (the canonical "end of last instruction" sentinel).
    if (!isBoundary(endOffset) && endOffset !== code.length) {
      fail(
        'EXCEPTION_TARGET_NOT_BOUNDARY',
        `exception entry #${i} end offset ${endOffset} is neither an instruction boundary nor the code length`,
        { index: i, offset: endOffset, targetOffset: endOffset, detail: { field: 'end' } },
      );
    }
    resolved.push({
      index: i,
      start: startOffset,
      end: endOffset,
      handler: handlerOffset,
    });
  });

  // Pairwise nesting check: ranges must be disjoint or fully contained.
  for (let i = 0; i < resolved.length; i++) {
    for (let j = i + 1; j < resolved.length; j++) {
      const a = resolved[i];
      const b = resolved[j];
      const overlap = a.start < b.end && b.start < a.end;
      if (!overlap) continue;
      const aContainsB = a.start <= b.start && b.end <= a.end;
      const bContainsA = b.start <= a.start && a.end <= b.end;
      if (!aContainsB && !bContainsA) {
        fail(
          'EXCEPTION_RANGES_OVERLAP',
          `exception entries #${a.index} [${a.start}, ${a.end}) and #${b.index} [${b.start}, ${b.end}) partially overlap`,
          {
            index: a.index,
            offset: a.start,
            detail: {
              a: { index: a.index, startOffset: a.start, endOffset: a.end, handlerOffset: a.handler },
              b: { index: b.index, startOffset: b.start, endOffset: b.end, handlerOffset: b.handler },
            },
          },
        );
      }
    }
  }

  // ---- Stage 4: CFG edges and reachability ------------------------------
  const edges: CfgEdge[] = [];
  const successorsPerInstruction: CfgSuccessor[][] = instructions.map(() => []);
  const edgeKeys = new Set<string>();

  function addEdge(kind: CfgEdge['kind'], fromIndex: number, toOffset: number): void {
    const toIndex = offsetToIndex.get(toOffset);
    if (toIndex === undefined) return; // boundary failures were reported above
    const edge: CfgEdge = {
      kind,
      fromIndex,
      fromOffset: instructions[fromIndex].offset,
      toIndex,
      toOffset: toOffset,
    };
    const key = `${edge.kind}:${edge.fromIndex}->${edge.toIndex}`;
    if (!edgeKeys.has(key)) {
      edgeKeys.add(key);
      edges.push(edge);
      successorsPerInstruction[fromIndex].push({ index: toIndex, offset: toOffset, kind });
    }
  }

  for (const insn of instructions) {
    const branchOffset = branchTargetOf.get(insn.index);
    if (branchOffset !== undefined) {
      addEdge('jump', insn.index, branchOffset);
    }
    if (!TERMINATORS.has(insn.opcode)) {
      const next = instructions[insn.index + 1];
      if (next) addEdge('fallthrough', insn.index, next.offset);
    }
  }

  // Exception edges: every instruction inside [start, end) can transfer to
  // the handler. Half-open so the boundary `end` is itself not protected.
  for (const range of resolved) {
    for (const insn of instructions) {
      if (insn.offset >= range.start && insn.offset < range.end) {
        addEdge('exception', insn.index, range.handler);
      }
    }
  }

  const reachable = new Set<number>([0]);
  const worklist: number[] = [0];
  while (worklist.length > 0) {
    const current = worklist.pop()!;
    for (const succ of successorsPerInstruction[current]) {
      if (!reachable.has(succ.index)) {
        reachable.add(succ.index);
        worklist.push(succ.index);
      }
    }
  }

  const verifiedInstructions: VerifiedInstruction[] = instructions.map((insn) => ({
    ...insn,
    reachable: reachable.has(insn.index),
    terminator: TERMINATORS.has(insn.opcode),
    successors: successorsPerInstruction[insn.index],
  }));

  // A reachable non-terminator running off the end of the code is illegal.
  for (const insn of verifiedInstructions) {
    if (!insn.reachable || insn.terminator) continue;
    const next = verifiedInstructions[insn.index + 1];
    if (!next) {
      fail(
        'FALLTHROUGH_PAST_END',
        `reachable instruction #${insn.index} at byte ${insn.offset} falls through past the end of code`,
        { index: insn.index, offset: insn.offset, targetOffset: code.length },
      );
    }
  }

  const exceptionRanges: ExceptionRangeInfo[] = resolved.map((r) => {
    const endIndex = offsetToIndex.get(r.end);
    return {
      index: r.index,
      startIndex: offsetToIndex.get(r.start)!,
      startOffset: r.start,
      // end == code.length points one past the last instruction
      endIndex: endIndex ?? instructions.length,
      endOffset: r.end,
      handlerIndex: offsetToIndex.get(r.handler)!,
      handlerOffset: r.handler,
    };
  });

  return {
    codeLength: code.length,
    entry: { index: 0, offset: 0 },
    instructions: verifiedInstructions,
    edges,
    exceptionRanges,
  };
}
