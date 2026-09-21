import { describe, expect, it } from 'vitest';
import {
  boundaries,
  BytecodeVerifyError,
  computeRelativeTarget,
  decode,
  OP_HALT,
  OP_JMP,
  OP_JMP32,
  OP_JZ,
  OP_NOP,
  OP_PUSH,
  verify,
  type ExceptionRangeInput,
} from '../src/index.js';

const NOP = OP_NOP;
const PUSH = OP_PUSH;
const JMP = OP_JMP;
const JMP32 = OP_JMP32;
const JZ = OP_JZ;
const HALT = OP_HALT;

function bytes(...b: number[]): Uint8Array {
  // Wrap negative operands (e.g. signed rel8 = -128) into a byte.
  return Uint8Array.from(b, (v) => v & 0xff);
}

function rel32(n: number): number[] {
  // Encode via Uint8Array so negative displacements (e.g. -0x80000000)
  // are taken modulo 256 instead of being rejected or truncated to 0.
  return [...new Uint8Array(new Int32Array([n]).buffer)];
}

function expectVerifyError(code: Uint8Array, errorCode: string, ranges?: ExceptionRangeInput[]) {
  try {
    verify(code, ranges);
  } catch (e) {
    expect(e).toBeInstanceOf(BytecodeVerifyError);
    expect((e as BytecodeVerifyError).code).toBe(errorCode);
    return e as BytecodeVerifyError;
  }
  throw new Error(`expected BytecodeVerifyError(${errorCode}) but verification succeeded`);
}

describe('stage 1: linear decode', () => {
  it('decodes', () => expect(decode(bytes(PUSH, 7, NOP))).toHaveLength(2));

  it('keeps raw byte offsets, instruction indices and lengths', () => {
    const insns = decode(bytes(NOP, PUSH, 0x80, 0x01, JMP32, 0, 0, 0, 0, HALT));
    expect(insns.map((i) => i.index)).toEqual([0, 1, 2, 3]);
    expect(insns.map((i) => i.offset)).toEqual([0, 1, 4, 9]);
    expect(insns.map((i) => i.length)).toEqual([1, 3, 5, 1]);
    expect(insns[1].operand).toBe(128);
  });

  it('builds the boundary table from opcode offsets only', () => {
    // PUSH u32 occupies offsets 1..5, none of which is a boundary
    expect([...boundaries(bytes(PUSH, 0xff, 0xff, 0xff, 0xff, 0x0f, HALT))]).toEqual([0, 6]);
  });

  it('rejects empty code', () => {
    expectVerifyError(new Uint8Array(0), 'EMPTY_CODE');
  });

  it('rejects unknown opcodes with offset and instruction index', () => {
    const err = expectVerifyError(bytes(NOP, 0xff), 'UNKNOWN_OPCODE');
    expect(err.index).toBe(1);
    expect(err.offset).toBe(1);
  });

  it('rejects unknown opcodes hidden after an unconditional terminator', () => {
    // The 0xdd byte is unreachable, but the whole code array is still decoded.
    const err = expectVerifyError(bytes(HALT, 0xdd), 'UNKNOWN_OPCODE');
    expect(err.offset).toBe(1);
    expect(err.index).toBe(1);
  });

  it('rejects truncated 1-byte, 4-byte and ULEB operands', () => {
    let err = expectVerifyError(bytes(JMP), 'TRUNCATED_OPERAND');
    expect(err.offset).toBe(0);

    err = expectVerifyError(bytes(JMP32, 0, 0, 0), 'TRUNCATED_OPERAND');
    expect(err.offset).toBe(0);

    err = expectVerifyError(bytes(PUSH), 'TRUNCATED_OPERAND');
    expect(err.offset).toBe(0);

    // continuation bit still set when the stream ends
    err = expectVerifyError(bytes(PUSH, 0x80), 'TRUNCATED_OPERAND');
    expect(err.offset).toBe(0);
  });

  it('accepts canonical u32-max LEB128 but rejects overflow and overlong encodings', () => {
    const max = bytes(PUSH, 0xff, 0xff, 0xff, 0xff, 0x0f, HALT);
    expect(decode(max)[0].operand).toBe(0xffffffff);

    expectVerifyError(bytes(PUSH, 0xff, 0xff, 0xff, 0xff, 0x10), 'INTEGER_OVERFLOW');
    expectVerifyError(
      bytes(PUSH, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00),
      'NONCANONICAL_ENCODING',
    );
  });

  it('rejects non-minimal (non-canonical) LEB128 encodings', () => {
    // 0x80 0x00 encodes 0 in two bytes
    expectVerifyError(bytes(PUSH, 0x80, 0x00, HALT), 'NONCANONICAL_ENCODING');
    // 0x81 0x80 0x00 encodes 1 in three bytes
    expectVerifyError(bytes(PUSH, 0x81, 0x80, 0x00, HALT), 'NONCANONICAL_ENCODING');
  });
});

describe('stage 2: relative jumps', () => {
  // rel = target - (opcodeOffset + instructionLength), i.e. relative to the
  // byte immediately after the branch operand.
  it('resolves forward jumps', () => {
    // JMP@0..1 to NOP@2: rel = 2 - 2 = 0
    const r = verify(bytes(JMP, 0, NOP, HALT));
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge).toMatchObject({
      fromIndex: 0,
      fromOffset: 0,
      toIndex: 1,
      toOffset: 2,
    });
    expect(r.instructions.map((i) => i.reachable)).toEqual([true, true, true]);
  });

  it('resolves backward jumps (loops)', () => {
    // NOP@0, NOP@1, HALT@2, JMP@3..4 to NOP@0: rel = 0 - 5 = -5
    // The final JMP itself is not reachable (entry stops at HALT@2).
    const r = verify(bytes(NOP, NOP, HALT, JMP, -5));
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge).toMatchObject({ fromIndex: 3, fromOffset: 3, toIndex: 0, toOffset: 0 });
    expect(r.instructions.map((i) => i.reachable)).toEqual([true, true, true, false]);
  });

  it('marks targets skipped by a loop unreachable', () => {
    // NOP@0, JMP@1..2 rel 2 -> HALT@5; NOP@3(dead); JMP@5..6 rel -7 -> NOP@0
    const r = verify(bytes(NOP, JMP, 2, NOP, HALT, JMP, -7));
    expect(r.edges.filter((e) => e.kind === 'jump').map((e) => e.toOffset)).toEqual([5, 0]);
    expect(r.instructions[2]).toMatchObject({ reachable: false, offset: 3 });
  });

  it('allows zero displacement (jump to the next instruction)', () => {
    // rel 0 lands on the byte after the operand, i.e. HALT@2
    const r = verify(bytes(JMP, 0, HALT));
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge.toOffset).toBe(2);
    expect(edge.toIndex).toBe(1);
  });

  it('allows jumps to self via a negative displacement', () => {
    // JMP@0..1 to itself@0: rel = 0 - 2 = -2; HALT@2 unreachable
    const r = verify(bytes(JMP, -2, HALT));
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge).toMatchObject({ fromIndex: 0, toIndex: 0, toOffset: 0 });
    expect(r.instructions[1].reachable).toBe(false);
  });

  it('allows maximal rel8 forward (+127)', () => {
    // JMP@0..1 rel 127 -> 129... one too far; rel 126 -> 128 (HALT)
    const code = bytes(JMP, 126, ...Array(126).fill(NOP), HALT);
    expect(code.length).toBe(129);
    const r = verify(code);
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge.toOffset).toBe(128);
    expect(edge.toIndex).toBe(127);
  });

  it('uses +127 to reach the byte right after a following instruction too', () => {
    // JMP@0..1 rel 127 -> 129 == length is rejected, demonstrating the limit
    const code = bytes(JMP, 127, ...Array(127).fill(NOP));
    expectVerifyError(code, 'JUMP_TARGET_OUT_OF_RANGE');
  });

  it('allows maximal rel8 backward (-128)', () => {
    // 128 NOPs @0..127, JMP@128..129; target = 130 - 128 = 2
    const code = bytes(...Array(128).fill(NOP), JMP, -128);
    expect(code.length).toBe(130);
    const r = verify(code);
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge.fromOffset).toBe(128);
    expect(edge.toOffset).toBe(2);
    expect(edge.toIndex).toBe(2);
  });

  it('allows maximal positive rel32 forward jumps', () => {
    // JMP32@0..4 rel 127 -> target 132; 127 NOPs @5..131, HALT@132
    const code = bytes(JMP32, ...rel32(127), ...Array(127).fill(NOP), HALT);
    expect(code.length).toBe(133);
    const r = verify(code);
    const edge = r.edges.find((e) => e.kind === 'jump')!;
    expect(edge.toOffset).toBe(132);
  });

  it('detects 32-bit integer overflow/underflow in target arithmetic', () => {
    expect(() => computeRelativeTarget(0xffffffff, 1)).toThrowError(
      /overflows 32-bit/,
    );
    expect(() => computeRelativeTarget(0, -1)).toThrowError(/overflows 32-bit/);
    expect(computeRelativeTarget(0xffffffff, 0)).toBe(0xffffffff);
    expect(computeRelativeTarget(0x80000000, -0x80000000)).toBe(0);

    // Place JMP32 at byte 0 so it underflows: post-operand PC 5 + rel(-6) = -1.
    const err = expectVerifyError(
      bytes(JMP32, ...rel32(-6)),
      'INTEGER_OVERFLOW',
    );
    expect(err.index).toBe(0);
    expect(err.offset).toBe(0);
    expect(err.detail).toMatchObject({ postOperandPc: 5, displacement: -6 });
  });

  it('rejects jumps to code end (offset == code length)', () => {
    // Reach the bad branch through a forward jump so the fallthrough-past-end
    // check cannot mask the range error.
    // JMP@0..1 +1 -> NOP@2; NOP@2 NOP@3; JMP@4..5 +2 -> 8 == length
    const err = expectVerifyError(
      bytes(JMP, 1, NOP, NOP, JMP, 2),
      'JUMP_TARGET_OUT_OF_RANGE',
    );
    expect(err.index).toBe(3);
    expect(err.offset).toBe(4);
    expect(err.targetOffset).toBe(8);
  });

  it('rejects jumps pointing past the end of code', () => {
    // JMP@0..1 +8 -> 10 > length 3
    const err = expectVerifyError(bytes(JMP, 8, HALT), 'JUMP_TARGET_OUT_OF_RANGE');
    expect(err.targetOffset).toBe(10);
  });

  it('rejects jumps into the middle of a PUSH operand', () => {
    // PUSH 7 @0..1; JMP@2..3 rel -3 -> byte 1 (operand interior)
    const code = bytes(PUSH, 7, JMP, -3);
    const err = expectVerifyError(code, 'JUMP_TARGET_NOT_BOUNDARY');
    expect(err.index).toBe(1);
    expect(err.offset).toBe(2);
    expect(err.targetOffset).toBe(1);
  });

  it('rejects jumps into the middle of a 4-byte rel32 operand', () => {
    // NOP@0, JMP32@1..5, JMP@6..7; rel -3 -> post-op 8 - 3 = 5 (operand byte)
    const code = bytes(NOP, JMP32, ...rel32(0), JMP, -3, HALT);
    const err = expectVerifyError(code, 'JUMP_TARGET_NOT_BOUNDARY');
    expect(err.offset).toBe(6);
    expect(err.targetOffset).toBe(5);
  });

  it('validates jumps in unreachable code too', () => {
    // JMP@0..1 rel 2 -> PUSH@4 (boundary), skipping JMP@2..3
    // JMP@2..3 rel +1 -> byte 5 (interior of PUSH 7 @4..5); unreachable
    const code = bytes(JMP, 2, JMP, 1, PUSH, 7, NOP, HALT);
    const err = expectVerifyError(code, 'JUMP_TARGET_NOT_BOUNDARY');
    expect(err.index).toBe(1);
    expect(err.offset).toBe(2);
    expect(err.targetOffset).toBe(5);
  });
});

describe('stage 3: exception ranges', () => {
  // NOP@0 NOP@1 PUSH0@2..3 HALT@4
  const code = bytes(NOP, NOP, PUSH, 0, HALT);

  it('accepts ranges whose end equals the code length', () => {
    // NOP@0 NOP@1 PUSH0@2..3 HALT@4, code length 5
    const r = verify(code, [{ startOffset: 0, endOffset: 5, handlerOffset: 4 }]);
    expect(r.exceptionRanges[0]).toMatchObject({
      index: 0,
      startIndex: 0,
      startOffset: 0,
      endIndex: 4,
      endOffset: 5,
      handlerIndex: 3,
      handlerOffset: 4,
    });
  });

  it('accepts properly nested ranges and identical ranges', () => {
    const nested: ExceptionRangeInput[] = [
      { startOffset: 0, endOffset: 4, handlerOffset: 4 },
      { startOffset: 1, endOffset: 2, handlerOffset: 4 },
    ];
    expect(() => verify(code, nested)).not.toThrow();

    const equal: ExceptionRangeInput[] = [
      { startOffset: 0, endOffset: 4, handlerOffset: 4 },
      { startOffset: 0, endOffset: 4, handlerOffset: 1 },
    ];
    expect(() => verify(code, equal)).not.toThrow();
  });

  it('rejects partially overlapping (non-nested) ranges', () => {
    const bad: ExceptionRangeInput[] = [
      { startOffset: 0, endOffset: 2, handlerOffset: 4 },
      { startOffset: 1, endOffset: 4, handlerOffset: 4 },
    ];
    const err = expectVerifyError(code, 'EXCEPTION_RANGES_OVERLAP', bad);
    expect(err.index).toBe(0);
    expect((err.detail as any).a.index).toBe(0);
    expect((err.detail as any).b.index).toBe(1);
  });

  it('rejects start/end/handler offsets that are not boundaries', () => {
    // byte 3 is the interior of PUSH's operand
    const err = expectVerifyError(code, 'EXCEPTION_TARGET_NOT_BOUNDARY', [
      { startOffset: 3, endOffset: 4, handlerOffset: 4 },
    ]);
    expect(err.offset).toBe(3);

    expectVerifyError(code, 'EXCEPTION_TARGET_NOT_BOUNDARY', [
      { startOffset: 0, endOffset: 3, handlerOffset: 4 },
    ]);
    expectVerifyError(code, 'EXCEPTION_TARGET_NOT_BOUNDARY', [
      { startOffset: 0, endOffset: 4, handlerOffset: 3 },
    ]);
  });

  it('rejects empty or reversed ranges', () => {
    expectVerifyError(code, 'EXCEPTION_RANGE_INVALID', [
      { startOffset: 2, endOffset: 2, handlerOffset: 4 },
    ]);
    expectVerifyError(code, 'EXCEPTION_RANGE_INVALID', [
      { startOffset: 4, endOffset: 0, handlerOffset: 4 },
    ]);
  });

  it('rejects out-of-range exception offsets', () => {
    expectVerifyError(code, 'EXCEPTION_RANGE_INVALID', [
      { startOffset: 0, endOffset: 99, handlerOffset: 4 },
    ]);
  });
});

describe('stage 4: CFG and reachability', () => {
  it('builds fallthrough, jump and exception edges with indices and offsets', () => {
    // NOP@0 NOP@1 PUSH0@2..3 HALT@4; protect [0,4) with handler 4
    const code = bytes(NOP, NOP, PUSH, 0, HALT);
    const r = verify(code, [{ startOffset: 0, endOffset: 4, handlerOffset: 4 }]);

    const exceptionEdges = r.edges.filter((e) => e.kind === 'exception');
    expect(exceptionEdges.map((e) => [e.fromIndex, e.fromOffset])).toEqual([
      [0, 0],
      [1, 1],
      [2, 2],
    ]);
    expect(exceptionEdges.every((e) => e.toOffset === 4 && e.toIndex === 3)).toBe(true);

    const fallthrough = r.edges.filter((e) => e.kind === 'fallthrough');
    expect(fallthrough.map((e) => [e.fromIndex, e.toIndex])).toEqual([
      [0, 1],
      [1, 2],
      [2, 3],
    ]);
  });

  it('marks valid but unreachable instructions (dead code) without failing', () => {
    // JMP@0..1 rel 0 -> HALT@3 (byte after operand +0 = 2... layout below)
    // JMP@0..2? lay out: JMP@0..1 rel +1 -> HALT@3; NOP@2 dead
    const code = bytes(JMP, 1, NOP, HALT);
    const r = verify(code);
    expect(r.instructions.map((i) => i.reachable)).toEqual([true, false, true]);
    expect(r.instructions[1]).toMatchObject({ index: 1, offset: 2, reachable: false });
  });

  it('still fails on a malformed unreachable instruction', () => {
    // JMP@0 +2 -> HALT@2; byte 3 is an unknown opcode and never executes
    const err = expectVerifyError(bytes(JMP, 2, HALT, 0xff), 'UNKNOWN_OPCODE');
    expect(err.index).toBe(2);
    expect(err.offset).toBe(3);
  });

  it('rejects reachable fallthrough past the end of code', () => {
    const err = expectVerifyError(bytes(NOP), 'FALLTHROUGH_PAST_END');
    expect(err.index).toBe(0);
    expect(err.offset).toBe(0);
    expect(err.targetOffset).toBe(1);

    // JZ jumping to itself still needs a fallthrough target at byte 2.
    expectVerifyError(bytes(JZ, -2), 'FALLTHROUGH_PAST_END');
  });

  it('handles conditional branches: jump edge plus fallthrough edge', () => {
    // JZ@0..1 rel +3 -> HALT@5 (2+3), falls through to NOP@2
    // NOP@2 -> PUSH0@3..4 -> HALT@5
    const r = verify(bytes(JZ, 3, NOP, PUSH, 0, HALT));
    const jump = r.edges.find((e) => e.kind === 'jump')!;
    const fall = r.edges.find((e) => e.kind === 'fallthrough')!;
    expect(jump).toMatchObject({ fromIndex: 0, toIndex: 3, toOffset: 5 });
    expect(fall).toMatchObject({ fromIndex: 0, toIndex: 1, toOffset: 2 });
    expect(r.instructions.every((i) => i.reachable)).toBe(true);
  });

  it('reports the entry, code length and per-instruction successors', () => {
    const r = verify(bytes(NOP, HALT));
    expect(r.entry).toEqual({ index: 0, offset: 0 });
    expect(r.codeLength).toBe(2);
    expect(r.instructions[0].successors).toEqual([
      { index: 1, offset: 1, kind: 'fallthrough' },
    ]);
    expect(r.instructions[1].successors).toEqual([]);
    expect(r.instructions[1].terminator).toBe(true);
  });
});
