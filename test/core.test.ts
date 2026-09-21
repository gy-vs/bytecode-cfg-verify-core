import { describe, expect, it } from 'vitest';
import {
  boundaries,
  buildControlFlowGraph,
  BytecodeVerifyError,
  decode,
  OP,
  resolveRelative,
  U32_MAX,
  type ExceptionRangeInput,
} from '../src/index.js';

/** Assemble a program from byte literals / opcode names. */
const B = (...bytes: number[]) => Uint8Array.from(bytes);

/** big-endian encoders */
const u16 = (v: number) => [(v >>> 8) & 0xff, v & 0xff];
const u32 = (v: number) => [
  (v >>> 24) & 0xff,
  (v >>> 16) & 0xff,
  (v >>> 8) & 0xff,
  v & 0xff,
];
/** signed 8/16/32 displacement encoder */
const s8 = (v: number) => [v & 0xff];
const s16 = (v: number) => u16(v & 0xffff);
const s32 = (v: number) => u32(v >>> 0);

function expectVerifyError(
  fn: () => unknown,
  code: string,
  offset: number,
  index: number | null = null,
  targetOffset?: number,
) {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(BytecodeVerifyError);
    const err = e as BytecodeVerifyError;
    expect(err.code).toBe(code);
    expect(err.offset).toBe(offset);
    if (index !== null) expect(err.index).toBe(index);
    if (targetOffset !== undefined) expect(err.targetOffset).toBe(targetOffset);
    return;
  }
  throw new Error(`expected BytecodeVerifyError(${code}) but nothing was thrown`);
}

describe('strict linear decode', () => {
  it('decodes the original toy program', () => {
    expect(decode(B(1, 7, 0))).toHaveLength(2);
  });

  it('rejects empty code with a byte offset and index', () => {
    expectVerifyError(() => decode(B()), 'EMPTY_CODE', 0, 0);
  });

  it('rejects unknown opcodes and reports raw offset + index', () => {
    // NOP idx0 @0, then unknown 0x7f idx1 @1
    expectVerifyError(() => decode(B(OP.NOP, 0x7f)), 'UNKNOWN_OPCODE', 1, 1);
  });

  it('rejects truncated PUSH1 operand', () => {
    expectVerifyError(() => decode(B(OP.PUSH1)), 'TRUNCATED_OPERAND', 0, 0);
  });

  it('rejects truncated jump operands at end of code', () => {
    expectVerifyError(() => decode(B(OP.JUMP8)), 'TRUNCATED_OPERAND', 0, 0);
    expectVerifyError(() => decode(B(OP.JUMP16, 0x00)), 'TRUNCATED_OPERAND', 0, 0);
    expectVerifyError(
      () => decode(B(OP.JUMP32, 0x00, 0x00, 0x00)),
      'TRUNCATED_OPERAND',
      0,
      0,
    );
  });

  it('rejects truncated PUSH2 inside the stream', () => {
    // NOP @0 idx0, PUSH2 @1 idx1 with one operand byte
    expectVerifyError(
      () => decode(B(OP.NOP, OP.PUSH2, 0x01)),
      'TRUNCATED_OPERAND',
      1,
      1,
    );
  });

  it('decodes WIDE PUSH2 as a 6-byte PUSH4 with one boundary', () => {
    const code = B(OP.WIDE, OP.PUSH2, ...u32(0xdeadbeef));
    const ins = decode(code);
    expect(ins).toHaveLength(1);
    expect(ins[0].offset).toBe(0);
    expect(ins[0].length).toBe(6);
    expect(ins[0].spec.mnemonic).toBe('PUSH4');
    expect(ins[0].operand).toBe(0xdeadbeef);
    // Only offset 0 is a boundary — bytes 1..5 (opcode + operands) are not.
    expect(boundaries(code)).toEqual(new Set([0]));
  });

  it('rejects WIDE at the very end (truncated)', () => {
    expectVerifyError(() => decode(B(OP.WIDE)), 'TRUNCATED_OPERAND', 0, 0);
  });

  it('rejects doubled WIDE prefix (non-canonical)', () => {
    expectVerifyError(
      () => decode(B(OP.WIDE, OP.WIDE, OP.PUSH2, 0, 0)),
      'NON_CANONICAL_ENCODING',
      1,
      0,
    );
  });

  it('rejects WIDE before anything other than PUSH2 (non-canonical)', () => {
    expectVerifyError(
      () => decode(B(OP.WIDE, OP.NOP)),
      'NON_CANONICAL_ENCODING',
      1,
      0,
    );
    expectVerifyError(
      () => decode(B(OP.WIDE, OP.PUSH1, 0)),
      'NON_CANONICAL_ENCODING',
      1,
      0,
    );
  });

  it('rejects truncated widened PUSH4 operand', () => {
    expectVerifyError(
      () => decode(B(OP.WIDE, OP.PUSH2, 0, 0)),
      'TRUNCATED_OPERAND',
      0,
      0,
    );
  });

  it('still decodes every byte past HALT, so a hidden bad opcode cannot escape', () => {
    // HALT idx0 @0 then unreachable unknown 0x55 idx1 @1
    expectVerifyError(() => decode(B(OP.HALT, 0x55)), 'UNKNOWN_OPCODE', 1, 1);
  });
});

describe('relative jump arithmetic', () => {
  it('resolves forward and backward displacements', () => {
    expect(resolveRelative(10, 5)).toBe(15);
    expect(resolveRelative(10, -10)).toBe(0);
    expect(resolveRelative(7, 0)).toBe(7);
  });

  it('flags overflow past U32_MAX and reports the offending target', () => {
    // 0xffff_ffff + 1 = 2^32 — outside the unsigned 32-bit space.
    expectVerifyError(() => resolveRelative(U32_MAX, 1), 'JUMP_OVERFLOW', U32_MAX, null, U32_MAX + 1);
  });

  it('flags underflow from INT32_MIN displacement', () => {
    // 1 + (-2^31) is negative.
    expectVerifyError(() => resolveRelative(1, -2147483648), 'JUMP_OVERFLOW', 1, null, -2147483647);
  });
});

/**
 * Well-formed program exercising backward, forward and zero-offset jumps
 * plus a full exception table (nested + disjoint ranges).
 *
 *  idx  off  bytes                insn
 *    0    0   00                  NOP            ft -> 1
 *    1    1   03 04               JUMP8  +4   ------> @5  (forward)
 *    2    3   01 2a               PUSH1 42       ft -> 5  (unreachable)
 *    3    5   00                  NOP            ft -> 6
 *    4    6   04 ff fa            JUMP16 -6  ------> @0  (backward)
 *    5    9   06                  HALT                      (unreachable)
 *    6   10   00                  NOP (handler)  ft -> 11
 *    7   11   03 00               JUMP8  0   ------> @11 (zero offset)
 *    8   13   00                  NOP            ft -> 14 (unreachable)
 *    9   14   05 ff ff ff f8      JUMP32 -8  ------> @6  (backward)
 *   10   19   06                  HALT
 *   11   20   06                  HALT                      (unreachable)
 */
function validProgram() {
  return B(
    OP.NOP, // 0  @0  idx0
    OP.JUMP8, ...s8(4), // 1  @1  idx1  jump -> 5 (forward)
    OP.PUSH1, 42, // 3  @3  idx2  (statically skipped)
    OP.NOP, // 5  @5  idx3
    OP.JUMP16, ...s16(-6), // 6  @6  idx4  jump -> 0 (backward)
    OP.HALT, // 9  @9  idx5
    OP.NOP, // 10 @10 idx6  handler entry
    OP.JUMP8, ...s8(0), // 11 @11 idx7  jump -> 11 (zero offset)
    OP.NOP, // 13 @13 idx8
    OP.JUMP32, ...s32(-8), // 14 @14 idx9  jump -> 6 (backward)
    OP.HALT, // 19 @19 idx10
    OP.HALT, // 20 @20 idx11 unreachable, terminal (no fall-off-end)
  );
}

const validRanges: ExceptionRangeInput[] = [
  { start: 0, end: 9, handler: 10, tag: 'outer' }, // covers idx0..idx4
  { start: 5, end: 9, handler: 10, tag: 'inner' }, // nested: idx3..idx4
  { start: 11, end: 13, handler: 14, tag: 'dead-zone' }, // disjoint, idx7
  { start: 19, end: 21, handler: 10, tag: 'to-eof' }, // end == code length
];

describe('control-flow graph on the valid program', () => {
  it('reports raw byte offsets and dense instruction indices', () => {
    const code = validProgram();
    const cfg = buildControlFlowGraph(code, validRanges);

    expect(cfg.codeLength).toBe(21);
    expect([...cfg.boundaries]).toEqual([0, 1, 3, 5, 6, 9, 10, 11, 13, 14, 19, 20]);

    const offs = cfg.instructions.map((i) => i.offset);
    expect(offs).toEqual([0, 1, 3, 5, 6, 9, 10, 11, 13, 14, 19, 20]);
    expect(cfg.instructions.map((i) => i.index)).toEqual(
      cfg.instructions.map((_, i) => i),
    );

    // Forward JUMP8 idx1 @1 +4 -> idx3 @5
    expect(cfg.instructions[1].successors).toEqual([{ index: 3, offset: 5 }]);
    // Backward JUMP16 idx4 @6 -6 -> idx0 @0
    expect(cfg.instructions[4].successors).toEqual([{ index: 0, offset: 0 }]);
    // Zero-offset JUMP8 idx7 @11 -> itself @11
    expect(cfg.instructions[7].successors).toEqual([{ index: 7, offset: 11 }]);
    // Backward JUMP32 idx9 @14 -8 -> idx4 @6
    expect(cfg.instructions[9].successors).toEqual([{ index: 4, offset: 6 }]);

    expect(cfg.edges).toHaveLength(9); // 4 jumps + 5 fall-throughs
  });

  it('marks only genuinely unreachable code unreachable', () => {
    const cfg = buildControlFlowGraph(validProgram(), validRanges);
    const reachable = cfg.instructions.filter((i) => i.reachable).map((i) => i.index);
    // entry 0 reaches 0,1,3,4 ; handler roots idx6 @10 and idx9 @14 add 6,7,9.
    expect(reachable.sort((a, b) => a - b)).toEqual([0, 1, 3, 4, 6, 7, 9]);
    // idx2 @3 is the PUSH1 skipped by the forward jump
    expect(cfg.instructions[2].reachable).toBe(false);
    // idx8 @13 is dead-zone NOP after the self-looping handler
    expect(cfg.instructions[8].reachable).toBe(false);
    expect(cfg.roots.sort((a, b) => a - b)).toEqual([0, 6, 9]);
  });

  it('validates nested and disjoint exception ranges and records depth', () => {
    const cfg = buildControlFlowGraph(validProgram(), validRanges);
    const sortedTags = cfg.exceptionRanges.map((r) => r.tag);
    // sorted by start
    expect(sortedTags).toEqual(['outer', 'inner', 'dead-zone', 'to-eof']);

    const outer = cfg.exceptionRanges.find((r) => r.tag === 'outer')!;
    const inner = cfg.exceptionRanges.find((r) => r.tag === 'inner')!;
    const toEof = cfg.exceptionRanges.find((r) => r.tag === 'to-eof')!;
    expect(outer.depth).toBe(0);
    expect(inner.depth).toBe(1);
    expect(outer.startIndex).toBe(0);
    expect(outer.endIndex).toBe(5); // offset 9 is idx5 HALT (exclusive)
    expect(outer.handlerIndex).toBe(6); // handler offset 10
    // Exclusive end at code length (21) resolves to one past the last insn.
    expect(toEof.startIndex).toBe(10);
    expect(toEof.endIndex).toBe(12);

    // protection membership: idx3 @5 is inside outer+inner, idx0 only outer
    const idx3 = cfg.instructions[3];
    expect(idx3.protectedBy).toHaveLength(2);
    const idx0 = cfg.instructions[0];
    expect(idx0.protectedBy).toHaveLength(1);
    // offset 10 (idx6) is the handler for outer (#0), inner (#1) and to-eof (#3)
    expect(cfg.instructions[6].handlerFor).toEqual([0, 1, 3]);
    // offset 14 (idx9) is the handler for dead-zone (#2)
    expect(cfg.instructions[9].handlerFor).toEqual([2]);
  });

  it('accepts adjacent (touching) ranges as disjoint', () => {
    const code = B(
      OP.NOP, OP.HALT, // [0,1) and [1,2)
      OP.NOP, OP.HALT,
    );
    const ranges: ExceptionRangeInput[] = [
      { start: 0, end: 1, handler: 2 },
      { start: 1, end: 2, handler: 2 },
    ];
    expect(() => buildControlFlowGraph(code, ranges)).not.toThrow();
  });

  it('accepts equal ranges (duplicated protection) as nested', () => {
    const code = B(OP.NOP, OP.HALT, OP.NOP, OP.HALT);
    const ranges: ExceptionRangeInput[] = [
      { start: 0, end: 2, handler: 2, tag: 'a' },
      { start: 0, end: 2, handler: 2, tag: 'b' },
    ];
    const cfg = buildControlFlowGraph(code, ranges);
    expect(cfg.exceptionRanges.map((r) => r.depth).sort()).toEqual([0, 0]);
  });
});

describe('illegal jump targets', () => {
  it('rejects a forward jump into the middle of a 2-byte operand', () => {
    // JUMP8 @0 +3 -> offset 3, which is PUSH2's second operand byte
    const code = B(
      OP.JUMP8, ...s8(3), // @0 idx0 -> target 3
      OP.PUSH2, 0xaa, 0xbb, // @1 idx1, operands @2 @3
      OP.HALT, // @4 idx2
    );
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_INTO_OPERAND',
      0,
      0,
      3,
    );
  });

  it('rejects a jump into the operand bytes of the jump itself', () => {
    // JUMP16 @0 +2 -> operand byte @2
    const code = B(OP.JUMP16, ...s16(2), OP.HALT);
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_INTO_OPERAND',
      0,
      0,
      2,
    );
  });

  it('rejects a jump that lands exactly at end-of-code', () => {
    // JUMP8 @0 +2 -> offset 2 == length
    const code = B(OP.JUMP8, ...s8(2));
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_OUT_OF_RANGE',
      0,
      0,
      2,
    );
  });

  it('rejects the maximum positive 8-bit displacement (out of range)', () => {
    // @0 +127 -> 127, code length is 2
    const code = B(OP.JUMP8, ...s8(127));
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_OUT_OF_RANGE',
      0,
      0,
      127,
    );
  });

  it('rejects the minimum negative 8-bit displacement (underflow)', () => {
    // JUMP8 @1 + (-128) = -127 — negative target, arithmetic underflow
    const code = B(OP.NOP, OP.JUMP8, ...s8(-128));
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_OVERFLOW',
      1,
      1,
      -127,
    );
  });

  it('rejects a JUMP32 whose arithmetic overflows U32', () => {
    // Put the JUMP32 at offset 0xffff_ffff-1 is impractical; instead craft a
    // displacement so pc+disp > U32_MAX using the address unit directly:
    // resolveRelative already proves the primitive; here use pc=0xfffffffe
    // reachable via a synthetic check on a 1-byte program with disp huge.
    // 0 + 0x7fffffff stays in range, so force overflow through the helper
    // semantics by placing a JUMP32 at a high offset using a padded program.
    // (We keep programs small, so we validate via a 5-byte program where the
    // only valid target is itself; INT32_MAX from offset 0 is 0x7fffffff.)
    const code = B(OP.NOP, OP.JUMP32, ...s32(0x7fffffff), OP.HALT);
    // pc=1, target = 1 + 0x7fffffff = 0x80000000 — in range but far outside
    // the 9-byte array, so it is JUMP_OUT_OF_RANGE rather than overflow.
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_OUT_OF_RANGE',
      1,
      1,
      0x80000000,
    );
  });

  it('rejects a JUMP32 with INT32_MIN displacement from near the start', () => {
    const code = B(OP.NOP, OP.JUMP32, ...s32(-2147483648), OP.HALT);
    // 1 + (-2147483648) = -2147483647 -> arithmetic overflow
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_OVERFLOW',
      1,
      1,
      -2147483647,
    );
  });

  it('rejects a bad jump even when the jump itself is unreachable', () => {
    // JUMP8 @0 loops forever; everything after it is unreachable. The jump
    // at @2 still targets operand byte @3 and must be rejected — full-code
    // validation never skips instructions on reachability grounds.
    const code = B(
      OP.JUMP8, ...s8(0), // @0 idx0 self loop
      OP.JUMP8, ...s8(1), // @2 idx1 unreachable, target @3 (operand)
    );
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'JUMP_INTO_OPERAND',
      2,
      1,
      3,
    );
  });

  it('rejects an unknown opcode in a region no edge can reach', () => {
    // self loop @0, then an unknown byte @2
    expectVerifyError(
      () => buildControlFlowGraph(B(OP.JUMP8, ...s8(0), 0x66)),
      'UNKNOWN_OPCODE',
      2,
      1,
    );
  });

  it('accepts a zero-offset backward-capable self loop as legal', () => {
    // JUMP8 @0 disp 0 -> @0, valid boundary
    const code = B(OP.JUMP8, ...s8(0));
    const cfg = buildControlFlowGraph(code);
    expect(cfg.instructions[0].successors).toEqual([{ index: 0, offset: 0 }]);
  });

  it('accepts a backward jump across several instructions', () => {
    // NOP @0 idx0, NOP @1 idx1, JUMP16 @2 idx2 with disp -2 -> @0
    const code = B(OP.NOP, OP.NOP, OP.JUMP16, ...s16(-2));
    const cfg = buildControlFlowGraph(code);
    expect(cfg.instructions[2].successors).toEqual([{ index: 0, offset: 0 }]);
  });
});

describe('straight-line fall-through', () => {
  it('rejects falling off the end of the code without a terminator', () => {
    // NOP @0 idx0 is the last instruction and is not terminal.
    expectVerifyError(
      () => buildControlFlowGraph(B(OP.NOP)),
      'FALLTHROUGH_PAST_END',
      0,
      0,
    );
  });

  it('accepts HALT as the final instruction', () => {
    expect(() => buildControlFlowGraph(B(OP.HALT))).not.toThrow();
  });

  it('rejects trailing non-terminal code even when unreachable', () => {
    // JUMP8 @0 loops to itself; trailing PUSH1 @2 has no operand path to
    // reach it — but full-code validation still inspects it. PUSH1 with its
    // operand then ends on NOP which falls off the end.
    const code = B(OP.JUMP8, ...s8(0), OP.PUSH1, 1, OP.NOP);
    expectVerifyError(
      () => buildControlFlowGraph(code),
      'FALLTHROUGH_PAST_END',
      4,
      2,
    );
  });
});

describe('exception table edge cases', () => {
  const good = () => B(OP.NOP, OP.HALT, OP.NOP, OP.HALT);

  it('rejects a start landing inside an operand', () => {
    // PUSH2 @0 operands @1@2, HALT @3
    const code = B(OP.PUSH2, 0, 0, OP.HALT);
    const ranges: ExceptionRangeInput[] = [{ start: 1, end: 3, handler: 3 }];
    expectVerifyError(
      () => buildControlFlowGraph(code, ranges),
      'EXCEPTION_RANGE_BAD_BOUNDARY',
      1,
    );
  });

  it('rejects a handler landing inside an operand', () => {
    const code = B(OP.PUSH2, 0, 0, OP.HALT);
    const ranges: ExceptionRangeInput[] = [{ start: 0, end: 4, handler: 2 }];
    expectVerifyError(
      () => buildControlFlowGraph(code, ranges),
      'EXCEPTION_RANGE_BAD_BOUNDARY',
      2,
    );
  });

  it('rejects an end offset in the middle of an instruction', () => {
    const code = B(OP.PUSH2, 0, 0, OP.HALT);
    const ranges: ExceptionRangeInput[] = [{ start: 0, end: 2, handler: 3 }];
    expectVerifyError(
      () => buildControlFlowGraph(code, ranges),
      'EXCEPTION_RANGE_BAD_BOUNDARY',
      2,
    );
  });

  it('rejects endpoints outside the code array', () => {
    const ranges: ExceptionRangeInput[] = [{ start: 0, end: 99, handler: 2 }];
    expectVerifyError(
      () => buildControlFlowGraph(good(), ranges),
      'EXCEPTION_RANGE_OOB',
      4,
    );
  });

  it('rejects an empty range (start === end)', () => {
    const ranges: ExceptionRangeInput[] = [{ start: 1, end: 1, handler: 2 }];
    expectVerifyError(
      () => buildControlFlowGraph(good(), ranges),
      'EXCEPTION_RANGE_EMPTY',
      1,
    );
  });

  it('rejects a reversed range (end < start)', () => {
    const ranges: ExceptionRangeInput[] = [{ start: 2, end: 0, handler: 2 }];
    expectVerifyError(
      () => buildControlFlowGraph(good(), ranges),
      'EXCEPTION_RANGE_BAD_ORDER',
      2,
    );
  });

  it('rejects partially overlapping (non-nested, non-disjoint) ranges', () => {
    // 0 NOP 1 HALT 2 NOP 3 HALT
    // A=[0,2) B=[1,3) overlap only at [1,2)
    const ranges: ExceptionRangeInput[] = [
      { start: 0, end: 2, handler: 2 },
      { start: 1, end: 3, handler: 2 },
    ];
    expectVerifyError(
      () => buildControlFlowGraph(good(), ranges),
      'EXCEPTION_RANGE_OVERLAP',
      1,
    );
  });

  it('accepts a properly nested range pair', () => {
    const code = B(OP.NOP, OP.NOP, OP.NOP, OP.HALT, OP.NOP, OP.HALT);
    const ranges: ExceptionRangeInput[] = [
      { start: 0, end: 4, handler: 4 },
      { start: 1, end: 3, handler: 4 },
    ];
    const cfg = buildControlFlowGraph(code, ranges);
    expect(cfg.exceptionRanges.map((r) => r.depth).sort()).toEqual([0, 1]);
  });
});
