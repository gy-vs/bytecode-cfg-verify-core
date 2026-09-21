import {
  boundaryIndexTable,
  decode,
  type DecodedInstruction,
} from './decode.js';
import { fail } from './errors.js';

export const U32_MAX = 0xffff_ffff;

/**
 * Compute `pc + disp` in the unsigned 32-bit address space.
 *
 * pc:   [0, 2^32-1]   instruction start (raw byte offset)
 * disp: signed 32-bit relative displacement
 *
 * The sum can leave the unsigned 32-bit range on either side
 * (pc + INT32_MAX may exceed 2^32-1; pc + INT32_MIN may be negative).
 * That is an integer-overflow failure — the target address does not exist.
 *
 * `source` optionally carries the failing jump instruction's own offset and
 * index so the reported error points at the jump, not only at its pc value.
 */
export function resolveRelative(
  pc: number,
  disp: number,
  source?: { offset: number; index: number },
): number {
  const target = pc + disp;
  if (!Number.isInteger(target) || target < 0 || target > U32_MAX) {
    fail(
      'JUMP_OVERFLOW',
      source?.offset ?? pc,
      source?.index ?? null,
      `relative target ${pc} + (${disp}) overflows the unsigned 32-bit address space`,
      target,
    );
  }
  return target;
}

/** Exception-protected region, given in raw byte offsets by the caller. */
export interface ExceptionRangeInput {
  /** First protected instruction offset (inclusive). */
  start: number;
  /** First offset after the protected region (exclusive). */
  end: number;
  /** Handler entry-point offset. */
  handler: number;
  /** Opaque label echoed into the report. */
  tag?: string | number;
}

export interface ExceptionRange {
  start: number;
  end: number;
  handler: number;
  startIndex: number;
  endIndex: number;
  handlerIndex: number;
  depth: number;
  tag?: string | number;
}

export interface CfgEdge {
  kind: 'jump' | 'fallthrough';
  fromIndex: number;
  fromOffset: number;
  toIndex: number;
  toOffset: number;
}

export interface InstructionRecord {
  index: number;
  offset: number;
  endOffset: number;
  mnemonic: string;
  operand?: number;
  reachable: boolean;
  successors: { index: number; offset: number }[];
  /** Indices of exception ranges whose region covers this instruction. */
  protectedBy: number[];
  /** Indices of exception ranges whose handler entry this is. */
  handlerFor: number[];
}

export interface ControlFlowGraph {
  codeLength: number;
  /** Raw byte offsets that are legal instruction starts. */
  boundaries: Set<number>;
  instructions: InstructionRecord[];
  edges: CfgEdge[];
  exceptionRanges: ExceptionRange[];
  /** Entry roots: instruction 0 and every exception handler. */
  roots: number[];
}

/**
 * Full verification pipeline:
 *
 *  1. strict linear decode of every byte (unknown opcode / truncation /
 *     non-canonical WIDE abort here),
 *  2. relative-jump resolution: checked 32-bit arithmetic, target inside
 *     the code array, target an instruction boundary (never an operand
 *     byte and never end-of-code),
 *  3. straight-line fall-through must not run past the end,
 *  4. exception ranges: bounded, boundary-aligned, non-empty, ordered and
 *     laminar (any two ranges are disjoint or properly nested — never
 *     partially overlapping),
 *  5. reachability from instruction 0 and all handlers (a bad instruction
 *     placed where it cannot be reached was already rejected in step 1;
 *     reachability is reported for the record).
 */
export function buildControlFlowGraph(
  code: Uint8Array,
  exceptionInputs: ExceptionRangeInput[] = [],
): ControlFlowGraph {
  const instrs = decode(code);
  const length = code.length;

  const boundary = new Set<number>(instrs.map((i) => i.offset));
  const indexAt = boundaryIndexTable(instrs);

  const edges: CfgEdge[] = [];
  const successors = new Map<number, { index: number; offset: number }[]>();

  const addEdge = (
    kind: CfgEdge['kind'],
    from: DecodedInstruction,
    toOffset: number,
  ): void => {
    const toIndex = indexAt.get(toOffset);
    if (toIndex === undefined) {
      // The decode table contains every instruction start, so an offset not
      // present here lands on an operand byte or on end-of-code.
      fail(
        toOffset === length ? 'JUMP_OUT_OF_RANGE' : 'JUMP_INTO_OPERAND',
        from.offset,
        from.index,
        toOffset === length
          ? `${from.spec.mnemonic} targets end-of-code (offset ${toOffset}), which is not an instruction`
          : `${from.spec.mnemonic} target offset ${toOffset} is not an instruction boundary`,
        toOffset,
      );
    }
    edges.push({
      kind,
      fromIndex: from.index,
      fromOffset: from.offset,
      toIndex,
      toOffset,
    });
    const list = successors.get(from.index) ?? [];
    list.push({ index: toIndex, offset: toOffset });
    successors.set(from.index, list);
  };

  // ---- phase 2/3: relative jumps and straight-line fall-through --------
  for (const ins of instrs) {
    if (ins.spec.jump) {
      const disp = ins.operand!;
      const target = resolveRelative(ins.offset, disp, {
        offset: ins.offset,
        index: ins.index,
      });
      if (target >= length) {
        fail(
          'JUMP_OUT_OF_RANGE',
          ins.offset,
          ins.index,
          `${ins.spec.mnemonic} target offset ${target} is outside the code array (length ${length})`,
          target,
        );
      }
      addEdge('jump', ins, target);
    }

    if (!ins.spec.terminal && ins.endOffset < length) {
      addEdge('fallthrough', ins, ins.endOffset);
    } else if (!ins.spec.terminal && ins.endOffset >= length) {
      // Last instruction falls straight off the end instead of HALT/JUMP.
      fail(
        'FALLTHROUGH_PAST_END',
        ins.offset,
        ins.index,
        `${ins.spec.mnemonic} at the end of the code falls through past offset ${length}`,
      );
    }
  }

  // ---- phase 4: exception table ---------------------------------------
  const rawRanges = exceptionInputs.map((r, i) => validateRangeShape(r, i, length, boundary, indexAt));

  // Laminar nesting check: every pair must be disjoint or fully nested.
  for (let a = 0; a < rawRanges.length; a++) {
    for (let b = a + 1; b < rawRanges.length; b++) {
      const x = rawRanges[a];
      const y = rawRanges[b];
      const disjoint = x.end <= y.start || y.end <= x.start;
      const nested = (x.start <= y.start && y.end <= x.end) || (y.start <= x.start && x.end <= y.end);
      if (!disjoint && !nested) {
        fail(
          'EXCEPTION_RANGE_OVERLAP',
          Math.max(x.start, y.start),
          null,
          `exception ranges #${a} [${x.start},${x.end}) and #${b} [${y.start},${y.end}) partially overlap; ranges must be disjoint or properly nested`,
        );
      }
    }
  }

  // Compute nesting depth and index-resolved range records. Equal ranges
  // are siblings (a handler can be registered twice), not nested.
  const exceptionRanges: ExceptionRange[] = rawRanges
    .map((r) => {
      let depth = 0;
      for (const outer of rawRanges) {
        if (
          outer !== r &&
          outer.start <= r.start &&
          r.end <= outer.end &&
          (outer.start < r.start || r.end < outer.end)
        ) {
          depth++;
        }
      }
      return {
        start: r.start,
        end: r.end,
        handler: r.handler,
        startIndex: indexAt.get(r.start)!,
        // Exclusive end at code length points one past the last instruction.
        endIndex: r.end === length ? instrs.length : indexAt.get(r.end)!,
        handlerIndex: indexAt.get(r.handler)!,
        depth,
        ...(r.tag !== undefined ? { tag: r.tag } : {}),
      };
    })
    .sort((a, b) => a.start - b.start || b.end - a.end);

  // ---- phase 5: reachability from entry + every handler ---------------
  const roots = [0, ...new Set(exceptionRanges.map((r) => r.handlerIndex))];
  const reachable = new Set<number>();
  const queue: number[] = [...roots];
  while (queue.length > 0) {
    const idx = queue.pop()!;
    if (reachable.has(idx)) continue;
    reachable.add(idx);
    for (const succ of successors.get(idx) ?? []) queue.push(succ.index);
  }

  const protectedBy = new Map<number, number[]>();
  const handlerFor = new Map<number, number[]>();
  exceptionRanges.forEach((r, i) => {
    for (const ins of instrs) {
      if (ins.offset >= r.start && ins.offset < r.end) {
        const list = protectedBy.get(ins.index) ?? [];
        list.push(i);
        protectedBy.set(ins.index, list);
      }
    }
    const list = handlerFor.get(r.handlerIndex) ?? [];
    list.push(i);
    handlerFor.set(r.handlerIndex, list);
  });

  const instructions: InstructionRecord[] = instrs.map((ins) => ({
    index: ins.index,
    offset: ins.offset,
    endOffset: ins.endOffset,
    mnemonic: ins.spec.mnemonic,
    ...(ins.operand !== undefined ? { operand: ins.operand } : {}),
    reachable: reachable.has(ins.index),
    successors: successors.get(ins.index) ?? [],
    protectedBy: protectedBy.get(ins.index) ?? [],
    handlerFor: handlerFor.get(ins.index) ?? [],
  }));

  return {
    codeLength: length,
    boundaries: boundary,
    instructions,
    edges,
    exceptionRanges,
    roots,
  };
}

interface ShapeCheckedRange {
  start: number;
  end: number;
  handler: number;
  tag?: string | number;
}

function validateRangeShape(
  r: ExceptionRangeInput,
  rangeIndex: number,
  length: number,
  boundary: Set<number>,
  indexAt: Map<number, number>,
): ShapeCheckedRange {
  const where = `exception range #${rangeIndex}`;

  for (const [name, off] of [
    ['start', r.start],
    ['end', r.end],
    ['handler', r.handler],
  ] as const) {
    if (!Number.isInteger(off) || off < 0 || off > length) {
      fail('EXCEPTION_RANGE_OOB', typeof off === 'number' ? Math.min(Math.max(off, 0), length) : 0, null,
        `${where} ${name}=${String(off)} is outside the code array [0,${length}]`);
    }
  }

  if (r.end < r.start) {
    fail('EXCEPTION_RANGE_BAD_ORDER', r.start, null,
      `${where} [${r.start},${r.end}) is reversed; end must not precede start`);
  }
  if (r.end === r.start) {
    fail('EXCEPTION_RANGE_EMPTY', r.start, null,
      `${where} [${r.start},${r.end}) is empty`);
  }

  // start and handler must be instruction starts. End is exclusive, so it
  // may equal code length (the offset just past the last instruction, which
  // decode guarantees is exactly aligned); otherwise it too must be a start.
  if (!boundary.has(r.start)) {
    fail('EXCEPTION_RANGE_BAD_BOUNDARY', r.start, null,
      `${where} start=${r.start} is not an instruction boundary`);
  }
  if (!boundary.has(r.handler)) {
    fail('EXCEPTION_RANGE_BAD_BOUNDARY', r.handler, null,
      `${where} handler=${r.handler} is not an instruction boundary`);
  }
  if (r.end !== length && !boundary.has(r.end)) {
    fail('EXCEPTION_RANGE_BAD_BOUNDARY', r.end, null,
      `${where} end=${r.end} is not an instruction boundary`);
  }

  return { start: r.start, end: r.end, handler: r.handler, ...(r.tag !== undefined ? { tag: r.tag } : {}) };
}
