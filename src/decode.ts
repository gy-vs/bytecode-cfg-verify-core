import { fail } from './errors.js';
import { INSTRUCTIONS, OP, WIDE_PUSH4, type InstructionSpec } from './isa.js';

/**
 * One fully decoded instruction.
 *
 *  offset      raw byte offset of the first byte of the instruction
 *              (also the boundary offset; for WIDE PUSH4 it is the WIDE byte)
 *  index       dense instruction index, in decode order (0-based)
 *  length      total size in bytes, including every operand byte
 *  spec        resolved instruction spec (PUSH4 for a WIDE-prefixed PUSH2)
 *  operand     decoded operand value, present whenever spec.operandSize > 0.
 *              Relative jumps are sign-extended to a JS number.
 */
export interface DecodedInstruction {
  offset: number;
  endOffset: number;
  index: number;
  length: number;
  opcode: number;
  spec: InstructionSpec;
  operand?: number;
  /** Raw offset of the WIDE prefix byte when the encoding used one. */
  wide?: boolean;
}

function readU8(code: Uint8Array, at: number): number {
  return code[at];
}

function readU16BE(code: Uint8Array, at: number): number {
  return (code[at] << 8) | code[at + 1];
}

function readU32BE(code: Uint8Array, at: number): number {
  // >>> 0 keeps the result an unsigned 32-bit integer.
  return (((code[at] << 24) | (code[at + 1] << 16) | (code[at + 2] << 8) | code[at + 3]) >>> 0);
}

/**
 * Decode the *entire* code array with a strict linear sweep. The sweep walks
 * every byte — code is never skipped because it is statically unreachable —
 * so a malicious program cannot hide a bad opcode inside an "unreachable"
 * region (e.g. after HALT).
 *
 * Fails (throws BytecodeVerifyError) on:
 *   - empty code
 *   - any unknown opcode
 *   - any operand that runs past the end of the code array (truncation)
 *   - a WIDE prefix at the end, doubled, or not followed by PUSH2
 *     (non-canonical encoding)
 */
export function decode(code: Uint8Array): DecodedInstruction[] {
  if (code.length === 0) {
    fail('EMPTY_CODE', 0, 0, 'code array is empty');
  }

  const out: DecodedInstruction[] = [];
  let at = 0;

  while (at < code.length) {
    const start = at;
    const index = out.length;
    const opcode = code[at++];

    let wide = false;
    if (opcode === OP.WIDE) {
      // Prefix at the very end: nothing follows it.
      if (at >= code.length) {
        fail('TRUNCATED_OPERAND', start, index, 'WIDE prefix at end of code is truncated');
      }
      const next = code[at];
      if (next === OP.WIDE) {
        fail('NON_CANONICAL_ENCODING', at, index, 'WIDE prefix may not be doubled');
      }
      if (next !== OP.PUSH2) {
        fail(
          'NON_CANONICAL_ENCODING',
          at,
          index,
          `WIDE prefix may only precede PUSH2, found opcode 0x${next.toString(16)}`,
        );
      }
      wide = true;
      at++; // consume the PUSH2 opcode byte
    }

    // The opcode that identifies the instruction's operand shape.
    const shapeOpcode = wide ? OP.PUSH2 : opcode;
    const baseSpec = INSTRUCTIONS[shapeOpcode];
    if (baseSpec === undefined) {
      fail('UNKNOWN_OPCODE', wide ? at - 1 : start, index, `unknown opcode 0x${shapeOpcode.toString(16)}`);
    }

    const spec = wide ? WIDE_PUSH4 : baseSpec;
    const operandSize = spec.operandSize;
    if (operandSize > 0 && at + operandSize > code.length) {
      fail(
        'TRUNCATED_OPERAND',
        start,
        index,
        `${spec.mnemonic} needs ${operandSize} operand byte(s), code ends at ${code.length}`,
      );
    }

    let operand: number | undefined;
    switch (spec.operand) {
      case 'imm8u':
        operand = readU8(code, at);
        break;
      case 'imm16u':
        operand = readU16BE(code, at);
        break;
      case 'imm32u':
        operand = readU32BE(code, at);
        break;
      case 'rel8':
        operand = signExtend(readU8(code, at), 8);
        break;
      case 'rel16':
        operand = signExtend(readU16BE(code, at), 16);
        break;
      case 'rel32':
        // Signed 32-bit. Bytes whose top bit is set become negative.
        operand = readU32BE(code, at) | 0;
        break;
      case 'none':
        break;
    }
    at += operandSize > 0 ? operandSize : 0;

    out.push({
      offset: start,
      endOffset: at,
      index,
      length: at - start,
      opcode: shapeOpcode,
      spec,
      operand,
      ...(wide ? { wide: true } : {}),
    });
  }

  return out;
}

/**
 * Build the instruction-boundary table: the set of raw byte offsets at
 * which a legal instruction starts. Operand bytes and end-of-code are not
 * in the table, so jumping "into the middle" of an instruction is rejected.
 */
export function boundaries(code: Uint8Array): Set<number> {
  return new Set(decode(code).map((ins) => ins.offset));
}

/** Map a raw byte offset to the dense instruction index starting there. */
export function boundaryIndexTable(
  instrs: DecodedInstruction[],
): Map<number, number> {
  return new Map(instrs.map((ins) => [ins.offset, ins.index]));
}

export function signExtend(value: number, bits: 8 | 16): number {
  const signBit = 1 << (bits - 1);
  return (value ^ signBit) - signBit;
}
