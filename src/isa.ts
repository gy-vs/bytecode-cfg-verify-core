/**
 * Instruction set used by the verifier. All immediates are big-endian.
 *
 *   opcode  name      operand
 *   0x00    NOP       (none)
 *   0x01    PUSH1     1-byte immediate
 *   0x02    PUSH2     2-byte immediate
 *   0x03    JUMP8     1-byte signed relative displacement
 *   0x04    JUMP16    2-byte signed relative displacement
 *   0x05    JUMP32    4-byte signed relative displacement
 *   0x06    HALT      (none, terminator: no fall-through)
 *   0x07    WIDE      prefix — only valid immediately before PUSH2,
 *                     which it widens to PUSH4 (4-byte immediate).
 *
 * Relative jump displacements are relative to the *start* of the jump
 * instruction. A zero displacement therefore lands on the jump itself.
 */

export const OP = {
  NOP: 0x00,
  PUSH1: 0x01,
  PUSH2: 0x02,
  JUMP8: 0x03,
  JUMP16: 0x04,
  JUMP32: 0x05,
  HALT: 0x06,
  WIDE: 0x07,
} as const;

export type OperandType =
  | 'none'
  | 'imm8u'
  | 'imm16u'
  | 'rel8'
  | 'rel16'
  | 'rel32'
  | 'imm32u';

export interface InstructionSpec {
  opcode: number;
  mnemonic: string;
  /** Operand size in bytes (0 when none). -1 marks the WIDE prefix. */
  operandSize: number;
  operand: OperandType;
  terminal: boolean;
  /** True for the relative-jump family. */
  jump: boolean;
}

export const INSTRUCTIONS: Readonly<Record<number, InstructionSpec>> = {
  [OP.NOP]: { opcode: OP.NOP, mnemonic: 'NOP', operandSize: 0, operand: 'none', terminal: false, jump: false },
  [OP.PUSH1]: { opcode: OP.PUSH1, mnemonic: 'PUSH1', operandSize: 1, operand: 'imm8u', terminal: false, jump: false },
  [OP.PUSH2]: { opcode: OP.PUSH2, mnemonic: 'PUSH2', operandSize: 2, operand: 'imm16u', terminal: false, jump: false },
  [OP.JUMP8]: { opcode: OP.JUMP8, mnemonic: 'JUMP8', operandSize: 1, operand: 'rel8', terminal: true, jump: true },
  [OP.JUMP16]: { opcode: OP.JUMP16, mnemonic: 'JUMP16', operandSize: 2, operand: 'rel16', terminal: true, jump: true },
  [OP.JUMP32]: { opcode: OP.JUMP32, mnemonic: 'JUMP32', operandSize: 4, operand: 'rel32', terminal: true, jump: true },
  [OP.HALT]: { opcode: OP.HALT, mnemonic: 'HALT', operandSize: 0, operand: 'none', terminal: true, jump: false },
  // WIDE alone is not a real instruction; it must prefix a PUSH2.
  [OP.WIDE]: { opcode: OP.WIDE, mnemonic: 'WIDE', operandSize: -1, operand: 'none', terminal: false, jump: false },
};

/** PUSH2 widened by WIDE -> PUSH4 (4-byte unsigned immediate). */
export const WIDE_PUSH4: InstructionSpec = {
  opcode: OP.PUSH2,
  mnemonic: 'PUSH4',
  operandSize: 4,
  operand: 'imm32u',
  terminal: false,
  jump: false,
};

export function isJumpSpec(spec: InstructionSpec): boolean {
  return spec.jump;
}
