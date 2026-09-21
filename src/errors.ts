/**
 * Verifier failure codes. The verifier fails closed: every structural
 * problem — unknown opcode, truncated/non-canonical encoding, bad jump
 * target, malformed exception range — aborts with one of these codes.
 */
export type VerifyErrorCode =
  // ---- decode phase (whole-code linear sweep) ----
  | 'EMPTY_CODE'
  | 'UNKNOWN_OPCODE'
  | 'TRUNCATED_OPERAND'
  | 'NON_CANONICAL_ENCODING'
  // ---- jump resolution phase ----
  | 'JUMP_OVERFLOW'
  | 'JUMP_OUT_OF_RANGE'
  | 'JUMP_INTO_OPERAND'
  // ---- straight-line fall-through ----
  | 'FALLTHROUGH_PAST_END'
  // ---- exception table phase ----
  | 'EXCEPTION_RANGE_OOB'
  | 'EXCEPTION_RANGE_BAD_BOUNDARY'
  | 'EXCEPTION_RANGE_EMPTY'
  | 'EXCEPTION_RANGE_BAD_ORDER'
  | 'EXCEPTION_RANGE_OVERLAP';

/** Context carried with every failure: raw byte offsets + instruction index. */
export interface VerifyErrorDetails {
  code: VerifyErrorCode;
  /** Byte offset in the raw code array where the problem was detected. */
  offset: number;
  /** Instruction index (when known). Index n = nth decoded instruction. */
  index: number | null;
  /** Optional secondary offset (e.g. a computed jump target). */
  targetOffset?: number;
  message: string;
}

export class BytecodeVerifyError extends Error {
  readonly code: VerifyErrorCode;
  readonly offset: number;
  readonly index: number | null;
  readonly targetOffset?: number;

  constructor(details: VerifyErrorDetails) {
    super(details.message);
    this.name = 'BytecodeVerifyError';
    this.code = details.code;
    this.offset = details.offset;
    this.index = details.index;
    if (details.targetOffset !== undefined) this.targetOffset = details.targetOffset;
  }
}

export function fail(
  code: VerifyErrorCode,
  offset: number,
  index: number | null,
  message: string,
  targetOffset?: number,
): never {
  throw new BytecodeVerifyError({ code, offset, index, message, targetOffset });
}
