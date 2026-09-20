export type Op={offset:number;opcode:number,operand?:number};
export function decode(code:Uint8Array){const out:Op[]=[];for(let at=0;at<code.length;){const opcode=code[at++];if(opcode===1){if(at>=code.length)throw new Error('truncated');out.push({offset:at-1,opcode,operand:code[at++]})}else out.push({offset:at-1,opcode})}return out}
export function boundaries(code:Uint8Array){return new Set(decode(code).map(op=>op.offset))}
