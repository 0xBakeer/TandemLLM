export declare const BUDGET: number;
export declare const gzipSize: (bytes: Uint8Array) => number;
export declare const distSizes: (dir: string) => { files: { path: string; gzip: number }[]; total: number };
