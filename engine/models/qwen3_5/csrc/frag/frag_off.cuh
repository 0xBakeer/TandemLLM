// SPD-201: the rotated input in the mma's A-fragment order: a member's rows in groups of 16 (zero rows to a
// multiple of 16), a group's K tiles in order, a tile's 32 lanes, a lane's a0..a3 of m16n8k16, so a lane's
// operands are one 16-byte load. Only where the values sit moves.
__device__ __forceinline__ size_t frag_off(int m, int k, int Kin) {
    const int rg = m >> 4, r_hi = (m >> 3) & 1, r_lo = m & 7;
    const int kt = k >> 4, k_hi = (k >> 3) & 1, t = (k >> 1) & 3, pr = k & 1;
    return ((((((size_t)rg * (Kin >> 4) + kt) * 8 + r_lo) * 4 + t) * 2 + k_hi) * 2 + r_hi) * 2 + pr;
}
