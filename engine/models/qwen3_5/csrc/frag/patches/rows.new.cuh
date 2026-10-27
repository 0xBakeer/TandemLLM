    const int M = y.size(0), Kin = xh.size(2), N = t.size(1) * 16;
    TORCH_CHECK(xh.size(1) == (M + 15) / 16 * 16 && cfg != 0, "fragment-major xh [G, M to 16, Kin], a register cfg");