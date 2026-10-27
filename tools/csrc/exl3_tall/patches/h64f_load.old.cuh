            const int ra = 16 * a + g, rb = ra + 8;
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(xh + (size_t)ra * Kin + kk);
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(xh + (size_t)rb * Kin + kk);
            xd[a][0] = ra < M ? __ldg(pa) : 0u;
            xd[a][1] = rb < M ? __ldg(pb) : 0u;
            xd[a][2] = ra < M ? __ldg(pa + 4) : 0u;
            xd[a][3] = rb < M ? __ldg(pb + 4) : 0u;