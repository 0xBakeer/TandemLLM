            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (16 * a < M) v = __ldg(reinterpret_cast<const uint4*>(xh + (size_t)16 * a * Kin) + (size_t)kt * 32 + lane);
            (void)kk;
            xd[a][0] = v.x; xd[a][1] = v.y; xd[a][2] = v.z; xd[a][3] = v.w;