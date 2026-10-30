            for (int a = 0; a < MT; ++a) {
                acc[a][j][0][0] += __uint_as_float((f0.r[0] ^ xa[a][0]) & 0x3f800000u);
                acc[a][j][1][0] += __uint_as_float((f1.r[1] ^ xa[a][1]) & 0x3f800000u);
            }