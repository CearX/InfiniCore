1.pos_ids: 形状 [seq_len, 3], 每列分别是t,h,w的pos_ids, strides = [3, 1](即pos_ids连续)
2.rope_section:
比如 dh = 128, rope_section = [16, 24, 24], 对应t, h, w; 即对于 x 和 x + (dh/2) 的一对, dh/2 = 64, 分别来自 t的前16， h的中间24, w的后24;具体以.cuh计算为准;
