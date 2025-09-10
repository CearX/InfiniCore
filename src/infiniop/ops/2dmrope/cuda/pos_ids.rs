/// 构造 pos_ids 表
#[allow(dead_code)]
pub(crate) fn build_pos_ids(h: usize, w: usize, d_patch: usize) -> Vec<u32> {
    let hp = h / d_patch;
    let wp = w / d_patch;
    let mut pos = vec![0; hp * wp * 2];

    let mut ptr = 0;
    for y in (0..hp).step_by(2) {
        for x in (0..wp).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    pos[ptr * 2] = (y + dy) as u32;
                    pos[ptr * 2 + 1] = (x + dx) as u32;
                    ptr += 1;
                }
            }
        }
    }

    pos
}
