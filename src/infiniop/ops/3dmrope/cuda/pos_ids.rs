/// # 构造 3D RoPE 的 pos_ids 表
///
/// 对于视觉和文本嵌入序列，我们为视觉部分计算 3D 旋转位置嵌入，为文本部分计算 1D 旋转位置嵌入。
///
/// ## 示例
///
/// 假设我们有一个视频输入，包含 3 个时序 patch，2 个高度 patch 和 2 个宽度 patch。
///
/// **输入序列：**
/// - `input_ids` = `[T T T T V V V V V V V V V V V V T T T T T]`
/// - 其中 `V` 代表视觉 patch，`T` 代表文本 patch
///
/// **文本部分 position_ids（图像前）：**
/// - `text temporal position_ids`: `[0, 1, 2, 3]`
/// - `text height position_ids`: `[0, 1, 2, 3]`
/// - `text width position_ids`: `[0, 1, 2, 3]`
///
/// **视觉部分 position_ids：**
/// - `vision temporal position_ids`: `[4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]`
/// - `vision height position_ids`: `[4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5]`
/// - `vision width position_ids`: `[4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]`
///
/// **文本部分 position_ids（图像后）：**
/// - `text temporal position_ids`: `[7, 8, 9, 10, 11]`
/// - `text height position_ids`: `[7, 8, 9, 10, 11]`
/// - `text width position_ids`: `[7, 8, 9, 10, 11]`
///
/// **计算规则：**
/// - 图像起始 `position_ids` 计算为图像前文本最大 `position_ids` 加 1
/// - 图像后文本起始 `position_ids` 计算为最大视觉 `position_ids` 加 1
#[allow(dead_code)]
pub fn build_3d_pos_ids(
    t: usize,
    h: usize,
    w: usize,
    d_patch: usize,
    pre_text_len: usize,
    post_text_len: usize,
) -> Vec<u32> {
    let spatial_merge_size = 2;
    let t_len = t;
    let h_len = h / d_patch / spatial_merge_size;
    let w_len = w / d_patch / spatial_merge_size;
    let vision_len = t_len * h_len * w_len;
    let total_len = pre_text_len + vision_len + post_text_len;

    let mut pos = vec![0u32; total_len * 3];
    let mut idx = 0;

    // 图像前文本
    for i in 0..pre_text_len as u32 {
        pos[idx * 3] = i;
        pos[idx * 3 + 1] = i;
        pos[idx * 3 + 2] = i;
        idx += 1;
    }

    // 图像
    let img_start_pos = pre_text_len as u32;
    for t in 0..t_len as u32 {
        for h in 0..h_len as u32 {
            for w in 0..w_len as u32 {
                let t_pos = img_start_pos + t;
                let h_pos = img_start_pos + h;
                let w_pos = img_start_pos + w;
                pos[idx * 3] = t_pos;
                pos[idx * 3 + 1] = h_pos;
                pos[idx * 3 + 2] = w_pos;
                idx += 1;
            }
        }
    }

    // 图像后文本
    let t_max_pos = img_start_pos + t_len as u32 - 1;
    let h_max_pos = img_start_pos + h_len as u32 - 1;
    let w_max_pos = img_start_pos + w_len as u32 - 1;
    let image_max_pos = max(t_max_pos, max(h_max_pos, w_max_pos));
    let text_start_pos = image_max_pos + 1;
    for i in 0..post_text_len as u32 {
        let pos_val = text_start_pos + i;
        pos[idx * 3] = pos_val;
        pos[idx * 3 + 1] = pos_val;
        pos[idx * 3 + 2] = pos_val;
        idx += 1;
    }

    assert_eq!(idx, total_len);
    pos
}
