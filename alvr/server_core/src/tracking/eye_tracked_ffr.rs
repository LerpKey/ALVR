// Dynamic Foveated Rendering (DFR) implementation
// Converts eye tracking data from FaceData to foveation center shift parameters
// Single-eye logic using combined_gaze for simplicity

use alvr_common::glam::{Vec2, Vec3};
use alvr_packets::FaceData;
use std::sync::RwLock;
use std::time::{Duration, SystemTime, Instant};
use std::collections::BTreeMap;

// DFR shift parameters for C++ interop
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DFRShiftParams {
    pub shift_x: f32,           // Combined horizontal shift [-1, 1]
    pub shift_y: f32,           // Combined vertical shift [-1, 1]
    pub is_eye_tracked: bool,   // Whether valid eye tracking data is available
}

impl Default for DFRShiftParams {
    fn default() -> Self {
        Self {
            shift_x: 0.0,
            shift_y: 0.0,
            is_eye_tracked: false,
        }
    }
}

// Global state for DFR
static GLOBAL_DFR_SHIFT: RwLock<Option<DFRShiftParams>> = RwLock::new(None);
static GLOBAL_LAST_UPDATE: RwLock<Option<SystemTime>> = RwLock::new(None);

// Synchronized DFR state for pipeline consistency
// Ensures server encoding and client inverse-FFR use identical shift values
static SYNCHRONIZED_DFR_SHIFT: RwLock<Option<DFRShiftParams>> = RwLock::new(None);
static SYNCHRONIZED_LAST_UPDATE: RwLock<Option<SystemTime>> = RwLock::new(None);
static SYNCHRONIZED_SEQUENCE_ID: RwLock<u64> = RwLock::new(0);

// Frame-Perfect数据绑定机制
// 保存FR shader实际使用的eyeshift数据，确保与VideoPacketHeader完全对应
#[derive(Debug, Clone, Copy)]
struct RenderedShiftData {
    shift: DFRShiftParams,
    sequence_id: u64,
    timestamp: Instant,
}

static LAST_FR_RENDERED_SHIFT: std::sync::LazyLock<RwLock<Option<RenderedShiftData>>> =
    std::sync::LazyLock::new(|| RwLock::new(None));

// 🎯 统一时间戳生成器：Frame-Perfect时间戳绑定机制
// 确保DFR编码和shift数据使用完全相同的SteamVR targetTimestampNs
static TIMESTAMP_SHIFT_CACHE: std::sync::LazyLock<RwLock<BTreeMap<u64, DFRShiftParams>>> =
    std::sync::LazyLock::new(|| RwLock::new(BTreeMap::new()));

// DFR configuration constants
const EYE_TRACKING_TIMEOUT: Duration = Duration::from_millis(200);
const SHIFT_GAIN: f32 = 0.8;                    // Scale factor for eye shift
const MAX_SHIFT: f32 = 0.4;                     // Maximum allowed shift

// PICO hardware coordinate ranges - critical for accurate mapping
const PICO_MAX_X: f32 = 0.6; // x gaze_vector range: [-0.6, 0.6]
const PICO_MAX_Y: f32 = 0.4; // y gaze_vector range: [-0.4, 0.4]

pub struct DynamicFoveatedRenderer {
    // No additional fields needed - PICO gaze vectors are directly usable
}

impl Default for DynamicFoveatedRenderer {
    fn default() -> Self {
        Self {}
    }
}

impl DynamicFoveatedRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update DFR parameters from face tracking data with synchronized pipeline
    pub fn update_from_face_data(&self, face_data: &FaceData) {
        // Calculate DFR data using combined_gaze (single-eye approach)
        let shift_params = self.calculate_dfr_shift_combined(face_data);

        // Always update the timestamp to prevent timeout, even if eye tracking data is invalid
        let now = SystemTime::now();
        if let Ok(mut time_guard) = GLOBAL_LAST_UPDATE.write() {
            *time_guard = Some(now);
        }

        // Update global state
        if let Ok(mut shift_guard) = GLOBAL_DFR_SHIFT.write() {
            if shift_params.is_eye_tracked {
                *shift_guard = Some(shift_params);
            } else {
                *shift_guard = None; // Clear DFR data to force FFR
            }
        }

        // Update synchronized shift for pipeline consistency
        let needs_sync_update = self.should_update_synchronized_shift(&shift_params);

        if needs_sync_update {
            if let Ok(mut sync_shift_guard) = SYNCHRONIZED_DFR_SHIFT.write() {
                if let Ok(mut sync_time_guard) = SYNCHRONIZED_LAST_UPDATE.write() {
                    if let Ok(mut seq_id_guard) = SYNCHRONIZED_SEQUENCE_ID.write() {
                        *seq_id_guard += 1;
                        let sequence_id = *seq_id_guard;

                        *sync_shift_guard = if shift_params.is_eye_tracked {
                            Some(shift_params)
                        } else {
                            None // FFR mode
                        };
                        *sync_time_guard = Some(now);

                        if shift_params.is_eye_tracked {
                            alvr_common::debug!("Pipeline Sync: Updated synchronized shift [seq={}] - DFR:({:.3},{:.3})",
                                              sequence_id, shift_params.shift_x, shift_params.shift_y);
                        } else {
                            alvr_common::debug!("Pipeline Sync: No valid eye data, returning FFR defaults [seq={}]", sequence_id);
                        }
                    }
                }
            }
        }

        // Simplified logging
        static mut TOTAL_FRAME_COUNT: u32 = 0;
        unsafe {
            TOTAL_FRAME_COUNT += 1;
            // Log summary every 60 frames (~1 second at 60fps) to avoid spam
            if TOTAL_FRAME_COUNT % 60 == 0 {
                if shift_params.is_eye_tracked {
                    alvr_common::info!("DFR Active: Frame {} - Using combined eye tracking", TOTAL_FRAME_COUNT);
                } else {
                    alvr_common::info!("FFR Active: Frame {} - No valid eye data", TOTAL_FRAME_COUNT);
                }
            }
        }
    }

    /// Determine if synchronized shift should be updated
    fn should_update_synchronized_shift(&self, new_shift: &DFRShiftParams) -> bool {
        // Check if we have existing synchronized data
        if let Ok(sync_guard) = SYNCHRONIZED_DFR_SHIFT.read() {
            if let Some(current_sync) = *sync_guard {
                // Compare eye tracking availability first
                if current_sync.is_eye_tracked != new_shift.is_eye_tracked {
                    return true; // Mode change: DFR ↔ FFR
                }

                if new_shift.is_eye_tracked {
                    // Compare shift values with threshold
                    const SHIFT_THRESHOLD: f32 = 0.02; // 2% threshold to avoid micro-movements

                    let x_diff = (current_sync.shift_x - new_shift.shift_x).abs();
                    let y_diff = (current_sync.shift_y - new_shift.shift_y).abs();

                    return x_diff > SHIFT_THRESHOLD || y_diff > SHIFT_THRESHOLD;
                }
            }
        }

        // No existing data or first update
        true
    }

    /// Calculate DFR shift using combined gaze data (single-eye approach)
    fn calculate_dfr_shift_combined(&self, face_data: &FaceData) -> DFRShiftParams {
        if let Some(pico_data) = &face_data.pico_eye_tracking_data {
            // Check if combined gaze data is valid
            let combined_valid = alvr_packets::eye_pose_status_pico::is_gaze_vector_valid(pico_data.combined_eye_pose_status);

            if combined_valid {
                let combined_gaze_direction = Vec3::new(
                    pico_data.combined_eye_gaze_vector[0],
                    pico_data.combined_eye_gaze_vector[1],
                    pico_data.combined_eye_gaze_vector[2],
                );

                if self.validate_gaze_direction(combined_gaze_direction) {
                    let screen_shift = self.gaze_to_screen_shift(combined_gaze_direction);

                    return DFRShiftParams {
                        shift_x: screen_shift.x,
                        shift_y: screen_shift.y,
                        is_eye_tracked: true,
                    };
                }
            }
        }

        // No valid data - return default (FFR mode)
        DFRShiftParams::default()
    }

    /// Validate that gaze direction is reasonable
    fn validate_gaze_direction(&self, gaze_direction: Vec3) -> bool {
        let length = gaze_direction.length();

        // Check for valid normalized vector (should be close to 1.0)
        if length < 0.5 || length > 2.0 {
            return false;
        }

        // Check for reasonable gaze ranges (PICO specific)
        if gaze_direction.x.abs() > PICO_MAX_X * 1.2 || gaze_direction.y.abs() > PICO_MAX_Y * 1.2 {
            return false;
        }

        true
    }

    /// Convert gaze direction to screen shift coordinates
    fn gaze_to_screen_shift(&self, gaze_direction: Vec3) -> Vec2 {
        // Direct mapping from PICO hardware ranges to screen coordinates
        // This ensures full eye movement range maps to full screen
        let screen_x = (gaze_direction.x / PICO_MAX_X).clamp(-1.0, 1.0);
        let screen_y = (gaze_direction.y / PICO_MAX_Y).clamp(-1.0, 1.0);

        // Apply gain and max shift limiting
        let shifted_x = (screen_x * SHIFT_GAIN).clamp(-MAX_SHIFT, MAX_SHIFT);
        let shifted_y = (screen_y * SHIFT_GAIN).clamp(-MAX_SHIFT, MAX_SHIFT);

        Vec2::new(shifted_x, shifted_y)
    }
}

/// Public interface for C++ integration (original single-eye function)
/// 🎯 关键修改：保存FR shader实际使用的数据以确保Frame-Perfect同步
#[no_mangle]
pub extern "C" fn get_eye_tracked_ffr_shift() -> DFRShiftParams {
    let shift_data = if let Ok(guard) = GLOBAL_DFR_SHIFT.read() {
        guard.unwrap_or_default()
    } else {
        DFRShiftParams::default()
    };

    // 🎯 关键：保存FR shader实际使用的数据
    // 这确保VideoPacketHeader中传输的shift与FR shader使用的shift完全相同
    if let Ok(mut rendered_shift_guard) = LAST_FR_RENDERED_SHIFT.write() {
        let current_seq_id = get_synchronized_sequence_id();
        *rendered_shift_guard = Some(RenderedShiftData {
            shift: shift_data,
            sequence_id: current_seq_id,
            timestamp: Instant::now(),
        });

        // 调试日志：验证数据捕获
        if shift_data.is_eye_tracked {
            alvr_common::debug!("🎯 FR_CAPTURE: Captured eyeshift [seq={}] DFR:({:.3},{:.3}) for frame rendering",
                              current_seq_id, shift_data.shift_x, shift_data.shift_y);
        } else {
            alvr_common::debug!("🎯 FR_CAPTURE: Captured FFR default [seq={}] for frame rendering", current_seq_id);
        }
    }

    shift_data
}

/// Get synchronized DFR shift for pipeline consistency
pub fn get_synchronized_dfr_shift() -> Option<DFRShiftParams> {
    if let Ok(guard) = SYNCHRONIZED_DFR_SHIFT.read() {
        *guard
    } else {
        None
    }
}

/// Get synchronized sequence ID for tracking pipeline sync
pub fn get_synchronized_sequence_id() -> u64 {
    if let Ok(guard) = SYNCHRONIZED_SEQUENCE_ID.read() {
        *guard
    } else {
        0
    }
}

/// Check if eye tracking data is recent enough to be valid
pub fn is_eye_tracking_active() -> bool {
    if let Ok(guard) = GLOBAL_LAST_UPDATE.read() {
        if let Some(last_update) = *guard {
            return SystemTime::now().duration_since(last_update)
                .map(|duration| duration < EYE_TRACKING_TIMEOUT)
                .unwrap_or(false);
        }
    }
    false
}

/// 🎯 获取FR shader实际使用的eyeshift数据
/// 用于VideoPacketHeader的Frame-Perfect绑定
pub fn get_last_fr_rendered_shift() -> Option<DFRShiftParams> {
    if let Ok(guard) = LAST_FR_RENDERED_SHIFT.read() {
        guard.as_ref().map(|data| data.shift)
    } else {
        None
    }
}

/// 🎯 获取FR shader使用的序列号
/// 确保时间一致性
pub fn get_last_fr_rendered_sequence_id() -> Option<u64> {
    if let Ok(guard) = LAST_FR_RENDERED_SHIFT.read() {
        guard.as_ref().map(|data| data.sequence_id)
    } else {
        None
    }
}

/// 验证FR rendered数据是否足够新（用于边缘情况处理）
pub fn is_last_fr_rendered_shift_recent() -> bool {
    if let Ok(guard) = LAST_FR_RENDERED_SHIFT.read() {
        if let Some(data) = guard.as_ref() {
            // 检查数据是否在合理的时间窗口内（比如1秒）
            return data.timestamp.elapsed() < Duration::from_millis(1000);
        }
    }
    false
}

/// 🎯 NEW: Frame-Perfect timestamp binding function
/// 使用SteamVR的精确targetTimestampNs将shift数据与时间戳绑定
/// 确保编码shift === 解码shift，彻底消除抖动
#[no_mangle]
pub extern "C" fn get_eye_tracked_ffr_shift_with_timestamp(target_timestamp_ns: u64) -> DFRShiftParams {
    // Edge case: 如果时间戳为0或无效，回退到当前shift
    if target_timestamp_ns == 0 {
        alvr_common::warn!("🎯 TIMESTAMP_BIND: Invalid timestamp 0, using fallback");
        return get_eye_tracked_ffr_shift(); // 使用旧接口作为fallback
    }

    // 🎯 使用精确时间戳绑定
    if let Ok(mut cache_guard) = TIMESTAMP_SHIFT_CACHE.write() {
        // 获取当前最新的shift数据
        let current_shift = if let Ok(guard) = GLOBAL_DFR_SHIFT.read() {
            guard.unwrap_or_default()
        } else {
            DFRShiftParams::default()
        };

        // 将shift数据与精确的SteamVR时间戳绑定
        cache_guard.insert(target_timestamp_ns, current_shift);

        // 清理过期数据 (保留最近100帧，防止内存无限增长)
        if cache_guard.len() > 100 {
            let keys_to_remove: Vec<u64> = cache_guard.keys().take(cache_guard.len() - 100).cloned().collect();
            for key in keys_to_remove {
                cache_guard.remove(&key);
            }
        }

        // 调试日志：验证数据绑定
        if current_shift.is_eye_tracked {
            alvr_common::debug!("🎯 TIMESTAMP_BIND: Cached shift for timestamp {} - DFR:({:.3},{:.3})",
                              target_timestamp_ns, current_shift.shift_x, current_shift.shift_y);
        } else {
            alvr_common::debug!("🎯 TIMESTAMP_BIND: Cached FFR default for timestamp {}", target_timestamp_ns);
        }

        return current_shift;
    }

    // 写锁失败：返回默认值
    alvr_common::error!("🎯 TIMESTAMP_BIND: Failed to acquire cache write lock");
    DFRShiftParams::default()
}

/// 🎯 用于VideoPacketHeader的Frame-Perfect检索
/// 基于精确的targetTimestampNs检索对应的shift数据
pub fn get_dfr_shift_for_timestamp(target_timestamp_ns: u64) -> Option<DFRShiftParams> {
    if let Ok(cache_guard) = TIMESTAMP_SHIFT_CACHE.read() {
        if let Some(cached_shift) = cache_guard.get(&target_timestamp_ns) {
            alvr_common::debug!("🎯 TIMESTAMP_RETRIEVE: Found cached shift for timestamp {} - DFR:({:.3},{:.3})",
                              target_timestamp_ns, cached_shift.shift_x, cached_shift.shift_y);
            return Some(*cached_shift);
        } else {
            alvr_common::warn!("🎯 TIMESTAMP_RETRIEVE: Timestamp {} not found in cache (cache size: {})",
                             target_timestamp_ns, cache_guard.len());
        }
    } else {
        alvr_common::error!("🎯 TIMESTAMP_RETRIEVE: Failed to acquire cache read lock");
    }

    None
}

/// 检查时间戳缓存状态（用于调试）
pub fn get_timestamp_cache_info() -> (usize, Option<u64>, Option<u64>) {
    if let Ok(cache_guard) = TIMESTAMP_SHIFT_CACHE.read() {
        let size = cache_guard.len();
        let oldest = cache_guard.keys().next().cloned();
        let newest = cache_guard.keys().next_back().cloned();
        (size, oldest, newest)
    } else {
        (0, None, None)
    }
}

/// Initialize DFR renderer
pub fn initialize_dfr() -> DynamicFoveatedRenderer {
    DynamicFoveatedRenderer::new()
}