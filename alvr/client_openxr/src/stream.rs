use crate::{
    graphics::{self, ProjectionLayerAlphaConfig, ProjectionLayerBuilder},
    interaction::{self, InteractionContext, InteractionSourcesConfig},
};
use alvr_client_core::{
    video_decoder::{self, VideoDecoderConfig, VideoDecoderSource},
    ClientCoreContext,
};
use alvr_common::{
    anyhow::Result,
    error,
    glam::{Quat, UVec2, Vec2},
    log,
    parking_lot::RwLock,
    Pose, RelaxedAtomic, HAND_LEFT_ID, HAND_RIGHT_ID, HEAD_ID,
};
use alvr_graphics::{
    compute_target_view_resolution, GraphicsContext, StreamRenderer, StreamViewParams,
};
use alvr_packets::{
    eye_pose_status_pico, EyeTrackingInputStatus, FaceData, RealTimeConfig, StreamConfig,
    ViewParams,
};
use alvr_session::{
    ClientsideFoveationConfig, ClientsideFoveationMode, ClientsidePostProcessingConfig, CodecType,
    FoveatedEncodingConfig, MediacodecProperty, PassthroughMode, UpscalingConfig,
};
use alvr_system_info::Platform;
use openxr as xr;
use std::{
    ptr,
    rc::Rc,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const DECODER_MAX_TIMEOUT_MULTIPLIER: f32 = 0.8;

pub struct ParsedStreamConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub use_full_range: bool,
    pub encoding_gamma: f32,
    pub enable_hdr: bool,
    pub passthrough: Option<PassthroughMode>,
    pub foveated_encoding_config: Option<FoveatedEncodingConfig>,
    pub clientside_foveation_config: Option<ClientsideFoveationConfig>,
    pub clientside_post_processing: Option<ClientsidePostProcessingConfig>,
    pub upscaling: Option<UpscalingConfig>,
    pub dynamic_foveated_center: Option<alvr_packets::DynamicFoveatedCenter>,
    pub force_software_decoder: bool,
    pub max_buffering_frames: f32,
    pub buffering_history_weight: f32,
    pub decoder_options: Vec<(String, MediacodecProperty)>,
    pub interaction_sources: InteractionSourcesConfig,
}

impl ParsedStreamConfig {
    pub fn new(config: &StreamConfig) -> Self {
        Self {
            view_resolution: config.negotiated_config.view_resolution,
            refresh_rate_hint: config.negotiated_config.refresh_rate_hint,
            use_full_range: config.negotiated_config.use_full_range,
            encoding_gamma: config.negotiated_config.encoding_gamma,
            enable_hdr: config.negotiated_config.enable_hdr,
            passthrough: config.settings.video.passthrough.as_option().cloned(),
            foveated_encoding_config: config
                .negotiated_config
                .enable_foveated_encoding
                .then(|| config.settings.video.foveated_encoding.as_option().cloned())
                .flatten(),
            clientside_foveation_config: config
                .settings
                .video
                .clientside_foveation
                .as_option()
                .cloned(),
            clientside_post_processing: config
                .settings
                .video
                .clientside_post_processing
                .as_option()
                .cloned(),
            upscaling: config.settings.video.upscaling.as_option().cloned(),
            dynamic_foveated_center: None, // Will be set via real-time config
            force_software_decoder: config.settings.video.force_software_decoder,
            max_buffering_frames: config.settings.video.max_buffering_frames,
            buffering_history_weight: config.settings.video.buffering_history_weight,
            decoder_options: config.settings.video.mediacodec_extra_options.clone(),
            interaction_sources: InteractionSourcesConfig::new(config),
        }
    }
}

fn classify_eye_tracking_state(
    face_sources: &interaction::FaceSources,
    eye_gazes: &[Option<Pose>; 2],
    pico_eye_tracking: Option<&alvr_packets::EyeTrackingDataPICO>,
) -> EyeTrackingInputStatus {
    let has_any_tracker = face_sources.eye_tracker_fb.is_some()
        || face_sources.face_tracker_pico.is_some()
        || face_sources.eye_tracker_htc.is_some()
        || face_sources.combined_eyes_source.is_some();

    if !has_any_tracker {
        return EyeTrackingInputStatus::Unsupported;
    }

    let has_tracked_pose = eye_gazes.iter().any(|pose| pose.is_some());

    let pico_gaze_valid = pico_eye_tracking.map_or(false, |data| {
        eye_pose_status_pico::is_gaze_vector_valid(data.left_eye_pose_status)
            || eye_pose_status_pico::is_gaze_vector_valid(data.right_eye_pose_status)
            || eye_pose_status_pico::is_gaze_vector_valid(data.combined_eye_pose_status)
    });

    if has_tracked_pose || pico_gaze_valid {
        EyeTrackingInputStatus::Active
    } else {
        EyeTrackingInputStatus::Standby
    }
}

pub struct StreamContext {
    core_context: Arc<ClientCoreContext>,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_context: Arc<RwLock<InteractionContext>>,
    stage_reference_space: Arc<xr::Space>,
    view_reference_space: Arc<xr::Space>,
    swapchains: [xr::Swapchain<xr::OpenGlEs>; 2],
    last_good_view_params: [ViewParams; 2],
    // Frame-reuse fix: Cache last valid shift data for use when reusing old frames
    // Without this, frame reuse during network latency would use (0,0) shift
    // while the cached frame was encoded with non-zero shift, causing visual glitches
    last_good_shift: Option<alvr_packets::DynamicFoveatedCenter>,
    input_thread: Option<JoinHandle<()>>,
    input_thread_running: Arc<RelaxedAtomic>,
    config: ParsedStreamConfig,
    target_view_resolution: UVec2,
    renderer: StreamRenderer,
    decoder: Option<(VideoDecoderConfig, VideoDecoderSource)>,
}

impl StreamContext {
    pub fn new(
        core_ctx: Arc<ClientCoreContext>,
        xr_session: xr::Session<xr::OpenGlEs>,
        gfx_ctx: Rc<GraphicsContext>,
        interaction_ctx: Arc<RwLock<InteractionContext>>,
        platform: Platform,
        config: ParsedStreamConfig,
    ) -> StreamContext {
        // Client log connectivity test - should always appear in logs
        log::info!(target: alvr_common::CLIENT_IMPL_DBG_LABEL, "*Client日志联通* - StreamContext创建成功，FFR配置启用状态: {}",
            config.foveated_encoding_config.is_some());
        interaction_ctx
            .write()
            .select_sources(&config.interaction_sources);

        let xr_exts = xr_session.instance().exts();

        if xr_exts.fb_display_refresh_rate.is_some() {
            xr_session
                .request_display_refresh_rate(config.refresh_rate_hint)
                .unwrap();
        }

        let foveation_profile = if let Some(config) = &config.clientside_foveation_config {
            if xr_exts.fb_swapchain_update_state.is_some()
                && xr_exts.fb_foveation.is_some()
                && xr_exts.fb_foveation_configuration.is_some()
            {
                let level;
                let dynamic;
                match config.mode {
                    ClientsideFoveationMode::Static { level: lvl } => {
                        level = lvl;
                        dynamic = false;
                    }
                    ClientsideFoveationMode::Dynamic { max_level } => {
                        level = max_level;
                        dynamic = true;
                    }
                };

                xr_session
                    .create_foveation_profile(Some(xr::FoveationLevelProfile {
                        level: xr::FoveationLevelFB::from_raw(level as i32),
                        vertical_offset: config.vertical_offset_deg,
                        dynamic: xr::FoveationDynamicFB::from_raw(dynamic as i32),
                    }))
                    .ok()
            } else {
                None
            }
        } else {
            None
        };

        let target_view_resolution =
            compute_target_view_resolution(config.view_resolution, &config.upscaling);
        let format = graphics::swapchain_format(&gfx_ctx, &xr_session, config.enable_hdr);

        let swapchains = [
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
        ];

        let renderer = StreamRenderer::new(
            gfx_ctx,
            config.view_resolution,
            target_view_resolution,
            [
                swapchains[0]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
                swapchains[1]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
            ],
            format,
            config.foveated_encoding_config.clone(),
            platform != Platform::Lynx && !((platform.is_pico()) && config.enable_hdr),
            config.use_full_range && !config.enable_hdr, // TODO: figure out why HDR doesn't need the limited range hackfix in staging?
            config.encoding_gamma,
            config.upscaling.clone(),
        );

        // Log FFR/DFR configuration status
        if let Some(ref ffr_config) = config.foveated_encoding_config {
            log::info!(target: alvr_common::CLIENT_GFX_DBG_LABEL, "*Client日志联通* - FFR渲染器已创建，ENABLE_FFE应为true，FFR配置: center_size=({:.2},{:.2}), edge_ratio=({:.1},{:.1})",
                ffr_config.center_size_x, ffr_config.center_size_y,
                ffr_config.edge_ratio_x, ffr_config.edge_ratio_y);
        } else {
            log::warn!(target: alvr_common::CLIENT_GFX_DBG_LABEL, "*Client日志联通* - 无FFR配置，ENABLE_FFE应为false，inverse-FFR将不工作");
        }

        core_ctx.send_active_interaction_profile(
            *HAND_LEFT_ID,
            interaction_ctx.read().hands_interaction[0].controllers_profile_id,
        );
        core_ctx.send_active_interaction_profile(
            *HAND_RIGHT_ID,
            interaction_ctx.read().hands_interaction[1].controllers_profile_id,
        );

        let input_thread_running = Arc::new(RelaxedAtomic::new(false));

        let stage_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        let view_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        let mut this = StreamContext {
            core_context: core_ctx,
            xr_session,
            interaction_context: interaction_ctx,
            stage_reference_space,
            view_reference_space,
            swapchains,
            last_good_view_params: [ViewParams::default(); 2],
            last_good_shift: None,
            input_thread: None,
            input_thread_running,
            config,
            target_view_resolution,
            renderer,
            decoder: None,
        };

        this.update_reference_space();

        this
    }

    pub fn uses_passthrough(&self) -> bool {
        self.config.passthrough.is_some()
    }

    pub fn update_reference_space(&mut self) {
        self.input_thread_running.set(false);

        self.stage_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        self.view_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        self.core_context.send_playspace(
            self.xr_session
                .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
                .unwrap()
                .map(|a| Vec2::new(a.width, a.height)),
        );

        if let Some(running) = self.input_thread.take() {
            running.join().ok();
        }

        self.input_thread_running.set(true);

        self.input_thread = Some(thread::spawn({
            let core_ctx = Arc::clone(&self.core_context);
            let xr_session = self.xr_session.clone();
            let interaction_ctx = Arc::clone(&self.interaction_context);
            let stage_reference_space = Arc::clone(&self.stage_reference_space);
            let view_reference_space = Arc::clone(&self.view_reference_space);
            let refresh_rate = self.config.refresh_rate_hint;
            let running = Arc::clone(&self.input_thread_running);
            move || {
                stream_input_loop(
                    &core_ctx,
                    xr_session,
                    &interaction_ctx,
                    &stage_reference_space,
                    &view_reference_space,
                    refresh_rate,
                    running,
                )
            }
        }));
    }

    pub fn maybe_initialize_decoder(&mut self, codec: CodecType, config_nal: Vec<u8>) {
        let new_config = VideoDecoderConfig {
            codec,
            force_software_decoder: self.config.force_software_decoder,
            max_buffering_frames: self.config.max_buffering_frames,
            buffering_history_weight: self.config.buffering_history_weight,
            options: self.config.decoder_options.clone(),
            config_buffer: config_nal,
        };

        let maybe_config = if let Some((config, _)) = &self.decoder {
            (new_config != *config).then_some(new_config)
        } else {
            Some(new_config)
        };

        if let Some(config) = maybe_config {
            let (mut sink, source) = video_decoder::create_decoder(config.clone(), {
                let ctx = Arc::clone(&self.core_context);
                move |maybe_timestamp: Result<Duration>| match maybe_timestamp {
                    Ok(timestamp) => ctx.report_frame_decoded(timestamp),
                    Err(e) => ctx.report_fatal_decoder_error(&e.to_string()),
                }
            });
            self.decoder = Some((config, source));

            self.core_context.set_decoder_input_callback(Box::new(
                move |timestamp, buffer| -> bool { sink.push_nal(timestamp, buffer) },
            ));
        }
    }

    pub fn update_real_time_config(&mut self, config: &RealTimeConfig) {
        self.config.passthrough = config.passthrough.clone();
        self.config.clientside_post_processing = config.clientside_post_processing.clone();
        self.config.dynamic_foveated_center = config.dynamic_foveated_center.clone();
    }

    pub fn render(
        &mut self,
        frame_interval: Duration,
        vsync_time: Duration,
    ) -> (ProjectionLayerBuilder, Duration) {
        let frame_poll_deadline = Instant::now()
            + Duration::from_secs_f32(
                frame_interval.as_secs_f32() * DECODER_MAX_TIMEOUT_MULTIPLIER,
            );
        let mut frame_result = None;
        if let Some((_, source)) = &mut self.decoder {
            while frame_result.is_none() && Instant::now() < frame_poll_deadline {
                frame_result = source.get_frame();
                thread::sleep(Duration::from_micros(500));
            }
        }

        let (timestamp, view_params, buffer_ptr, frame_perfect_shift) =
            if let Some((timestamp, buffer_ptr)) = frame_result {
                let view_params = self.core_context.report_compositor_start(timestamp);

                // Avoid passing invalid timestamp to runtime
                let timestamp =
                    Duration::max(timestamp, vsync_time.saturating_sub(Duration::from_secs(1)));

                self.last_good_view_params = view_params;

                // 🎯 获取Frame-Perfect绑定的shift数据：与编码时使用的shift完全相同
                // DFRv4: Per-eye shift support for proper encode/decode alignment
                let frame_perfect_shift =
                    self.core_context
                        .get_frame_perfect_shift(timestamp)
                        .map(|shift| alvr_packets::DynamicFoveatedCenter {
                            center_shift_x: shift.shift_x,
                            center_shift_y: shift.shift_y,
                            left_shift_x: shift.left_shift_x,
                            left_shift_y: shift.left_shift_y,
                            right_shift_x: shift.right_shift_x,
                            right_shift_y: shift.right_shift_y,
                            sequence_id: Some(shift.sequence_id),
                        });

                // Frame-reuse fix: Cache shift data for use when reusing old frames
                // This ensures network latency fallback uses correct inverse-FFR parameters
                self.last_good_shift = frame_perfect_shift.clone();

                (timestamp, view_params, buffer_ptr, frame_perfect_shift)
            } else {
                // Frame reuse path: No new frame available (network latency)
                // Use cached shift data to match the cached frame's encoding
                (
                    vsync_time,
                    self.last_good_view_params,
                    ptr::null_mut(),
                    self.last_good_shift.clone(),
                )
            };

        let left_swapchain_idx = self.swapchains[0].acquire_image().unwrap();
        let right_swapchain_idx = self.swapchains[1].acquire_image().unwrap();

        self.swapchains[0]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();
        self.swapchains[1]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();

        // 🎯 FRAME-PERFECT DEBUG: Verify Frame-Perfect vs Async data alignment
        static mut RENDER_FRAME_COUNT: u32 = 0;
        unsafe {
            RENDER_FRAME_COUNT += 1;

            // Force log every 30 frames with println to bypass debug group filtering
            if RENDER_FRAME_COUNT % 30 == 0 {
                println!(
                    "=== CLIENT FRAME-PERFECT DEBUG Frame {} ===",
                    RENDER_FRAME_COUNT
                );

                // 🎯 显示Frame-Perfect数据（正确的）
                if let Some(ref center) = frame_perfect_shift {
                    let seq_info = if let Some(seq_id) = center.sequence_id {
                        format!(" [seq={}]", seq_id)
                    } else {
                        " [no-seq]".to_string()
                    };

                    println!(
                        "CLIENT FRAME-PERFECT: Frame {} timestamp={:?} - DFR{} shift=({:.3},{:.3})",
                        RENDER_FRAME_COUNT,
                        timestamp,
                        seq_info,
                        center.center_shift_x,
                        center.center_shift_y
                    );

                    log::info!(target: alvr_common::CLIENT_GFX_DBG_LABEL,
                        "🎯 FRAME-PERFECT: Frame {} ts={:?} - DFR{} shift=({:.3},{:.3})",
                        RENDER_FRAME_COUNT, timestamp, seq_info, center.center_shift_x, center.center_shift_y);
                } else {
                    println!("CLIENT FRAME-PERFECT: Frame {} timestamp={:?} - FFR mode (no Frame-Perfect DFR data)",
                        RENDER_FRAME_COUNT, timestamp);

                    log::debug!(target: alvr_common::CLIENT_GFX_DBG_LABEL,
                        "🎯 FRAME-PERFECT: Frame {} ts={:?} - FFR mode (no Frame-Perfect DFR data)",
                        RENDER_FRAME_COUNT, timestamp);
                }

                // 🎯 对比显示异步配置数据（有问题的旧方式）
                if let Some(ref async_center) = self.config.dynamic_foveated_center {
                    println!(
                        "CLIENT ASYNC-CONFIG: Frame {} - DFR shift=({:.3},{:.3}) [DEPRECATED]",
                        RENDER_FRAME_COUNT,
                        async_center.center_shift_x,
                        async_center.center_shift_y
                    );
                } else {
                    println!(
                        "CLIENT ASYNC-CONFIG: Frame {} - No async DFR data",
                        RENDER_FRAME_COUNT
                    );
                }

                // CRITICAL: Verify Frame-Perfect data is being passed to renderer
                println!(
                    "CLIENT: About to call renderer.render() with Frame-Perfect shift: {}",
                    if frame_perfect_shift.is_some() {
                        "PRESENT"
                    } else {
                        "ABSENT"
                    }
                );
            }
        }

        unsafe {
            // 🎯 使用Frame-Perfect绑定的shift数据进行Inverse FFR
            // 确保inverse FFR使用的shift与服务端编码时的shift完全一致
            self.renderer.render(
                buffer_ptr,
                [
                    StreamViewParams {
                        swapchain_index: left_swapchain_idx,
                        reprojection_rotation: Quat::IDENTITY,
                        fov: view_params[0].fov,
                    },
                    StreamViewParams {
                        swapchain_index: right_swapchain_idx,
                        reprojection_rotation: Quat::IDENTITY,
                        fov: view_params[1].fov,
                    },
                ],
                self.config.passthrough.as_ref(),
                frame_perfect_shift.as_ref(), // 🎯 关键修改：使用Frame-Perfect数据而非异步配置
            )
        };

        self.swapchains[0].release_image().unwrap();
        self.swapchains[1].release_image().unwrap();

        if !buffer_ptr.is_null() {
            if let Some(xr_now) = crate::xr_runtime_now(self.xr_session.instance()) {
                self.core_context.report_submit(
                    timestamp,
                    vsync_time.saturating_sub(Duration::from_nanos(xr_now.as_nanos() as u64)),
                );
            }
        }

        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.target_view_resolution.x as _,
                height: self.target_view_resolution.y as _,
            },
        };

        let clientside_post_processing = self
            .xr_session
            .instance()
            .exts()
            .fb_composition_layer_settings
            .and(self.config.clientside_post_processing.clone());

        let layer = ProjectionLayerBuilder::new(
            &self.stage_reference_space,
            [
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(view_params[0].pose))
                    .fov(crate::to_xr_fov(view_params[0].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[0])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(view_params[1].pose))
                    .fov(crate::to_xr_fov(view_params[1].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[1])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
            ],
            self.config
                .passthrough
                .clone()
                .map(|mode| ProjectionLayerAlphaConfig {
                    premultiplied: matches!(
                        mode,
                        PassthroughMode::Blend {
                            premultiplied_alpha: true,
                            ..
                        } | PassthroughMode::RgbChromaKey(_)
                            | PassthroughMode::HsvChromaKey(_)
                    ),
                }),
            clientside_post_processing,
        );

        (layer, timestamp)
    }
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        self.input_thread_running.set(false);
        self.input_thread.take().unwrap().join().ok();
    }
}

fn stream_input_loop(
    core_ctx: &ClientCoreContext,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_ctx: &RwLock<InteractionContext>,
    stage_reference_space: &xr::Space,
    view_reference_space: &xr::Space,
    refresh_rate: f32,
    running: Arc<RelaxedAtomic>,
) {
    let platform = alvr_system_info::platform();

    let mut last_controller_poses = [Pose::default(); 2];
    let mut last_palm_poses = [Pose::default(); 2];
    let mut last_view_params = [ViewParams::default(); 2];

    let mut deadline = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / refresh_rate);
    while running.value() {
        let int_ctx = &*interaction_ctx.read();
        // Streaming related inputs are updated here. Make sure every input poll is done in this
        // thread
        if let Err(e) = xr_session.sync_actions(&[(&int_ctx.action_set).into()]) {
            error!("{e}");
            return;
        }

        let Some(now) = crate::xr_runtime_now(xr_session.instance()).map(crate::from_xr_time)
        else {
            error!("Cannot poll tracking: invalid time");
            return;
        };

        let target_time = now + core_ctx.get_total_prediction_offset();

        let Some((head_motion, local_views)) = interaction::get_head_data(
            &xr_session,
            platform,
            stage_reference_space,
            view_reference_space,
            now,
            target_time,
            &last_view_params,
        ) else {
            continue;
        };

        if let Some(views) = local_views {
            core_ctx.send_view_params(views);
            last_view_params = views;
        }

        let mut device_motions = Vec::with_capacity(3);

        device_motions.push((*HEAD_ID, head_motion));

        let (left_hand_motion, left_hand_skeleton) = crate::interaction::get_hand_data(
            &xr_session,
            platform,
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[0],
            &mut last_controller_poses[0],
            &mut last_palm_poses[0],
        );
        let (right_hand_motion, right_hand_skeleton) = crate::interaction::get_hand_data(
            &xr_session,
            platform,
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[1],
            &mut last_controller_poses[1],
            &mut last_palm_poses[1],
        );

        // Note: When multimodal input is enabled, we are sure that when free hands are used
        // (not holding controllers) the controller data is None.
        if int_ctx.multimodal_hands_enabled || left_hand_skeleton.is_none() {
            if let Some(motion) = left_hand_motion {
                device_motions.push((*HAND_LEFT_ID, motion));
            }
        }
        if int_ctx.multimodal_hands_enabled || right_hand_skeleton.is_none() {
            if let Some(motion) = right_hand_motion {
                device_motions.push((*HAND_RIGHT_ID, motion));
            }
        }

        let eye_gazes = interaction::get_eye_gazes(
            &xr_session,
            &int_ctx.face_sources,
            stage_reference_space,
            now,
            platform,
        );

        let pico_eye_tracking_data =
            interaction::get_pico_eye_tracking_data(&int_ctx.face_sources, now);

        let eye_tracking_state = classify_eye_tracking_state(
            &int_ctx.face_sources,
            &eye_gazes,
            pico_eye_tracking_data.as_ref(),
        );

        let face_data = FaceData {
            eye_gazes,
            fb_face_expression: interaction::get_fb_face_expression(&int_ctx.face_sources, now).or(
                interaction::get_pico_face_expression(&int_ctx.face_sources, now),
            ),
            htc_eye_expression: interaction::get_htc_eye_expression(&int_ctx.face_sources, now),
            htc_lip_expression: interaction::get_htc_lip_expression(&int_ctx.face_sources, now),
            pico_eye_tracking_data,
            eye_tracking_state,
        };

        if let Some((tracker, joint_count)) = &int_ctx.body_sources.body_tracker_fb {
            device_motions.append(&mut interaction::get_fb_body_tracking_points(
                stage_reference_space,
                now,
                tracker,
                *joint_count,
            ));
        }

        if let Some(tracker) = &int_ctx.body_sources.body_tracker_bd {
            device_motions.append(&mut interaction::get_bd_body_tracking_points(
                stage_reference_space,
                now,
                tracker,
            ));
        }

        if let Some(tracker) = &int_ctx.body_sources.motion_tracker_bd {
            device_motions.append(&mut interaction::get_bd_motion_trackers(now, tracker));
        }

        // Collect high-frequency WiFi metrics for DRL-based ABR research
        let wifi_metrics = {
            #[cfg(target_os = "android")]
            {
                // Use caching to avoid expensive JNI calls at 3x refresh rate
                static WIFI_CACHE: std::sync::OnceLock<std::sync::Mutex<Option<(alvr_system_info::WiFiMetrics, std::time::Instant)>>> = std::sync::OnceLock::new();
                let cache = WIFI_CACHE.get_or_init(|| std::sync::Mutex::new(None));

                let mut cache_guard = cache.lock().unwrap();
                let now = std::time::Instant::now();

                // Refresh WiFi data every 100ms to balance performance and responsiveness
                if cache_guard.is_none() || now.duration_since(cache_guard.as_ref().unwrap().1) > std::time::Duration::from_millis(100) {
                    if let Some(wifi_data) = alvr_system_info::get_wifi_metrics() {
                        *cache_guard = Some((wifi_data, now));
                    }
                }

                cache_guard.as_ref().map(|(data, _)| {
                    // Convert from alvr_system_info::WiFiMetrics to alvr_packets::WiFiMetrics
                    Some(alvr_packets::WiFiMetrics {
                        timestamp_ns: data.timestamp_ns,
                        rssi_dbm: data.rssi_dbm,
                        frequency_mhz: data.frequency_mhz,
                        link_speed_mbps: data.link_speed_mbps,
                        mcs_index: None,                   // Not available in current implementation
                        snr_db: None,                      // Not available in current implementation
                        tx_bitrate_mbps: None,             // Not available in current implementation
                        rx_bitrate_mbps: None,             // Not available in current implementation
                        tx_retries: None,                  // Not available in current implementation
                        tx_failures: None,                 // Not available in current implementation
                        wifi_standard: "UNKNOWN".to_string(), // Not fabricating
                        channel_width: 0,                  // Not fabricating
                        guard_interval: None,              // Not available in current implementation
                    })
                }).flatten()
            }
            #[cfg(not(target_os = "android"))]
            {
                None
            }
        };

        core_ctx.send_tracking(
            Duration::from_nanos(now.as_nanos() as u64),
            device_motions,
            [left_hand_skeleton, right_hand_skeleton],
            face_data,
            wifi_metrics,
        );

        let button_entries = interaction::update_buttons(&xr_session, &int_ctx.button_actions);
        if !button_entries.is_empty() {
            core_ctx.send_buttons(button_entries);
        }

        deadline += frame_interval / 3;
        thread::sleep(deadline.saturating_duration_since(Instant::now()));
    }
}
