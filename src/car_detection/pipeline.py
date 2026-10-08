"""On-site snapshot calibration followed by car detection and ground mapping."""
import argparse
import json
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
import cv2
import numpy as np

from .config import (
    ConfigError,
    PIPELINE_DEFAULTS,
    PROJECT_ROOT,
    TRACKER_DEFAULTS,
    PipelineConfig,
    validate_pipeline_config,
)

# Ultralytics settings directory (not the model path). The parent directory
# is used by default; set YOLO_CONFIG_DIR if it is not writable.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))
from .calibration import choose_corners as select_calibration_points
from .calibration import snapshot as capture_snapshot
from .detector.inference import detect as detect_frame
from .detector.tracker import TemporalTracker
from .device import resolve_device
from .geometry.floor import FloorCalibration
from .rendering import (
    draw as render_frame,
    map_point,
    vehicle_ground_pixel,
)
from .tracking.identity import VehicleIdentityResolver, vehicle_missing_record
from .tracking.motion import MotionEstimator, missing_record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--source', default='0',
        help='Camera index such as 0, or a video file path.',
    )
    p.add_argument(
        '--tile-width-mm', type=float, default=PIPELINE_DEFAULTS.tile_width_mm,
        help='Width of one floor tile in mm (default: 600).',
    )
    p.add_argument(
        '--tile-height-mm', type=float, default=PIPELINE_DEFAULTS.tile_height_mm,
        help='Length of one floor tile in mm (default: 600).',
    )
    p.add_argument(
        '--cols', type=int, default=PIPELINE_DEFAULTS.columns,
        help='Number of floor tile columns; calibration uses cols + 1 points per row.',
    )
    p.add_argument(
        '--rows', type=int, default=PIPELINE_DEFAULTS.rows,
        help='Number of floor tile rows; calibration uses rows + 1 point rows.',
    )
    p.add_argument(
        '--measured', action='store_true',
        help='Mark the supplied tile dimensions as physically measured.',
    )
    # Model paths default relative to this file, not the shell's working folder.
    # Override with --model when the weights live elsewhere.
    # The original bundled model uses class 1 for toy cars.
    p.add_argument(
        '--model', type=Path,
        default=PIPELINE_DEFAULTS.model_path,
        help='YOLO weights (default: the original bundled checkpoint).',
    )
    p.add_argument(
        '--vehicle-class', type=int, nargs='+',
        default=list(PIPELINE_DEFAULTS.vehicle_classes),
        metavar='ID',
        help='Model class IDs to detect (default: 1; class 0 is the cone).',
    )
    p.add_argument('--device', default=PIPELINE_DEFAULTS.device,
                   help='auto, NVIDIA 0, Apple mps, or cpu')
    p.add_argument('--imgsz', type=int, default=PIPELINE_DEFAULTS.image_size,
                   help='YOLO inference size (default: 512).')
    p.add_argument('--conf', type=float, default=PIPELINE_DEFAULTS.confidence)
    p.add_argument(
        '--no-warmup', action='store_true',
        help='Skip the startup inference warm-up.',
    )
    p.add_argument(
        '--roi-padding', type=int, default=PIPELINE_DEFAULTS.roi_padding,
        help='Pixels added around color regions before YOLO (default: 256).',
    )
    p.add_argument(
        '--max-roi-crops', type=int, default=PIPELINE_DEFAULTS.max_roi_crops,
        help='Maximum number of YOLO crops per detection frame (default: 4).',
    )
    p.add_argument(
        '--min-roi-size', type=int, default=PIPELINE_DEFAULTS.min_roi_size,
        help='Minimum square ROI side in pixels (default: 512).',
    )
    p.add_argument('--min-motion-speed', type=float,
                   default=PIPELINE_DEFAULTS.min_motion_speed,
                   help='Hide movement arrow below this speed, mm/s')
    p.add_argument('--min-motion-displacement', type=float,
                   default=PIPELINE_DEFAULTS.min_motion_displacement,
                   help='Minimum recent displacement for an arrow, mm')
    p.add_argument('--detect-every', type=int,
                   default=PIPELINE_DEFAULTS.detect_every,
                   help='Run detection every N frames (default: 2)')
    p.add_argument('--max-frames',type=int)
    # Results are opt-in; --save-results enables this output directory.
    p.add_argument('--save-results',action='store_true',help='Save calibration, images, video and position logs (default: display only)')
    p.add_argument('--output-dir', type=Path, default=PIPELINE_DEFAULTS.output_dir)
    args: PipelineConfig = PipelineConfig.from_namespace(p.parse_args(argv))
    try:
        validate_pipeline_config(args)
    except ConfigError as exc:
        p.error(str(exc))
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError('Cannot open source. Try another camera index; close other camera apps and check permissions.')
    # Keep calibration in memory when results are not requested.
    session=None
    writer=None
    grid_indices = None
    try:
        frame = capture_snapshot(cap)
        if frame is None:
            return
        selected = select_calibration_points(frame, args.cols, args.rows)
        if selected is None:
            return
        points, grid_indices = selected
        calibration = FloorCalibration.from_corners(
            points, frame.shape[1::-1], args.tile_width_mm, args.tile_height_mm,
            args.cols, args.rows, args.measured, grid_indices,
        )
        H = calibration.matrix
        size = frame.shape[1::-1]
        metadata = {
            'source': args.source,
            'image_size': size,
            'image_to_world_mm': H.tolist(),
            'reference_pixels': np.asarray(points).tolist(),
            'reference_grid_indices': grid_indices,
            'tile_width_mm': args.tile_width_mm,
            'tile_height_mm': args.tile_height_mm,
            'tile_count': [args.cols, args.rows],
            'dimensions_measured': args.measured,
            'limitations': (
                'Fixed camera required; no lens undistortion; bbox bottom-center is '
                'a ground-position proxy; track IDs are not permanent vehicle '
                'identity; movement direction is trajectory direction; no UDP.'
            ),
            'schema_version': 1,
            'coordinate_system': (
                'Grid points are clicked row-major; origin is first point, '
                '+X toward the second point, +Y toward the first point of the '
                'second row; Z=0; units mm'
            ),
        }
        if args.save_results:
            session=args.output_dir/uuid.uuid4().hex[:12]
            session.mkdir(parents=True)
            (session/'calibration.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            cv2.imencode('.jpg',frame)[1].tofile(str(session/'calibration_snapshot.jpg'))
            print('Calibration saved:',session,flush=True)
        else:
            print('Display only: calibration and detection results will not be saved.',flush=True)
        from ultralytics import YOLO
        args.device = resolve_device(args.device)
        import torch
        device_name = (
            torch.cuda.get_device_name(int(args.device))
            if args.device.isdigit() and torch.cuda.is_available()
            else args.device
        )
        print(
            f'Compute device: {args.device} ({device_name}) | '
            f'imgsz={args.imgsz} | detect every={args.detect_every}',
            flush=True,
        )
        model = YOLO(str(args.model))
        model_class_names = {
            class_id: model.names.get(class_id)
            for class_id in args.vehicle_class
        }
        if model.task != 'detect' or any(
            class_name is None for class_name in model_class_names.values()
        ):
            raise ValueError(
                f'Model must expose detection classes {args.vehicle_class}; '
                f'available classes: {model.names}'
            )
        print(
            'Detection classes: ' + ', '.join(
                f'{class_id} ({class_name})'
                for class_id, class_name in model_class_names.items()
            ),
            flush=True,
        )
        if not args.no_warmup:
            warmup_size = max(32, min(args.min_roi_size, 512))
            warmup = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
            warmup_started = time.perf_counter()
            model.predict(
                warmup,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=args.vehicle_class,
                device=args.device,
                verbose=False,
            )
            print(
                f'Model warm-up: {(time.perf_counter() - warmup_started) * 1000:.0f} ms',
                flush=True,
            )
        tracker = TemporalTracker(
            TRACKER_DEFAULTS.ema_alpha,
            TRACKER_DEFAULTS.confirm_hits,
            TRACKER_DEFAULTS.confirm_window,
            TRACKER_DEFAULTS.max_missed,
            TRACKER_DEFAULTS.match_iou,
            TRACKER_DEFAULTS.max_center_distance,
            TRACKER_DEFAULTS.box_alpha,
        )
        motion=MotionEstimator(min_speed_mm_s=args.min_motion_speed,min_displacement_mm=args.min_motion_displacement)
        identity_resolver=VehicleIdentityResolver()
        states={car_id:vehicle_missing_record(car_id,0) for car_id in (1,2)}
        known_ids={1,2}
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = (
            float(reported_fps)
            if np.isfinite(reported_fps) and 0 < reported_fps <= 240
            else 30.0
        )
        if not isinstance(source,int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        if session is not None and not isinstance(source, int):
            writer = cv2.VideoWriter(
                str(session / 'ground_detection.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (1200, 800),
            )
            if not writer.isOpened():
                raise RuntimeError('Cannot create output video')
        index = 0
        started = time.perf_counter()
        perf_average = {}
        last_report = started
        previous_timestamp_us = -1
        log_path = session / 'positions.jsonl' if session is not None else None
        with (log_path.open('w', encoding='utf-8')
              if log_path is not None else nullcontext(None)) as log:
            while True:
                loop_started = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1::-1] != size:
                    raise RuntimeError('Resolution changed; recalibration required')
                read_finished=time.perf_counter()
                # Timestamp after reading and before inference so inference time
                # does not distort motion deltas.
                timestamp_us = (
                    int((time.perf_counter() - started) * 1e6)
                    if isinstance(source, int)
                    else round(index / fps * 1e6)
                )
                # Some codecs report a very high or rounded FPS, which can
                # produce duplicate timestamps after conversion to microseconds.
                # MotionEstimator requires strict ordering, so enforce it here.
                if timestamp_us <= previous_timestamp_us:
                    timestamp_us = previous_timestamp_us + 1
                previous_timestamp_us = timestamp_us
                detection_frame = index % args.detect_every == 0
                if detection_frame:
                    visible = tracker.update(detect_frame(model, frame, args))
                else:
                    visible = tracker.predict()
                detection_finished = time.perf_counter()
                active = [car for car in visible if car.get('missed', 0) == 0]
                active = identity_resolver.resolve(frame, active)
                active_ids = {
                    car['car_id'] for car in active if car.get('car_id') is not None
                }
                for car_id in known_ids - active_ids:
                    motion.mark_missing(car_id)
                    states[car_id]=vehicle_missing_record(car_id,timestamp_us)
                if detection_frame:
                    for car in active:
                        car_id = car.get('car_id')
                        if car_id is None:
                            continue
                        # Restart motion history when identity moves to another track ID.
                        if states[car_id].get('track_id') != car['track_id']:
                            motion.mark_missing(car_id)
                        uv = vehicle_ground_pixel(car['box'])
                        xy = map_point(uv, H).tolist()
                        state = {
                            'car_id': car_id,
                            'track_id': car['track_id'],
                            'timestamp_us': timestamp_us,
                            'visible': True,
                            'image_uv': uv,
                            'floor_xy_mm': xy,
                            'dimensions_measured': args.measured,
                            'predicted_frame': False,
                            'vehicle_name': car['vehicle_name'],
                            'identity_confidence': car['identity_confidence'],
                        }
                        state.update(
                            motion.update(car_id, timestamp_us, xy)
                        )
                        states[car_id] = state
                # Prediction frames also update motion history. This keeps the
                # movement arrow responsive when detection is throttled.
                for car in active:
                    if not detection_frame:
                        car_id = car.get('car_id')
                        state = states.get(car_id)
                        if state is not None:
                            uv = vehicle_ground_pixel(car['box'])
                            xy = map_point(uv, H).tolist()
                            state.update(
                                image_uv=uv,
                                floor_xy_mm=xy,
                                predicted_frame=True,
                            )
                            state.update(
                                motion.update(car_id, timestamp_us, xy)
                            )
                for state in states.values():
                    state['timestamp_us']=timestamp_us
                tracking_finished = time.perf_counter()
                canvas, cars = render_frame(
                    frame, active, H,
                    args.cols * args.tile_width_mm,
                    args.rows * args.tile_height_mm,
                    args.measured, args.cols, args.rows, states,
                )
                drawing_finished = time.perf_counter()
                # Use the previous smoothed frame time instead of display FPS.
                if perf_average:
                    rate = 1000 / max(perf_average['total_ms'], 0.001)
                    crop_count = getattr(args, 'last_detect_perf', {}).get('crops', 0)
                    cv2.putText(
                        canvas,
                        f'FPS {rate:.1f} | device {args.device} | crops {crop_count}',
                        (12, 758), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1,
                    )
                if log is not None:
                    records = list(states.values()) or [missing_record(None, timestamp_us)]
                    log.write(json.dumps({
                        'frame': index,
                        'timestamp_us': timestamp_us,
                        'cars': records,
                        'no_car_position_mm': [-1000.0, -1000.0] if not active else None,
                    }) + '\n')
                if writer:
                    writer.write(canvas)
                if session is not None and index==20:
                    cv2.imencode('.jpg',canvas)[1].tofile(str(session/'preview.jpg'))
                index += 1
                cv2.imshow('Ground detection | Q/ESC quit', canvas)
                if (
                    cv2.waitKey(1) & 255 in (27, ord('q'))
                    or cv2.getWindowProperty(
                        'Ground detection | Q/ESC quit', cv2.WND_PROP_VISIBLE
                    ) < 1
                ):
                    break
                completed = time.perf_counter()
                detection_perf = getattr(args, 'last_detect_perf', {}) if detection_frame else {}
                measured_perf = {
                    'read_ms': (read_finished - loop_started) * 1000,
                    'detect_ms': (detection_finished - read_finished) * 1000,
                    'track_ms': (tracking_finished - detection_finished) * 1000,
                    'draw_ms': (drawing_finished - tracking_finished) * 1000,
                    'display_io_ms': (completed - drawing_finished) * 1000,
                    'total_ms': (completed - loop_started) * 1000,
                    'roi_ms': detection_perf.get('roi_ms', 0),
                    'model_ms': detection_perf.get('model_ms', 0),
                }
                for name, value in measured_perf.items():
                    perf_average[name] = (
                        value if name not in perf_average
                        else 0.9 * perf_average[name] + 0.1 * value
                    )
                if completed - last_report >= 2:
                    fps = 1000 / max(perf_average['total_ms'], 0.001)
                    metrics = ' | '.join(
                        f'{name}={value:.1f}'
                        for name, value in perf_average.items()
                    )
                    crop_count = getattr(args, 'last_detect_perf', {}).get('crops', 0)
                    print(
                        f'Performance (smoothed): FPS={fps:.1f} | {metrics} '
                        f'| crops={crop_count}',
                        flush=True,
                    )
                    last_report = completed
                if args.max_frames and index >= args.max_frames:
                    break
        print(
            'Frames processed:', index,
            'session:', session if session is not None else 'not saved',
            flush=True,
        )
        if perf_average:
            print('Final performance (smoothed):',json.dumps(perf_average),flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
