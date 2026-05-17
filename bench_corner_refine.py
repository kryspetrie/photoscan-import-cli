#!/usr/bin/env python3
"""Benchmark corner refinement timing v3."""
import time, onnxruntime as ort
from PIL import Image
import onnx_inference.photocrop as pc

det_session = ort.InferenceSession(str(pc.DEFAULT_DETECTION_MODEL), providers=ort.get_available_providers())
pose_session = ort.InferenceSession(str(pc.DEFAULT_POSE_MODEL), providers=ort.get_available_providers())
img = Image.open('real_world_example.jpg')

# Warmup
for _ in range(2):
    pc.infer_single(det_session, pose_session, 'real_world_example.jpg')

# Count inference calls
orig_det = pc.run_detection
orig_pose = pc.run_pose
counts = {'det': 0, 'pose': 0}

def cd(*a, **kw):
    counts['det'] += 1
    return orig_det(*a, **kw)
def cp(*a, **kw):
    counts['pose'] += 1
    return orig_pose(*a, **kw)

pc.run_detection = cd
pc.run_pose = cp

for model_name in ['pose', 'detection']:
    counts['det'] = 0; counts['pose'] = 0
    res = pc.infer_single(det_session, pose_session, 'real_world_example.jpg',
        corner_refine=True, corner_refine_model=model_name, corner_refine_iterations=2)
    print(f'{model_name}: det={counts["det"]}, pose={counts["pose"]}')

# Baseline
counts['det'] = 0; counts['pose'] = 0
res = pc.infer_single(det_session, pose_session, 'real_world_example.jpg')
print(f'baseline: det={counts["det"]}, pose={counts["pose"]}')

pc.run_detection = orig_det
pc.run_pose = orig_pose

# Timing
for mode in ['baseline', 'pose', 'detection']:
    times = []
    for i in range(4):
        t0 = time.perf_counter()
        pc.infer_single(det_session, pose_session, 'real_world_example.jpg',
            corner_refine=(mode != 'baseline'),
            corner_refine_model=mode if mode != 'baseline' else 'pose',
            corner_refine_iterations=2)
        times.append((time.perf_counter() - t0) * 1000)
    print(f'{mode}: avg={sum(times[1:])/len(times[1:]):.0f}ms  runs={[f"{t:.0f}ms" for t in times]}')