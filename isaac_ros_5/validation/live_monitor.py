"""Read-only live monitor: records pose/detection/mask outputs with receive times; saves rgb+mask pairs."""
import argparse, json, time
import numpy as np, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection3DArray

ap = argparse.ArgumentParser(); ap.add_argument('--duration', type=float, default=60); ap.add_argument('--out', required=True)
ap.add_argument('--pose_topic', default='/output'); ap.add_argument('--save_pairs', type=int, default=3)
a = ap.parse_args()
rclpy.init(); n = rclpy.create_node('trocar_live_monitor')
t0 = time.monotonic(); st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
rec = {'rgb': [], 'det': [], 'seg': [], 'pose': [], 'in_image': [], 'in_info': []}
rgb_by_stamp = {}; pairs = 0

def img_arr(m, ch):
    return np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)

def cb_in(k):
    return lambda m: rec[k].append([time.monotonic() - t0, st(m.header)])

def cb_rgb(m):
    rec['rgb'].append([time.monotonic() - t0, st(m.header), m.header.frame_id, m.encoding, m.width, m.height])
    if pairs < a.save_pairs: rgb_by_stamp[st(m.header)] = img_arr(m, 3).copy()
    for k in sorted(rgb_by_stamp)[:-6]: rgb_by_stamp.pop(k)

def cb_det(m):
    rec['det'].append([time.monotonic() - t0, st(m.header), [[d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y,
        d.results[0].hypothesis.score if d.results else None, d.results[0].hypothesis.class_id if d.results else None] for d in m.detections]])

def cb_seg(m):
    global pairs
    arr = img_arr(m, 1)[:, :, 0]; nz = np.argwhere(arr > 0)
    bb = [int(nz[:, 1].min()), int(nz[:, 0].min()), int(nz[:, 1].max()), int(nz[:, 0].max())] if len(nz) else None
    rec['seg'].append([time.monotonic() - t0, st(m.header), m.encoding, m.width, m.height, int(len(nz)), bb])
    s = st(m.header)
    if pairs < a.save_pairs and s in rgb_by_stamp and len(nz):
        np.savez_compressed(a.out + '/pair_%d.npz' % pairs, rgb=rgb_by_stamp[s], mask=arr, stamp=s); pairs += 1

def cb_pose(m):
    ps = []
    for d in m.detections:
        for r in d.results:
            p, q = r.pose.pose.position, r.pose.pose.orientation
            ps.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w, r.hypothesis.score])
    rec['pose'].append([time.monotonic() - t0, st(m.header), m.header.frame_id, ps])

n.create_subscription(Image, '/image_rect', cb_in('in_image'), qos_profile_sensor_data)
n.create_subscription(CameraInfo, '/camera_info', cb_in('in_info'), qos_profile_sensor_data)
n.create_subscription(Image, '/rgb/image_rect_color', cb_rgb, qos_profile_sensor_data)
n.create_subscription(Detection2DArray, '/detections_output', cb_det, qos_profile_sensor_data)
n.create_subscription(Image, '/segmentation', cb_seg, qos_profile_sensor_data)
n.create_subscription(Detection3DArray, a.pose_topic, cb_pose, qos_profile_sensor_data)
while time.monotonic() - t0 < a.duration: rclpy.spin_once(n, timeout_sec=0.05)
json.dump(rec, open(a.out + '/live.json', 'w'))
for k, v in rec.items():
    span = (v[-1][0] - v[0][0]) if len(v) > 1 else 0
    print(k, 'n=%d' % len(v), 'hz=%.2f' % ((len(v) - 1) / span if span else 0), 'first=%.1fs' % v[0][0] if v else '')
print('pairs saved', pairs)
