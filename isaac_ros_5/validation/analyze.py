import json, sys, numpy as np
from collections import Counter
from scipy.spatial.transform import Rotation as R
d = json.load(open(sys.argv[1] + '/live.json')); gt = json.load(open(sys.argv[2]))
cam = gt['/World/rsd455/RSD455/Camera_OmniVision_OV9782_Color/camera_color_optical_frame']
Rwc = R.from_quat(cam['q']); twc = np.array(cam['p'])
tro = {k.split('/')[-1]: v for k, v in gt.items() if 'trocar' in k}
ne = [p for p in d['pose'] if p[3]]
print('pose msgs', len(d['pose']), 'non-empty', len(ne), 'frames', Counter(p[2] for p in d['pose']))
print('det msgs', len(d['det']), 'non-empty', sum(1 for x in d['det'] if x[2]), 'dets/msg', Counter(len(x[2]) for x in d['det']))
sc = [b[4] for x in d['det'] for b in x[2]]; print('det scores min/med/max', np.min(sc), np.median(sc), np.max(sc)) if sc else None
print('seg', len(d['seg']), 'mask px median', np.median([s[5] for s in d['seg']]))
errs = []; picks = Counter(); axang = []
for t, s, f, ps in ne:
    p = np.array(ps[0][:3]); q = ps[0][3:7]
    pw = Rwc.apply(p) + twc; Rw = Rwc * R.from_quat(q)
    name, g = min(tro.items(), key=lambda kv: np.linalg.norm(np.array(kv[1]['p']) - pw))
    picks[name] += 1
    e = pw - np.array(g['p']); errs.append(e)
    xa = Rw.apply([1, 0, 0]); xg = R.from_quat(g['q']).apply([1, 0, 0])
    axang.append(np.degrees(np.arccos(np.clip(np.dot(xa, xg), -1, 1))))
errs = np.array(errs) * 1000
print('nearest gt trocar', picks)
print('pos err mm  mean', errs.mean(0).round(1), 'norm median %.1f p90 %.1f max %.1f' % tuple(np.percentile(np.linalg.norm(errs, axis=1), [50, 90, 100])))
print('long-axis angle err deg median %.1f p90 %.1f max %.1f' % tuple(np.percentile(axang, [50, 90, 100])))
# pose jitter (frame to frame) in camera frame
P = np.array([p[3][0][:3] for p in ne]) * 1000; print('pose std mm (cam frame)', P.std(0).round(2))
# latency: pose receive time minus input image receive time with same stamp
inmap = {round(s, 4): t for t, s in d['in_image']}
lat = [t - inmap[round(s, 4)] for t, s, f, ps in d['pose'] if round(s, 4) in inmap]
print('latency input->pose ms median %.0f p90 %.0f max %.0f (n=%d)' % (*np.percentile(np.array(lat) * 1000, [50, 90, 100]), len(lat)))
lat2 = [t - inmap[round(s, 4)] for t, s, *_ in d['seg'] if round(s, 4) in inmap]
print('latency input->mask ms median %.0f' % (np.median(lat2) * 1000))
