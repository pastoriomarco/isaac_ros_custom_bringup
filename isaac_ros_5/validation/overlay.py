import json, sys, numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
run, gtf, obj, out = sys.argv[1:5]
d = json.load(open(run + '/live.json')); gt = json.load(open(gtf))
K = np.array([[634.09, 0, 640], [0, 634.09, 360], [0, 0, 1]])
V = np.array([[float(x) for x in l.split()[1:4]] for l in open(obj) if l.startswith('v ')])[::15]
cam = gt['/World/rsd455/RSD455/Camera_OmniVision_OV9782_Color/camera_color_optical_frame']
Rcw = R.from_quat(cam['q']).inv(); tcw = -Rcw.apply(cam['p'])
pz = np.load(run + '/pair_0.npz'); rgb, mask, stamp = pz['rgb'], pz['mask'], float(pz['stamp'])
im = Image.fromarray(rgb).convert('RGB'); ov = np.array(im).astype(float)
ov[mask > 0] = ov[mask > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
im = Image.fromarray(ov.astype(np.uint8)); dr = ImageDraw.Draw(im)
def proj(Pc):
    uv = (K @ Pc.T).T; return uv[:, :2] / uv[:, 2:]
for k, g in gt.items():
    if 'trocar' not in k: continue
    Pc = Rcw.apply(R.from_quat(g['q']).apply(V) + g['p']) + tcw
    for u, v in proj(Pc): dr.point((u, v), fill=(0, 255, 0))
pose = min(d['pose'], key=lambda p: abs(p[1] - stamp)); p = pose[3][0]
Pc = R.from_quat(p[3:7]).apply(V) + p[:3]
for u, v in proj(Pc): dr.point((u, v), fill=(255, 0, 255))
det = min(d['det'], key=lambda x: abs(x[1] - stamp))
for cx, cy, w, h, s, c in det[2]:  # 640 letterbox -> 1280x720: scale 2, pad 140 rows
    x0, y0, x1, y1 = (cx - w/2) * 2, (cy - h/2 - 140) * 2, (cx + w/2) * 2, (cy + h/2 - 140) * 2
    dr.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2); dr.text((x0 + 3, y0 + 3), '%.2f' % s, fill=(255, 255, 0))
nz = np.argwhere(mask > 0); print('mask bbox xyxy', nz[:, 1].min(), nz[:, 0].min(), nz[:, 1].max(), nz[:, 0].max(), 'stamp', stamp, 'pose stamp', pose[1])
dr.text((10, 10), 'blue: FoundationPose input mask   yellow: YOLO boxes (unletterboxed)   green: GT mesh   magenta: estimated pose mesh', fill=(255, 255, 255))
im.save(out); print('saved', out)
