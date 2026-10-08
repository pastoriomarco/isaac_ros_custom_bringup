import rclpy, json, sys
from simulation_interfaces.srv import GetEntityState, GetSimulationState
rclpy.init(); n = rclpy.create_node('gt_reader_trocar')
c = n.create_client(GetEntityState, '/get_entity_state'); c.wait_for_service(timeout_sec=10)
names = ['/World/trocar_short'] + ['/World/trocar_short_%02d' % i for i in range(1, 6)] + ['/World/rsd455/RSD455/Camera_OmniVision_OV9782_Color/camera_color_optical_frame']
out = {}
for e in names:
    r = GetEntityState.Request(); r.entity = e
    f = c.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=10)
    s = f.result().state; p = s.pose.position; q = s.pose.orientation
    out[e] = {'frame': s.header.frame_id, 'stamp': s.header.stamp.sec + s.header.stamp.nanosec * 1e-9, 'p': [p.x, p.y, p.z], 'q': [q.x, q.y, q.z, q.w]}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for k, v in out.items(): print(k.split('/')[-1], [round(x, 4) for x in v['p']], [round(x, 4) for x in v['q']])
