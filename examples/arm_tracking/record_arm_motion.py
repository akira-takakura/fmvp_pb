"""Read visible right-arm RGB-D and link poses from an initialized PyBullet env.

Never changes visibility, steps the environment, applies an action, or commands
a physical robot. Object/link IDs are read dynamically, not fixed to body 2.
"""
from pathlib import Path
import numpy as np


def visible_points(depth, segmentation, view, projection, body, links):
    depth=np.asarray(depth);seg=np.asarray(segmentation,dtype=np.int64)
    if depth.ndim!=2 or seg.shape!=depth.shape:raise ValueError('depth/segmentation shape mismatch')
    link=(seg>>24)-1
    mask=(seg>=0)&((seg&((1<<24)-1))==int(body))&np.isin(link,links)
    mask &= np.isfinite(depth)&(depth>=0)&(depth<1)
    y,x=np.nonzero(mask);h,w=depth.shape
    clip=np.c_[2*(x+.5)/w-1,1-2*(y+.5)/h,2*depth[y,x]-1,np.ones(len(x))]
    inverse=np.linalg.inv(np.asarray(projection).reshape(4,4,order='F')@np.asarray(view).reshape(4,4,order='F'))
    world=clip@inverse.T
    world=world[:,:3]/world[:,3,None]
    return world,link[y,x],np.c_[x,y],mask


def pose_matrix(position, quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
        raise ValueError('expected a finite nonzero xyzw quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=position
    return T


def capture_env(env,time_s,policy_update=False):
    """Call after reset/setup_camera, then after each simulated step, in sim time."""
    human=env.human
    links=list(dict.fromkeys([human.right_upperarm,human.right_forearm,human.right_hand]))
    rgb,depth,seg=env.get_camera_image_depth()
    points,point_links,pixels,mask=visible_points(depth,seg,env.view_matrix,env.projection_matrix,human.body,links)
    poses=np.array([pose_matrix(*human.get_pos_orient(link)) for link in links])
    return dict(time_s=np.asarray(float(time_s)),policy_update=np.asarray(bool(policy_update)),
        arm_points_m=points,arm_link_ids=point_links,arm_pixels_xy=pixels,
        shoulder_m=np.asarray(human.get_pos_orient(human.j_right_shoulder_x)[0]),
        link_ids=np.asarray(links),link_T_world=poses,rgb=np.asarray(rgb)[...,:3],visible_arm_mask=mask)


def save_frame(folder,index,frame):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{index:06d}.npz',**frame)


def load_frame(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def expected_roi(reference,current,roi,source_indices):
    """Rigid-link surface proxy; not deformable skin ground truth."""
    result=np.empty_like(roi)
    point_links=reference['arm_link_ids'][source_indices]
    for link in np.unique(point_links):
        old=np.flatnonzero(reference['link_ids']==link)
        new=np.flatnonzero(current['link_ids']==link)
        if len(old)!=1 or len(new)!=1:raise ValueError('missing link pose for ROI ground truth')
        old_T=reference['link_T_world'][old[0]];new_T=current['link_T_world'][new[0]]
        delta=new_T@np.linalg.inv(old_T);which=point_links==link
        result[which]=roi[which]@delta[:3,:3].T+delta[:3,3]
    return result


class VisibleArmRecorder:
    """NumPy-only recorder for an existing simulator loop; no stepping/reset."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.index = 0
        self.last_time_s = None

    def capture(self, env, simulation_time_s, policy_update=False):
        stamp = float(simulation_time_s)
        if self.last_time_s is not None and stamp <= self.last_time_s:
            raise ValueError("simulation timestamps must increase")
        frame = capture_env(env, stamp, policy_update)
        save_frame(self.folder, self.index, frame)
        self.index += 1
        self.last_time_s = stamp
        return frame
