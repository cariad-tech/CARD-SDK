# viz_sample.py
# example run --> python viz_sample.py --base_root "/media/T7/cariad dataset/" --split_file "/media/T7/cariad dataset/val_all_data_clean.txt" --idx 8465
import argparse, numpy as np, torch, cv2, matplotlib.pyplot as plt
from dataloader import UnifiedCardDataset

def chw_to_hwc_uint8(img_t, mean, std):
    img = img_t.detach().cpu().float().numpy().transpose(1,2,0)
    img = (img * std + mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def depth_to_colormap(depth_chw, vmin=0.1, vmax=60.0):
    if depth_chw is None: return None
    d = depth_chw[0].detach().cpu().numpy()
    d = np.clip(d, vmin, vmax)
    d_norm = (d - vmin) / max(1e-6, (vmax - vmin))
    d_uint8 = (d_norm * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(d_uint8, cv2.COLORMAP_TURBO)[:, :, ::-1]
    alpha = (d > 0).astype(np.float32)[...,None]
    return (cm.astype(np.float32) * alpha + 255*(1-alpha)).astype(np.uint8)

def overlay(img, cm, a=0.6):
    return img if cm is None else (a*cm + (1-a)*img).astype(np.uint8)

def backproject(depth_chw, K_3x3):
    d = depth_chw[0].detach().cpu().numpy()
    K = K_3x3.detach().cpu().numpy()
    H,W = d.shape
    u,v = np.meshgrid(np.arange(W), np.arange(H))
    m = d > 0
    if not np.any(m): return np.zeros((0,3),np.float32), np.zeros((0,2),np.float32)
    z = d[m].astype(np.float32); u = u[m].astype(np.float32); v = v[m].astype(np.float32)
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    x = (u - cx) * z / fx; y = (v - cy) * z / fy
    return np.stack([x,y,z],1), np.stack([u,v],1)

def project(pts_cam, K, img_shape):
    H,W = img_shape[:2]
    z = pts_cam[:,2]; m = z > 1e-6
    z = np.maximum(z, 1e-6)
    u = (K[0,0]*pts_cam[:,0]/z) + K[0,2]
    v = (K[1,1]*pts_cam[:,1]/z) + K[1,2]
    inb = (u>=0)&(u<W)&(v>=0)&(v<H)
    m = m & inb
    return np.stack([u,v],1), m

def transform(pts, T):
    N = pts.shape[0]
    ph = np.hstack([pts, np.ones((N,1),np.float32)])
    return (T @ ph.T).T[:, :3]

def draw_pts(img, uv, m, color=(0,255,0), r=1):
    out = img.copy()
    for p in uv[m].astype(np.int32):
        cv2.circle(out, (int(p[0]),int(p[1])), r, color, -1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", required=True)
    ap.add_argument("--split_file", required=True)   # relative to base_root
    ap.add_argument("--idx", type=int, default=0)
    args = ap.parse_args()

    ds = UnifiedCardDataset(
        base_root=args.base_root,
        split_file=args.split_file,
        mode="val",
        height=352, width=1216,
        context_offsets=(-1,0,+1),
        stereo=False,
        use_color_aug=False, use_geom_aug=False
    )

    item = ds[args.idx]
    cam = item["meta"]["center_cam"]
    frames = item[cam]["frames"]

    f0 = frames[0]
    mean = ds.rgb_mean[0,0,:]; std = ds.rgb_std[0,0,:]

    rgb0 = chw_to_hwc_uint8(f0["rgb"], mean, std)
    K0   = f0["K"].numpy()
    Tcw0 = f0["T_cam_world"].numpy()
    agg0 = f0["depth_agg"]

    agg_vis = depth_to_colormap(agg0)
    over0   = overlay(rgb0, agg_vis, 0.6)

    # choose neighbor (prefer -1)
    neighbor = -1 if (-1 in frames and frames[-1] is not None) else (+1 if (+1 in frames and frames[+1] is not None) else None)

    if neighbor is None:
        # show only center views
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.title("center rgb (t)"); plt.imshow(rgb0); plt.axis("off")
        plt.subplot(1,2,2); plt.title("center rgb (t) + agg depth"); plt.imshow(over0); plt.axis("off")
        plt.tight_layout(); plt.show()
        return

    fn = frames[neighbor]
    rgbn = chw_to_hwc_uint8(fn["rgb"], mean, std)
    Kn   = fn["K"].numpy()
    Tcwn = fn["T_cam_world"].numpy()

    # backproject agg@t to 3D in center cam
    pts_c, _ = backproject(agg0, f0["K"])
    if pts_c.shape[0] == 0:
        print("No agg points > 0; showing only center overlays.")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.title("center rgb (t)"); plt.imshow(rgb0); plt.axis("off")
        plt.subplot(1,2,2); plt.title("center rgb (t) + agg depth"); plt.imshow(over0); plt.axis("off")
        plt.tight_layout(); plt.show()
        return

    # center cam -> world
    Pw = transform(pts_c, Tcw0)

    # world -> center cam (sanity)
    Twc0 = np.linalg.inv(Tcw0)
    Pc0  = transform(Pw, Twc0)
    uv0, m0 = project(Pc0, K0, rgb0.shape)
    rgb0_pts = draw_pts(rgb0, uv0, m0, (0,255,0), 1)

    # world -> neighbor cam
    Twcn = np.linalg.inv(Tcwn)
    Pcn  = transform(Pw, Twcn)
    uvn, mn = project(Pcn, Kn, rgbn.shape)
    rgbn_pts = draw_pts(rgbn, uvn, mn, (255,0,0), 1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,8))
    plt.subplot(2,2,1); plt.title("center rgb (t)"); plt.imshow(rgb0); plt.axis("off")
    plt.subplot(2,2,2); plt.title("center rgb (t) + agg depth"); plt.imshow(over0); plt.axis("off")
    plt.subplot(2,2,3); plt.title("center rgb (t) + reproj(agg@t)"); plt.imshow(rgb0_pts); plt.axis("off")
    plt.subplot(2,2,4); plt.title(f"neighbor rgb (t{neighbor:+}) + reproj(agg@t)"); plt.imshow(rgbn_pts); plt.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
