import numpy as np
import sklearn.neighbors
import scipy.sparse
import scipy.sparse.linalg
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import time

# Gaussian kernel
def gaussian_kernel(distances, sigma=0.1):
    return np.exp(-(distances ** 2) / (2.0 * sigma ** 2))


def compute_edge_strength(image_f, kernel=3, blur_sigma=0.0):
    gray = cv2.cvtColor(image_f, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0.0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return mag / (mag.max() + 1e-7)


# Feature
def build_features(image_f,
                   spatial_weight=0.05,
                   use_edges=False,
                   use_hsv=True,
                   seed_features=None,
                   distance_weight=0.35,
                   edge_weight=1.0,
                   edge_kernel=3,
                   edge_blur=0.0):
    h, w = image_f.shape[:2]
    xg, yg = np.meshgrid(np.arange(w), np.arange(h))
    xn = xg.flatten() / float(w) * spatial_weight
    yn = yg.flatten() / float(h) * spatial_weight

    if use_hsv:
        hsv = cv2.cvtColor(image_f, cv2.COLOR_BGR2HSV)
        H_rad = hsv[:, :, 0].flatten() * (np.pi / 180.0)
        S = hsv[:, :, 1].flatten()
        V = hsv[:, :, 2].flatten()
        
        cols = [np.cos(H_rad), np.sin(H_rad), S, V, xn, yn]
    else:
        lab = cv2.cvtColor(image_f, cv2.COLOR_BGR2Lab)
        L_  = lab[:, :, 0].flatten() / 100.0
        a_  = (lab[:, :, 1].flatten() + 128) / 255.0
        b_  = (lab[:, :, 2].flatten() + 128) / 255.0
        cols = [L_, a_, b_, xn, yn]

    if use_edges:
        edge_mag = compute_edge_strength(image_f, kernel=edge_kernel,
                                         blur_sigma=edge_blur)
        cols.append(edge_mag.flatten() * edge_weight)

    # Add trimap-distance guidance for hard cases (white cloth on white wall)
    # These channels bias KNN toward pixels with similar geodesic role:
    # closer to known FG interior vs closer to known BG exterior
    if seed_features is not None:
        d_fg, d_bg = seed_features
        cols.append(d_fg.flatten() * distance_weight)
        cols.append(d_bg.flatten() * distance_weight)

    return np.column_stack(cols).astype(np.float64)


# Trimap preprocessing
def dilate_trimap(trimap, radius=5):
    k         = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    fg_eroded = cv2.erode((trimap == 255).astype(np.uint8), k)
    bg_eroded = cv2.erode((trimap ==   0).astype(np.uint8), k)
    out       = np.full_like(trimap, 128)
    out[fg_eroded == 1] = 255
    out[bg_eroded == 1] = 0
    return out


def build_seed_distance_features(trimap):
    fg_known = (trimap == 255).astype(np.uint8)
    bg_known = (trimap == 0).astype(np.uint8)

    # distanceTransform computes distance to zero pixels, so invert masks
    d_fg = cv2.distanceTransform(1 - fg_known, cv2.DIST_L2, 3)
    d_bg = cv2.distanceTransform(1 - bg_known, cv2.DIST_L2, 3)

    d_fg = d_fg / (d_fg.max() + 1e-7)
    d_bg = d_bg / (d_bg.max() + 1e-7)
    return d_fg.astype(np.float32), d_bg.astype(np.float32)


# KNN MATTING
def knn_matting(image_bgr, trimap,
                my_lambda=100,
                K=10,
                spatial_weight=0.05,
                sigma=0.1,
                use_edges=False,
                dilate_radius=0,
                use_hsv=True,
                use_seed_distance=True,
                distance_weight=0.35,
                edge_weight=1.0,
                edge_kernel=3,
                edge_blur=0.0,
                use_edge_barrier=False,
                edge_barrier=0.0,
                edge_barrier_kernel=3,
                edge_barrier_blur=1.0):
    h, w = image_bgr.shape[:2]
    N    = h * w

    if dilate_radius > 0:
        trimap = dilate_trimap(trimap, radius=dilate_radius)

    image_f  = image_bgr.astype(np.float32) / 255.0
    trimap_f = trimap.astype(np.float32) / 255.0

    seed_features = None
    if use_seed_distance:
        seed_features = build_seed_distance_features(trimap)

    X = build_features(image_f, spatial_weight=spatial_weight,
                       use_edges=use_edges, use_hsv=use_hsv,
                       seed_features=seed_features,
                       distance_weight=distance_weight,
                       edge_weight=edge_weight,
                       edge_kernel=edge_kernel,
                       edge_blur=edge_blur)

    print(f"    Finding {K}-NN  (N={N})...")
    t0 = time.time()
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=K + 1,     
        algorithm='ball_tree',
        n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]    
    indices   = indices[:, 1:]
    print(f"    KNN done in {time.time()-t0:.1f}s")

    kv  = gaussian_kernel(distances, sigma=sigma)
    ri  = np.repeat(np.arange(N), K)
    ci  = indices.flatten()
    weights = kv.flatten()

    # Edge barrier suppresses links that cross strong boundaries.
    # This helps preserve narrow transparent gaps (sleeve-body gap).
    if use_edge_barrier and edge_barrier > 0.0:
        edge_mag = compute_edge_strength(image_f,
                                         kernel=edge_barrier_kernel,
                                         blur_sigma=edge_barrier_blur)
        edge_1d = edge_mag.flatten().astype(np.float64)
        barrier_term = 0.5 * (edge_1d[ri] + edge_1d[ci])
        weights *= np.exp(-edge_barrier * barrier_term)

    A   = scipy.sparse.csr_matrix(
              (weights, (ri, ci)), shape=(N, N), dtype=np.float64)
    A   = A.maximum(A.T)            
    D   = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten(), format='csr')
    L   = (D - A).tocsr()

    # Trimap constraint matrices
    is_fg    = (trimap_f == 1.0).astype(np.float64).flatten()
    is_bg    = (trimap_f == 0.0).astype(np.float64).flatten()
    is_known = is_fg + is_bg
    M        = scipy.sparse.diags(is_known, format='csr')
    v        = is_fg

    # SOLVE
    LHS      = (L + my_lambda * M).tocsr()
    RHS      = my_lambda * v
    warnings.filterwarnings('ignore')
    alpha_1d = _solve(LHS, RHS)

    # eliminates solver float imprecision
    alpha_1d[is_fg.astype(bool)] = 1.0
    alpha_1d[is_bg.astype(bool)] = 0.0
    alpha_1d = np.clip(alpha_1d, 0.0, 1.0)

    return alpha_1d.reshape((h, w))


def _solve(LHS, RHS):
    try:
        t0    = time.time()
        alpha = scipy.sparse.linalg.lsqr(LHS, RHS, atol=1e-6, btol=1e-6)[0]
        print(f"    LSQR done in {time.time()-t0:.1f}s")
        return alpha
    except Exception as e:
        print(f"    LSQR failed ({e}), using spsolve...")
        t0    = time.time()
        alpha = scipy.sparse.linalg.spsolve(LHS, RHS)
        print(f"    spsolve done in {time.time()-t0:.1f}s")
        return alpha


# SPEED COMPARISON  — bear image only
# def run_speed_comparison(image_bgr, trimap, img_name='bear', K=10):
#     print(f"\n  [Speed Comparison] building system for {img_name}...")
#     h, w = image_bgr.shape[:2]
#     N    = h * w

#     image_f  = image_bgr.astype(np.float32) / 255.0
#     trimap_f = trimap.astype(np.float32) / 255.0
#     X        = build_features(image_f, spatial_weight=0.05)

#     nn = sklearn.neighbors.NearestNeighbors(
#         n_neighbors=K+1, algorithm='ball_tree', n_jobs=-1)
#     nn.fit(X)
#     dist, idx = nn.kneighbors(X)
#     dist, idx = dist[:, 1:], idx[:, 1:]

#     kv  = gaussian_kernel(dist, sigma=0.1)
#     ri  = np.repeat(np.arange(N), K)
#     ci  = idx.flatten()
#     A   = scipy.sparse.csr_matrix(
#               (kv.flatten(), (ri, ci)), shape=(N, N))
#     A   = A.maximum(A.T)
#     D   = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten(), format='csr')
#     Lp  = (D - A).tocsr()
#     is_fg = (trimap_f == 1.0).astype(np.float64).flatten()
#     is_bg = (trimap_f == 0.0).astype(np.float64).flatten()
#     M_m   = scipy.sparse.diags(is_fg + is_bg, format='csr')
#     LHS   = (Lp + 100 * M_m).tocsr()
#     RHS   = 100 * is_fg

#     timings    = {}   
#     chart_labels = {} 

#     # 1. spsolve
#     try:
#         t = time.time()
#         scipy.sparse.linalg.spsolve(LHS, RHS)
#         timings['spsolve']      = time.time() - t
#         chart_labels['spsolve'] = 'spsolve\n(direct LU)'
#         print(f"    spsolve : {timings['spsolve']:.2f}s")
#     except Exception as e:
#         print(f"    spsolve failed: {e}")

#     # 2. LSQR
#     t = time.time()
#     scipy.sparse.linalg.lsqr(LHS, RHS, atol=1e-5, btol=1e-5, iter_lim=300)
#     timings['lsqr']      = time.time() - t
#     chart_labels['lsqr'] = 'LSQR\n(iterative)'
#     print(f"    LSQR    : {timings['lsqr']:.2f}s")

#     # 3. PyAMG
#     try:
#         import pyamg
#         t = time.time()
#         ml = pyamg.smoothed_aggregation_solver(LHS)
#         ml.solve(RHS, tol=1e-6)
#         timings['pyamg']      = time.time() - t
#         chart_labels['pyamg'] = 'PyAMG\n(multigrid)'
#         print(f"    PyAMG   : {timings['pyamg']:.2f}s")
#     except ImportError:
#         print("    PyAMG not installed (pip install pyamg) — skipping")

#     # Bar chart
#     if timings:
#         labels = [chart_labels[k] for k in timings]
#         vals   = list(timings.values())
#         colors = ['#4C72B0', '#DD8452', '#55A868'][:len(labels)]
#         fig, ax = plt.subplots(figsize=(max(5, len(labels)*2+1), 4))
#         bars = ax.bar(labels, vals, color=colors, width=0.5)
#         ax.set_ylabel('Time (seconds)')
#         ax.set_title(f'Solver Speed Comparison — {img_name} (N={N}, K={K})')
#         for bar, v in zip(bars, vals):
#             ax.text(bar.get_x() + bar.get_width()/2,
#                     bar.get_height() + max(vals)*0.02,
#                     f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
#         plt.tight_layout()
#         path = f'./result_lsqr/{img_name}_speed_comparison.png'
#         plt.savefig(path, dpi=150, bbox_inches='tight')
#         plt.close()
#         print(f"    Saved speed chart -> {path}")

#     return timings



# Composite
def composite(alpha, image_bgr, bg_bgr):
    h, w = image_bgr.shape[:2]
    a3   = alpha[:, :, np.newaxis]
    if bg_bgr is not None:
        out = a3 * image_bgr + (1.0 - a3) * cv2.resize(bg_bgr, (w, h))
    else:
        out = a3 * image_bgr
    return out.astype(np.uint8)


def save_alpha_grid(alphas_dict, img_name):
    n    = len(alphas_dict)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, alpha) in zip(axes, alphas_dict.items()):
        ax.imshow(alpha, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label, fontsize=7)
        ax.axis('off')
    plt.suptitle(f'{img_name} — alpha matte comparison', fontsize=10)
    plt.tight_layout()
    path = f'./result_lsqr/{img_name}_alpha_grid.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved alpha grid -> {path}")


# MAIN
if __name__ == '__main__':
    os.makedirs('./result_lsqr', exist_ok=True)

    bg_original = cv2.imread('./background/garden.png')

    image_configs = {
        'bear': {
            'configs': [
                dict(label='baseline_K10',
                     K=10, spatial_weight=0.05, sigma=0.10,
                     my_lambda=100, use_edges=False, dilate_radius=0),
                
                dict(label='nonlocal_K15',
                     K=15, spatial_weight=0.02, sigma=0.08,
                     my_lambda=100, use_edges=False, dilate_radius=0),

                dict(label='fur_crisp_local_K15',
                     K=15, 
                     spatial_weight=0.08, 
                     sigma=0.08, 
                     my_lambda=100, use_edges=False, dilate_radius=0),
            ],
            'run_speed': True,
        },
        'woman': {
            'configs': [
                dict(label='flow_lambda_40_K10',
                     K=10, 
                     spatial_weight=0.05, 
                     sigma=0.10,            
                     my_lambda=40,         
                     use_edges=False, 
                     dilate_radius=0,
                     use_hsv=False, use_seed_distance=False, use_edge_barrier=False),

                dict(label='trace_hair_K12',
                     K=12,                  
                     spatial_weight=0.08,   
                     sigma=0.08,            
                     my_lambda=100, 
                     use_edges=False, 
                     dilate_radius=0,
                     use_hsv=False, use_seed_distance=False, use_edge_barrier=False),

                dict(label='the_absolute_limit_K12',
                     K=12,                  
                     spatial_weight=0.10,  
                     sigma=0.09,            
                     my_lambda=50,          
                     use_edges=False, 
                     dilate_radius=0,
                     use_hsv=False, use_seed_distance=False, use_edge_barrier=False),
            ],
            'run_speed': False,
        },
        'white_cloth': {
            'configs': [  
                dict(label='K15_sg0.07',
                     K=15,
                     spatial_weight=0.22,
                     sigma=0.07,
                     my_lambda=350,
                     dilate_radius=0,       
                     use_hsv=False,          
                     use_edges=True,        
                     use_seed_distance=True, 
                     distance_weight=0.65,
                     edge_weight=1.0,
                     edge_kernel=3),

                dict(label='K20_sg0.07',
                     K=20,
                     spatial_weight=0.22,
                     sigma=0.07,
                     my_lambda=350,
                     dilate_radius=0,        
                     use_hsv=False,        
                     use_edges=True,         
                     use_seed_distance=True, 
                     distance_weight=0.65,
                     edge_weight=1.0,
                     edge_kernel=3),

                dict(label='K20_sg0.05',
                     K=20,
                     spatial_weight=0.22,
                     sigma=0.05,
                     my_lambda=400,
                     dilate_radius=0,       
                     use_hsv=False,         
                     use_edges=True,         
                     use_seed_distance=True,
                     distance_weight=0.65,
                     edge_weight=1.0,
                     edge_kernel=3),
            ],
            'run_speed': False,
        },
    }

    for img_name, spec in image_configs.items():
        print(f"\n{'='*55}")
        print(f"  IMAGE: {img_name.upper()}")
        print(f"{'='*55}")

        image  = cv2.imread(f'./image/{img_name}.png')
        trimap = cv2.imread(f'./trimap/{img_name}.png', cv2.IMREAD_GRAYSCALE)
        if image is None or trimap is None:
            print(f"  Skipping {img_name}: files not found.")
            continue

        collected_alphas = {}

        for cfg in spec['configs']:
            label = cfg.pop('label')
            print(f"\n  Config: {label}  params={cfg}")

            alpha = knn_matting(image, trimap, **cfg)
            cfg['label'] = label

            collected_alphas[label] = alpha
            cv2.imwrite(f'./result_lsqr/{img_name}_{label}.png',
                        composite(alpha, image, bg_original))
            cv2.imwrite(f'./result_lsqr/{img_name}_{label}_alpha.png',
                        (alpha * 255).astype(np.uint8))
            print(f"  Saved: {img_name}_{label}")

        # Alpha matte comparison grid
        save_alpha_grid(collected_alphas, img_name)

        # Speed comparison — bear only
        # if spec['run_speed']:
        #     run_speed_comparison(image, trimap, img_name=img_name, K=10)

    print("\n\nAll processing complete!")