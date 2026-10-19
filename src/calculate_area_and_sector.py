import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 从文件加载参数
def load_params_from_file(params_file="params.txt"):
    params = {}
    
    with open(params_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 7:
                idx = int(parts[0])
                r0 = float(parts[1])
                r1 = float(parts[2])
                r2 = float(parts[3])
                t0 = float(parts[4])
                t1 = float(parts[5])
                t2 = float(parts[6])
                params[idx] = (r0, r1, r2, t0, t1, t2)
    
    print(f"参数加载完成，共 {len(params)} 份参数")
    return params

def cubic_bezier(p0, p1, p2, p3, num_points=30):
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    curve = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    return curve

def generate_square_control_points(r0, r1, r2, cell_size=33, radius_ratio=0.8):
    p0 = np.array([r0, 0])  # 0°
    p1 = np.array([r1 * np.cos(np.pi/8), r1 * np.sin(np.pi/8)])  # 22.5°
    p2 = np.array([r2 * np.cos(np.pi/4), r2 * np.sin(np.pi/4)])  # 45°
    
    p3 = np.array([p1[1], p1[0]])  # 67.5°
    p4 = np.array([0, r0])  # 90°
    
    all_points = []
    
    all_points.append(tuple(p0))        # 0°
    all_points.append(tuple(p1))        # 22.5°
    all_points.append(tuple(p2))        # 45°
    all_points.append(tuple(p3))        # 67.5°
    all_points.append(tuple(p4))        # 90°
    
    all_points.append(tuple([-p3[0], p3[1]]))  # 112.5°
    all_points.append(tuple([-p2[0], p2[1]]))  # 135°
    all_points.append(tuple([-p1[0], p1[1]]))  # 157.5°
    all_points.append(tuple([-p0[0], p0[1]]))  # 180°
    
    all_points.append(tuple([-p1[0], -p1[1]]))  # 202.5°
    all_points.append(tuple([-p2[0], -p2[1]]))  # 225°
    all_points.append(tuple([-p3[0], -p3[1]]))  # 247.5°
    all_points.append(tuple([-p4[0], -p4[1]]))  # 270°
    
    all_points.append(tuple([p3[0], -p3[1]]))  # 292.5°
    all_points.append(tuple([p2[0], -p2[1]]))  # 315°
    all_points.append(tuple([p1[0], -p1[1]]))  # 337.5°

    unique_points = list(set(all_points))
    
    control_points = np.array([list(pt) for pt in unique_points])
    
    angles = np.arctan2(control_points[:, 1], control_points[:, 0])
    angles[angles < 0] += 2*np.pi
    sorted_indices = np.argsort(angles)
    control_points = control_points[sorted_indices]
    
    return control_points

def create_closed_bezier_curve(control_points, tensions):

    n = len(control_points)
    curve_points = []
    
    t0, t1, t2 = tensions
    
    for i in range(n):
        p0 = control_points[i]
        p3 = control_points[(i + 1) % n]
        
        p_prev = control_points[(i - 1) % n]
        p_next = control_points[(i + 2) % n]
        
        tangent_in = p0 - p_prev
        tangent_out = p3 - p0
        tangent_next = p_next - p3
        
        # 归一化切线
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-6 else v
        
        tangent_in = normalize(tangent_in)
        tangent_out = normalize(tangent_out)
        tangent_next = normalize(tangent_next)
        
        # 计算平均切线以确保C2连续性
        avg_tangent_p1 = normalize(tangent_in + tangent_out)
        avg_tangent_p2 = normalize(tangent_out + tangent_next)
        
        # 计算当前曲线段的中点角度，用于确定张力
        mid_x = (p0[0] + p3[0]) / 2
        mid_y = (p0[1] + p3[1]) / 2
        mid_angle = np.arctan2(mid_y, mid_x)
        
        angle_deg = np.degrees(mid_angle)
        if angle_deg < 0:
            angle_deg += 360
        
        normalized_angle = angle_deg % 90
        if normalized_angle < 22.5:
            tension = t0
        elif normalized_angle < 67.5:
            tension = t1
        else:
            tension = t0
        
        dist = np.linalg.norm(p3 - p0)
        p1 = p0 + tension * dist * avg_tangent_p1
        p2 = p3 - tension * dist * avg_tangent_p2
        
        segment = cubic_bezier(p0, p1, p2, p3)
        curve_points.extend(segment)
    
    return np.array(curve_points)

# 缩放曲线 以（0，0）为中心缩放
def scale_curve_to_envelope(curve, target_side=30):
    if len(curve) == 0:
        return curve
    
    max_coord = np.max(np.abs(curve))
    
    if max_coord == 0:
        return curve
    
    scale_factor = (target_side / 2) / max_coord
    
    scaled_curve = curve * scale_factor
    
    return scaled_curve

def generate_sector_curve(r0, r1, r2, tensions, target_side=30):
    p0 = np.array([r0, 0])  # 0°
    p1 = np.array([r1 * np.cos(np.pi/8), r1 * np.sin(np.pi/8)])  # 22.5°
    p2 = np.array([r2 * np.cos(np.pi/4), r2 * np.sin(np.pi/4)])  # 45°
    p3 = np.array([p1[1], p1[0]])  # 67.5°
    p4 = np.array([-r0, 0])  # 90°
    p5 = np.array([-p1[0], p1[1]])  # 112.5° 
    p6 = np.array([-p2[0], p2[1]])  # 135° 
    p7 = np.array([-p3[0], p3[1]])  # 157.5° 
    p8 = np.array([-r0, 0])  # 180° 
    p9 = np.array([-p1[0], -p1[1]])  # 202.5° 
    p10 = np.array([-p2[0], -p2[1]])  # 225° 
    p11 = np.array([-p3[0], -p3[1]])  # 247.5°
    p12 = np.array([r0, 0])  # 270°
    p13 = np.array([p1[0], -p1[1]])  # 292.5°
    p14 = np.array([p2[0], -p2[1]])  # 315° 
    p15 = np.array([p3[0], -p3[1]])  # 337.5° 
    
    control_points = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15])
    
    full_curve = create_closed_bezier_curve(control_points, tensions)
    
    # 缩放曲线，使其包络正方形边长为30mm
    scaled_curve = scale_curve_to_envelope(full_curve, target_side)
    
    # 提取0°~45°扇区的曲线点
    start_idx = -1
    for i, point in enumerate(scaled_curve):
        x, y = point[0], point[1]
        if x >= -1e-8 and y >= -1e-8 and y <= x + 1e-8:
            start_idx = i
            break
    
    sector_points = []
    if start_idx != -1:
        for i in range(start_idx, len(scaled_curve)):
            point = scaled_curve[i]
            x, y = point[0], point[1]
            if x >= -1e-8 and y >= -1e-8 and y <= x + 1e-8:
                sector_points.append(point)
            else:
                break
    
    final_points = []
    
    final_points.append([16.5, 0.0])
    
    if len(sector_points) > 0:
        final_points.extend(sector_points)
    
    final_points.append([16.5, 16.5])
    
    return np.array(final_points)

# 计算多边形的面积
def calculate_polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def calculate_area_ratio(curve, square_side=33):
    square_area = square_side * square_side
    
    curve_area = calculate_polygon_area(curve)
    
    area_ratio = curve_area / square_area
    
    return area_ratio

def plot_full_curve(curve, cell_size=33, save_path=None):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    
    square = plt.Rectangle((-16.5, -16.5), 33, 33, 
                          fill=False, edgecolor='black', linestyle='-', linewidth=1)
    ax.add_patch(square)
    
    if len(curve) > 0:
        ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2)
        
        ax.fill(curve[:, 0], curve[:, 1], 'lightblue', alpha=0.4)
    
    ax.set_xlim(-16.5, 16.5)
    ax.set_ylim(-16.5, 16.5)
    ax.set_aspect('equal')
    
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def process_image_by_index(idx, params_file="params.txt", cell_size=33, radius_ratio=0.8, save_dir="full_curve_new", target_side=30):
    params = load_params_from_file(params_file)
    
    if idx not in params:
        print(f"序号 {idx} 不存在于参数文件中")
        return
    
    r0, r1, r2, t0, t1, t2 = params[idx]
    tensions = (t0, t1, t2)
    
    print(f"处理序号 {idx}，参数：r0={r0:.6f}, r1={r1:.6f}, r2={r2:.6f}, t0={t0:.6f}, t1={t1:.6f}, t2={t2:.6f}")
    
    control_points = generate_square_control_points(r0, r1, r2, cell_size, radius_ratio)
    
    curve = create_closed_bezier_curve(control_points, tensions)
    
    scaled_curve = scale_curve_to_envelope(curve, target_side)
    
    area_ratio = calculate_area_ratio(scaled_curve, 33)
    area_percent = int(area_ratio * 100)
    print(f"面积比例：{area_ratio:.6f}，面积占比百分数：{area_percent}%")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    sector_points = []
    sector_points.append([16.5, 0.0])
    
    for point in scaled_curve:
        x, y = point[0], point[1]
        if x >= -1e-8 and y >= -1e-8 and y <= x + 1e-8:
            sector_points.append(point)
    
    sector_points.append([16.5, 16.5])
    
    sector_curve = np.array(sector_points)
    
    txt_save_path = os.path.join(save_dir, f"{area_percent}_{idx:05d}_sector.txt")
    with open(txt_save_path, 'w') as f:
        f.write("x,y\n")
        for point in sector_curve:
            f.write(f"{point[0]:.6f},{point[1]:.6f}\n")
    print(f"八分之一曲线已保存到 {txt_save_path}")
    
    save_path = os.path.join(save_dir, f"{area_percent}_{idx:05d}.png")
    plot_full_curve(scaled_curve, target_side, save_path)
    
    return area_ratio

import re

# 从文件夹中提取图片序号
def extract_indices_from_folder(folder_path):
    indices = []
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return indices
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            match = re.match(r'(\d+)\.png', filename)
            if match:
                idx = int(match.group(1))
                indices.append(idx)
    
    return indices

def process_folders(folders,save_dir="full_curve_new"):
    for folder in folders:
        print(f"处理文件夹：{folder}")
        indices = extract_indices_from_folder(folder)
        print(f"找到 {len(indices)} 个图片序号")
        
        for idx in indices:
            process_image_by_index(idx=idx,save_dir=save_dir)

def main():
    """Create sector-point files from one or more generator output folders.

    The original research script contained local workstation paths in this
    function. They are intentionally replaced with explicit command-line
    arguments so the public release never assumes a private data location.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute void-area ratios and write one-eighth sector points."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Folders containing numbered PNG images produced by the geometry generator.",
    )
    parser.add_argument(
        "--output-dir",
        default="full_curve_new",
        help="Directory for generated sector files and scaled images.",
    )
    args = parser.parse_args()
    process_folders(args.folders, args.output_dir)

if __name__ == "__main__":
    main()
