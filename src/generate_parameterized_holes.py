import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


pregenerated_params = []
pregenerated_tensions = []
param_index = 0

# 从文件加载参数
def load_params_from_file(params_file="params_30000.txt"):
    global pregenerated_params, pregenerated_tensions, param_index
    
    print(f"从 {params_file} 加载参数...")
    
    pregenerated_params = []
    pregenerated_tensions = []
    
    with open(params_file, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 7:
                r0 = float(parts[1])
                r1 = float(parts[2])
                r2 = float(parts[3])
                t0 = float(parts[4])
                t1 = float(parts[5])
                t2 = float(parts[6])
                
                pregenerated_params.append([r0, r1, r2])
                pregenerated_tensions.append([t0, t1, t2])
    
    param_index = 0
    print(f"参数加载完成，共 {len(pregenerated_params)} 份参数")

def line_intersection(line1, line2):
    """检查两条线段是否相交"""
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # 平行或重合
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    if 0 < ua < 1 and 0 < ub < 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    return None

# 生成唯一的随机参数
def generate_unique_params(max_radius, num_params=10000):
    params_set = set()
    params = []
    
    while len(params) < num_params:
        # 扩大随机参数范围，从0开始，确保更大的多样性
        r0 = np.random.uniform(0, max_radius)
        r1 = np.random.uniform(0, max_radius)
        r2 = np.random.uniform(0, max_radius)
        
        # 确保半径有足够的差异，避免控制点重叠
        if abs(r0 - r1) < 0.05 * max_radius or abs(r1 - r2) < 0.05 * max_radius or abs(r0 - r2) < 0.05 * max_radius:
            continue
        
        # 将参数转换为元组，用于去重
        param_tuple = (round(r0, 6), round(r1, 6), round(r2, 6))
        
        if param_tuple not in params_set:
            params_set.add(param_tuple)
            params.append([r0, r1, r2])
            
        if len(params) % 1000 == 0:
            print(f"已生成 {len(params)}/{num_params} 个唯一参数")
    
    print(f"预生成完成，共生成 {len(params)} 个唯一参数")
    return params

# 生成唯一的张力参数
def generate_unique_tensions(num_params=10000):
    tensions_set = set()
    tensions = []
    
    while len(tensions) < num_params:
        # 扩大张力参数范围，从0.05到0.8
        t0 = np.random.uniform(0.05, 0.8)
        t1 = np.random.uniform(0.05, 0.8)
        t2 = np.random.uniform(0.05, 0.8)
        
        # 确保张力有足够的差异
        if abs(t0 - t1) < 0.05 or abs(t1 - t2) < 0.05 or abs(t0 - t2) < 0.05:
            continue
        
        # 将参数转换为元组，用于去重
        tension_tuple = (round(t0, 6), round(t1, 6), round(t2, 6))
        
        if tension_tuple not in tensions_set:
            tensions_set.add(tension_tuple)
            tensions.append([t0, t1, t2])
            
        if len(tensions) % 1000 == 0:
            print(f"已生成 {len(tensions)}/{num_params} 个唯一张力参数")
    
    print(f"张力参数预生成完成，共生成 {len(tensions)} 个唯一参数")
    return tensions

# 初始化预生成参数
def init_pregenerated_params(cell_size=33, radius_ratio=0.8, num_params=10000):
    global pregenerated_params, pregenerated_tensions
    max_radius = (cell_size / 2) * radius_ratio
    
    # 生成半径参数
    print("开始预生成半径参数...")
    pregenerated_params = generate_unique_params(max_radius, num_params)
    
    # 生成张力参数
    print("开始预生成张力参数...")
    pregenerated_tensions = generate_unique_tensions(num_params)
    
    print(f"预生成参数已初始化，共 {len(pregenerated_params)} 个半径参数和 {len(pregenerated_tensions)} 个张力参数")


def generate_and_save_params(num_params=30000, cell_size=33, radius_ratio=0.8):
    max_radius = (cell_size / 2) * radius_ratio
    
    # 生成半径参数
    print("开始生成半径参数...")
    radius_params = generate_unique_params(max_radius, num_params)
    
    # 生成张力参数
    print("开始生成张力参数...")
    tension_params = generate_unique_tensions(num_params)
    
    # 保存参数到文件
    params_file = "params_30000.txt"
    print(f"保存参数到 {params_file}...")
    
    with open(params_file, 'w') as f:
        f.write("idx,r0,r1,r2,t0,t1,t2\n")
        for i in range(num_params):
            r0, r1, r2 = radius_params[i]
            t0, t1, t2 = tension_params[i]
            f.write(f"{i+1},{r0:.6f},{r1:.6f},{r2:.6f},{t0:.6f},{t1:.6f},{t2:.6f}\n")
    
    print(f"参数保存完成，共 {num_params} 份参数")
    return radius_params, tension_params

# 获取下一组预生成参数
def get_next_params():
    """
    获取下一组预生成参数
    """
    global param_index, pregenerated_params, pregenerated_tensions
    
    if param_index < len(pregenerated_params):
        params = pregenerated_params[param_index]
        tensions = pregenerated_tensions[param_index]
        param_index += 1
        return params, tensions
    else:
        # 如果预生成参数用完，使用随机生成
        print("预生成参数已用完，使用随机生成...")
        # 生成随机半径参数
        max_radius = 13.2  # 假设cell_size=33, radius_ratio=0.8
        r0 = np.random.uniform(0, max_radius)
        r1 = np.random.uniform(0, max_radius)
        r2 = np.random.uniform(0, max_radius)
        
        # 确保半径有足够的差异
        while abs(r0 - r1) < 0.05 * max_radius or abs(r1 - r2) < 0.05 * max_radius or abs(r0 - r2) < 0.05 * max_radius:
            r0 = np.random.uniform(0, max_radius)
            r1 = np.random.uniform(0, max_radius)
            r2 = np.random.uniform(0, max_radius)
        
        # 生成随机张力参数
        t0 = np.random.uniform(0.05, 0.8)
        t1 = np.random.uniform(0.05, 0.8)
        t2 = np.random.uniform(0.05, 0.8)
        
        # 确保张力有足够的差异
        while abs(t0 - t1) < 0.05 or abs(t1 - t2) < 0.05 or abs(t0 - t2) < 0.05:
            t0 = np.random.uniform(0.05, 0.8)
            t1 = np.random.uniform(0.05, 0.8)
            t2 = np.random.uniform(0.05, 0.8)
        
        return [r0, r1, r2], [t0, t1, t2]

def cubic_bezier(p0, p1, p2, p3, num_points=30):
    """计算三次Bézier曲线上的点"""
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    curve = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    return curve

def generate_square_control_points(cell_size=33, radius_ratio=0.8):
    max_radius = (cell_size / 2) * radius_ratio
    
    try:
        (r0, r1, r2), _ = get_next_params()
    except IndexError:
        # 如果预生成参数用完，使用随机生成
        print("预生成参数已用完，使用随机生成...")
        r0 = np.random.uniform(0, max_radius)
        r1 = np.random.uniform(0, max_radius)
        r2 = np.random.uniform(0, max_radius)
        
        # 确保半径有足够的差异
        while abs(r0 - r1) < 0.05 * max_radius or abs(r1 - r2) < 0.05 * max_radius or abs(r0 - r2) < 0.05 * max_radius:
            r0 = np.random.uniform(0, max_radius)
            r1 = np.random.uniform(0, max_radius)
            r2 = np.random.uniform(0, max_radius)
    
    # 计算基础点坐标
    p0 = np.array([r0, 0])  # 0°
    p1 = np.array([r1 * np.cos(np.pi/8), r1 * np.sin(np.pi/8)])  # 22.5°
    p2 = np.array([r2 * np.cos(np.pi/4), r2 * np.sin(np.pi/4)])  # 45°
    
    p3 = np.array([p1[1], p1[0]])  # 67.5°
    p4 = np.array([0, r0])  # 90°
    

    # 收集所有点，使用集合去重
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
    
    print(f"生成的控制点数量: {len(control_points)}")
    
    print("控制点坐标:")
    for i, pt in enumerate(control_points):
        print(f"{i}: ({pt[0]:.3f}, {pt[1]:.3f})")
    
    return control_points, [r0, r1, r2]

def create_closed_bezier_curve(control_points):
    n = len(control_points)
    curve_points = []
    
    # 获取预生成的张力参数
    try:
        _, base_tensions = get_next_params()
    except IndexError:
        # 如果预生成参数用完，使用随机生成
        print("预生成张力参数已用完，使用随机生成...")
        base_tensions = [np.random.uniform(0.05, 0.8) for _ in range(3)]
    
    for i in range(n):
        p0 = control_points[i]
        p3 = control_points[(i + 1) % n]
        
        # 计算前一个和后一个点
        p_prev = control_points[(i - 1) % n]
        p_next = control_points[(i + 2) % n]
        
        # 计算切线方向
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
        
        # 将角度转换为0-360度
        angle_deg = np.degrees(mid_angle)
        if angle_deg < 0:
            angle_deg += 360
        
        normalized_angle = angle_deg % 90
        if normalized_angle < 22.5:
            tension = base_tensions[0]
        elif normalized_angle < 67.5:
            tension = base_tensions[1]
        else:
            tension = base_tensions[0]
        
        dist = np.linalg.norm(p3 - p0)
        p1 = p0 + tension * dist * avg_tangent_p1
        p2 = p3 - tension * dist * avg_tangent_p2
        
        segment = cubic_bezier(p0, p1, p2, p3)
        curve_points.extend(segment)
    
    return np.array(curve_points), base_tensions

def plot_lattice(ax, control_points, curve_points, cell_size=33, show_control_points=True):
    square = plt.Rectangle((-cell_size/2, -cell_size/2), cell_size, cell_size, 
                          fill=False, edgecolor='gray', linestyle='--', linewidth=1)
    ax.add_patch(square)
    
    if show_control_points:
        ax.scatter(control_points[:, 0], control_points[:, 1], c='red', s=30, 
                   label='Control Points', zorder=5)
        
        for i in range(len(control_points)):
            p1 = control_points[i]
            p2 = control_points[(i + 1) % len(control_points)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.3, linewidth=0.8)
    
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2)
    
    ax.fill(curve_points[:, 0], curve_points[:, 1], 'lightblue', alpha=0.4)
    
    ax.set_xlim(-cell_size/2 * 1.2, cell_size/2 * 1.2)
    ax.set_ylim(-cell_size/2 * 1.2, cell_size/2 * 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if show_control_points:
        ax.legend(loc='upper right')

def check_curve_intersection(curve_points):
    n = len(curve_points)
    
    step = max(1, n // 20)  # 每隔step个点检查一次
    
    for i in range(0, n - 3, step):
        for j in range(i + 2 * step, n - 1, step):
            p1 = curve_points[i]
            p2 = curve_points[i + 1]
            p3 = curve_points[j]
            p4 = curve_points[j + 1]
            
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
            intersect = (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))
            
            if intersect:
                return True
    
    return False

def check_sector_curve(curve_points):
    sector_points = []
    for point in curve_points:
        x, y = point[0], point[1]
       
        if x >= -1e-8 and y >= -1e-8 and y <= x + 1e-8:
            sector_points.append(point)
    
    if len(sector_points) < 2:
        return False  
    

    dense_samples = []
    for i in range(len(sector_points) - 1):
        p1 = sector_points[i]
        p2 = sector_points[i + 1]
                
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        
        num_samples = max(5, int(distance * 20))
        
        for t in np.linspace(0, 1, num_samples):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            dense_samples.append((x, y))
    
    for x, y in dense_samples:
        if (abs(x) < 1e-8 and abs(y) < 1e-8) or (abs(x - y) < 1e-8):
            continue
        
        if x <= 1e-8 or y <= 1e-8 or y >= x - 1e-8:
            return False      

    for i in range(len(dense_samples) - 1):
        x1, y1 = dense_samples[i]
        x2, y2 = dense_samples[i + 1]
        
        if (y1 > 1e-8 and y2 < -1e-8) or (y1 < -1e-8 and y2 > 1e-8):
            t = -y1 / (y2 - y1)
            if 0 < t < 1:
                x_intersect = x1 + t * (x2 - x1)
                if x_intersect > -1e-8:
                    return False  
        
        if (y1 < x1 - 1e-8 and y2 > x2 + 1e-8) or (y1 > x1 + 1e-8 and y2 < x2 - 1e-8):
            denominator = (y2 - x2) - (y1 - x1)
            if abs(denominator) > 1e-8:
                t = (x1 - y1) / denominator
                if 0 < t < 1:
                    x_intersect = x1 + t * (x2 - x1)
                    y_intersect = y1 + t * (y2 - y1)
                    if x_intersect > -1e-8 and y_intersect > -1e-8:
                        return False 
    
    for i in range(len(dense_samples) - 1):
        p1_start = dense_samples[i]
        p1_end = dense_samples[i + 1]
        
        for j in range(i + 2, len(dense_samples) - 1):
            p2_start = dense_samples[j]
            p2_end = dense_samples[j + 1]
            
            if line_intersection([p1_start, p1_end], [p2_start, p2_end]):
                return False 

    if sector_points:
        max_x = max(p[0] for p in sector_points)
        for x, y in dense_samples:
            if x < -1e-8 or y < -1e-8 or y > x + 1e-8:
                return False 
    
    return True 

def generate_lattice_images(start_idx=1, num_images=1000, cell_size=33, radius_ratio=0.8, batch_size=200):
    print(f"开始生成 {num_images} 个方形晶格图像，从序号 {start_idx} 开始...")
    
    params_file = "params.txt"
    with open(params_file, 'a') as f:
        for batch in range(0, num_images, batch_size):
            batch_end = min(batch + batch_size, num_images)
            print(f"处理批次 {batch//batch_size + 1}/{(num_images + batch_size - 1)//batch_size}，生成图像 {start_idx + batch}-{start_idx + batch_end - 1}")
            
            for i in range(batch, batch_end):
                current_idx = start_idx + i

                folder_idx = (current_idx - 1) // 2000
                folder_start = folder_idx * 2000 + 1
                folder_end = min((folder_idx + 1) * 2000, start_idx + num_images - 1)
                folder_name = f"{folder_start}-{folder_end}"
                
                result_dir = os.path.join("result", folder_name)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                # 生成控制点和曲线，确保0°-45°扇区的曲线满足条件
                max_attempts = 10 
                for attempt in range(max_attempts):
                    # 生成控制点和参数
                    control_points, radius_params = generate_square_control_points(cell_size=cell_size, radius_ratio=radius_ratio)
                    
                    # 生成曲线和张力参数
                    curve, tension_params = create_closed_bezier_curve(control_points)
                    
                    # 检查0°-45°扇区的曲线是否满足条件
                    if not check_sector_curve(curve):
                        print(f"扇区曲线不满足条件，第 {attempt+1} 次尝试重新生成...")
                        continue
                    
                    # 检查整个曲线是否自相交
                    if not check_curve_intersection(curve):
                        break
                    # 如果相交，重新生成
                    print(f"曲线自相交，第 {attempt+1} 次尝试重新生成...")
                
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                plot_lattice(ax, control_points, curve, cell_size=cell_size, show_control_points=True)
                
                save_path = os.path.join(result_dir, f"{current_idx:05d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')  
                plt.close(fig)
                
                r0, r1, r2 = radius_params
                t0, t1, t2 = tension_params
                f.write(f"{current_idx},{r0:.6f},{r1:.6f},{r2:.6f},{t0:.6f},{t1:.6f},{t2:.6f}\n")
                
                del control_points, curve, radius_params, tension_params
                
                if (i + 1) % 20 == 0:
                    import gc
                    gc.collect()
                    print(f"已生成 {i+1}/{num_images} 个图像")
    
    print(f"图像生成完成，共生成 {num_images} 个方形晶格图像")
    print(f"图像已保存到 result 文件夹下的子文件夹中")
    print(f"参数已记录到 {params_file} 文件")



def main():
    """主函数"""
    # 生成3万份随机参数并保存到文件中
    generate_and_save_params(num_params=30000)
    
    # 加载参数
    load_params_from_file()
    
    # 生成3万张图片，每2000份保存到一个文件夹
    total_images = 30000
    batch_size = 2000
    
    for i in range(0, total_images, batch_size):
        start_idx = i + 1
        end_idx = min(i + batch_size, total_images)
        num_images = end_idx - start_idx + 1
        
        print(f"\n生成图片 {start_idx}-{end_idx}...")
        
        global param_index
        param_index = i
        generate_lattice_images(start_idx=start_idx, num_images=num_images)
    

if __name__ == "__main__":
    main()
