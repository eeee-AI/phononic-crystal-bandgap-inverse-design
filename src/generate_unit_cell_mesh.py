import numpy as np
import gmsh
import os
import math

# 在三个点A(16.5, 16.5)、B(0,0)、C(16.5,0)之间均匀插入6个点
def interpolate_points_on_triangle():
    # 定义三个顶点
    A = (16.5, 16.5)
    B = (0.0, 0.0)
    C = (16.5, 0.0)
    
    # 初始化点列表
    kpoints = []
    
    # 1. 从A点开始
    kpoints.append(A)
    
    # 2. 在A和B之间插入6个点
    for i in range(1, 7):  # i从1到6
        t = i / 7.0
        x = A[0] + t * (B[0] - A[0])
        y = A[1] + t * (B[1] - A[1])
        kpoints.append((round(x, 6), round(y, 6)))
    
    # 3. 添加B点
    kpoints.append(B)
    
    # 4. 在B和C之间插入6个点
    for i in range(1, 7):  # i从1到6
        t = i / 7.0
        x = B[0] + t * (C[0] - B[0])
        y = B[1] + t * (C[1] - B[1])
        kpoints.append((round(x, 6), round(y, 6)))
    
    # 5. 添加C点
    kpoints.append(C)
    
    # 6. 在C和A之间插入6个点
    for i in range(1, 7):  # i从1到6
        t = i / 7.0
        x = C[0] + t * (A[0] - C[0])
        y = C[1] + t * (A[1] - C[1])
        kpoints.append((round(x, 6), round(y, 6)))
    
    return kpoints


# 在任意两点之间均匀插入指定数量的点
def interpolate_generic(point1, point2, num_points=6):
    points = []
    for i in range(1, num_points + 1):
        t = i / (num_points + 1.0)
        x = point1[0] + t * (point2[0] - point1[0])
        y = point1[1] + t * (point2[1] - point1[1])
        points.append((round(x, 6), round(y, 6)))
    return points


def interpolate_triangle_vertices(a, b, c, num_points_per_edge=6):
    kpoints = []
    
    # 从A点开始
    kpoints.append(a)
    
    # 在A和B之间插入点
    kpoints.extend(interpolate_generic(a, b, num_points_per_edge))
    
    # 添加B点
    kpoints.append(b)
    
    # 在B和C之间插入点
    kpoints.extend(interpolate_generic(b, c, num_points_per_edge))
    
    # 添加C点
    kpoints.append(c)
    
    # 在C和A之间插入点
    kpoints.extend(interpolate_generic(c, a, num_points_per_edge))
    
    return kpoints

# 从sector.txt文件中读取点序
def read_sector_points(file_path):
    points = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        if len(parts) == 2:
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                points.append([x, y])
            except ValueError:
                continue
    
    return np.array(points)

# 使用gmsh从给定的点创建2D模型，划分三角形单元
def create_gmsh_2d_model(points, output_prefix):
    gmsh.initialize()
    
    gmsh.model.add(f"Model_2d_{output_prefix}")
    
    unique_points = []
    seen = set()
    for point in points:
        x, y = point
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(point)
    
    points = np.array(unique_points)
    print(f"处理后有 {len(points)} 个唯一的点")
    
    point_tags = []
    for point in points:
        x, y = point
        tag = gmsh.model.geo.addPoint(x, y, 0, 0.5) 
        point_tags.append(tag)
    
    lines = []
    n = len(point_tags)
    for i in range(n):
        start_tag = point_tags[i]
        end_tag = point_tags[(i + 1) % n]
        line = gmsh.model.geo.addLine(start_tag, end_tag)
        lines.append(line)
    
    curve_loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    gmsh.model.geo.synchronize()
    
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6) 
    
    try:
        gmsh.model.mesh.generate(2)
    except Exception as e:
        print(f"网格生成时出错: {e}")
        gmsh.option.setNumber("Mesh.Algorithm", 1) 
        gmsh.model.mesh.generate(2)
    
    output_file = f"{output_prefix}_2d.msh"
    gmsh.write(output_file)
    print(f"Gmsh网格文件已生成：{output_file}")
    
    abaqus_file = f"{output_prefix}_2d.inp"
    gmsh.write(abaqus_file)
    print(f"Abaqus输入文件已生成：{abaqus_file}")
    
    gmsh.finalize()
    
    return abaqus_file

def clean_2d_inp_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        i = 0
        n = len(lines)
        
        while i < n:
            line = lines[i]
            
            if line.strip().startswith('*ELEMENT, type='):
                if 'CPS3' in line:
                    f.write(line)
                    i += 1
                    while i < n and lines[i].strip() and not lines[i].strip().startswith('*'):
                        f.write(lines[i])
                        i += 1
                else:
                    i += 1
                    while i < n and lines[i].strip() and not lines[i].strip().startswith('*'):
                        i += 1
            else:
                f.write(line)
                i += 1
    
    print(f"已清理文件 {input_file}，只保留CPS3单元，保存到 {output_file}")

# 读取inp文件，提取节点和单元
def read_inp_file(file_path):
    nodes = {}
    elements = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i].strip()
        if line == '*NODE':
            i += 1
            while i < n and not lines[i].strip().startswith('*'):
                node_line = lines[i].strip()
                if node_line:
                    parts = node_line.split(',')
                    if len(parts) >= 4:
                        node_id = int(parts[0].strip())
                        x = float(parts[1].strip())
                        y = float(parts[2].strip())
                        z = float(parts[3].strip()) if len(parts) > 3 else 0.0
                        nodes[node_id] = (x, y, z)
                i += 1
        else:
            i += 1
    
    i = 0
    while i < n:
        line = lines[i].strip()
        if line == '*ELEMENT, type=CPS3, ELSET=Surface1':
            i += 1
            while i < n and not lines[i].strip().startswith('*'):
                elem_line = lines[i].strip()
                if elem_line:
                    parts = elem_line.split(',')
                    if len(parts) >= 4:
                        elem_id = int(parts[0].strip())
                        node1 = int(parts[1].strip())
                        node2 = int(parts[2].strip())
                        node3 = int(parts[3].strip())
                        elements.append((elem_id, node1, node2, node3))
                i += 1
        else:
            i += 1
    
    return nodes, elements

def mirror_nodes(nodes, mirror_type):
    mirrored_nodes = {}
    
    for node_id, (x, y, z) in nodes.items():
        if mirror_type == 'y=x':
            new_x, new_y = y, x
        elif mirror_type == 'y':
            new_x, new_y = -x, y
        elif mirror_type == 'x':
            new_x, new_y = x, -y
        elif mirror_type == 'y=-x':
            new_x, new_y = -y, -x
        else:
            raise ValueError(f"未知的镜像类型: {mirror_type}")
        
        mirrored_nodes[node_id] = (new_x, new_y, z)
    
    return mirrored_nodes

def mirror_elements(elements, mirror_type):
    mirrored_elements = []
    
    for elem_id, node1, node2, node3 in elements:
        if mirror_type == 'y=x':
            new_nodes = (node1, node3, node2)
        elif mirror_type == 'y':
            new_nodes = (node1, node3, node2)
        elif mirror_type == 'x':
            new_nodes = (node1, node3, node2)
        elif mirror_type == 'y=-x':
            new_nodes = (node1, node3, node2)
        else:
            raise ValueError(f"未知的镜像类型: {mirror_type}")
        
        mirrored_elements.append((elem_id, *new_nodes))
    
    return mirrored_elements

def generate_full_2d_model(input_file, output_file):
    original_nodes, original_elements = read_inp_file(input_file)
    
    full_nodes = {}
    full_elements = []
    
    existing_nodes = {}
    
    for node_id, coords in original_nodes.items():
        full_nodes[node_id] = coords
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        existing_nodes[key] = node_id
    
    for elem in original_elements:
        full_elements.append(elem)
    
    max_node_id = max(original_nodes.keys()) if original_nodes else 0
    max_elem_id = max([elem[0] for elem in original_elements]) if original_elements else 0
    
    mirror_45_90_nodes = mirror_nodes(original_nodes, 'y=x')
    mirror_45_90_elements = mirror_elements(original_elements, 'y=x')
    
    node_id_map_45_90 = {}
    for old_id, coords in mirror_45_90_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_45_90[old_id] = new_node_id
    
    for elem in mirror_45_90_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_45_90[old_node1]
        new_node2 = node_id_map_45_90[old_node2]
        new_node3 = node_id_map_45_90[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    mirror_90_135_nodes = mirror_nodes(original_nodes, 'y')
    mirror_90_135_elements = mirror_elements(original_elements, 'y')
    
    node_id_map_90_135 = {}
    for old_id, coords in mirror_90_135_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_90_135[old_id] = new_node_id
    
    for elem in mirror_90_135_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_90_135[old_node1]
        new_node2 = node_id_map_90_135[old_node2]
        new_node3 = node_id_map_90_135[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    mirror_135_180_nodes = mirror_nodes(mirror_45_90_nodes, 'y')
    mirror_135_180_elements = mirror_elements(mirror_45_90_elements, 'y')
    
    node_id_map_135_180 = {}
    for old_id, coords in mirror_135_180_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_135_180[old_id] = new_node_id
    
    for elem in mirror_135_180_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_135_180[old_node1]
        new_node2 = node_id_map_135_180[old_node2]
        new_node3 = node_id_map_135_180[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    mirror_180_225_nodes = mirror_nodes(original_nodes, 'x')
    mirror_180_225_elements = mirror_elements(original_elements, 'x')
    
    node_id_map_180_225 = {}
    for old_id, coords in mirror_180_225_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_180_225[old_id] = new_node_id
    
    for elem in mirror_180_225_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_180_225[old_node1]
        new_node2 = node_id_map_180_225[old_node2]
        new_node3 = node_id_map_180_225[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    mirror_225_270_nodes = mirror_nodes(mirror_45_90_nodes, 'x')
    mirror_225_270_elements = mirror_elements(mirror_45_90_elements, 'x')
    
    node_id_map_225_270 = {}
    for old_id, coords in mirror_225_270_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_225_270[old_id] = new_node_id
    
    for elem in mirror_225_270_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_225_270[old_node1]
        new_node2 = node_id_map_225_270[old_node2]
        new_node3 = node_id_map_225_270[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    mirror_270_315_nodes = mirror_nodes(original_nodes, 'y=-x')
    mirror_270_315_elements = mirror_elements(original_elements, 'y=-x')
    
    node_id_map_270_315 = {}
    for old_id, coords in mirror_270_315_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_270_315[old_id] = new_node_id
    
    for elem in mirror_270_315_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_270_315[old_node1]
        new_node2 = node_id_map_270_315[old_node2]
        new_node3 = node_id_map_270_315[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    mirror_315_360_nodes = mirror_nodes(mirror_45_90_nodes, 'y=-x')
    mirror_315_360_elements = mirror_elements(mirror_45_90_elements, 'y=-x')
    
    node_id_map_315_360 = {}
    for old_id, coords in mirror_315_360_nodes.items():
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key in existing_nodes:
            new_node_id = existing_nodes[key]
        else:
            max_node_id += 1
            new_node_id = max_node_id
            full_nodes[new_node_id] = coords
            existing_nodes[key] = new_node_id
        node_id_map_315_360[old_id] = new_node_id
    
    for elem in mirror_315_360_elements:
        max_elem_id += 1
        old_node1, old_node2, old_node3 = elem[1], elem[2], elem[3]
        new_node1 = node_id_map_315_360[old_node1]
        new_node2 = node_id_map_315_360[old_node2]
        new_node3 = node_id_map_315_360[old_node3]
        full_elements.append((max_elem_id, new_node1, new_node2, new_node3))
    
    with open(output_file, 'w') as f:
        f.write('*NODE\n')
        for node_id, (x, y, z) in sorted(full_nodes.items()):
            f.write(f'{node_id},{x:.6f},{y:.6f},{z:.6f}\n')
        
        f.write('*ELEMENT, type=CPS3, ELSET=Surface1\n')
        for elem in sorted(full_elements, key=lambda x: x[0]):
            f.write(f'{elem[0]},{elem[1]},{elem[2]},{elem[3]}\n')
        
        f.write('*MATERIAL, NAME=Material-1\n')
        f.write('*ELASTIC\n')
        f.write('210000.,0.3\n')
        f.write('*SOLID SECTION, ELSET=Surface1, MATERIAL=Material-1\n')
        f.write('1.0\n')
    
    print(f"完整2D模型已生成并保存到 {output_file}")
    print(f"节点数量: {len(full_nodes)}")
    print(f"单元数量: {len(full_elements)}")

# 将2D的三角形单元转换为3D的三棱柱单元
def generate_c3d6_from_2d(input_file, output_file, thickness=2.0, num_layers=2):
    output_file_name = os.path.basename(output_file)
    # 读取2D模型的节点和单元
    nodes_2d, elements_2d = read_inp_file(input_file)
    
    # 计算每层的厚度
    layer_thickness = thickness / num_layers
    
    # 生成3D节点
    nodes_3d = {}
    max_node_id = max(nodes_2d.keys()) if nodes_2d else 0
    
    node_id_maps = [] 
    
    for layer in range(num_layers + 1):
        z = layer * layer_thickness
        node_id_map = {}
        
        for node_id, (x, y, _) in nodes_2d.items():
            max_node_id += 1
            node_id_map[node_id] = max_node_id
            nodes_3d[max_node_id] = (x, y, z)
        
        node_id_maps.append(node_id_map)
    
    # 生成3D单元（三棱柱单元 C3D6）
    elements_3d = []
    max_elem_id = 0
    
    for layer in range(num_layers):
        for elem in elements_2d:
            elem_id = elem[0]
            node1_2d = elem[1]
            node2_2d = elem[2]
            node3_2d = elem[3]
            
            node1_layer1 = node_id_maps[layer][node1_2d]
            node2_layer1 = node_id_maps[layer][node2_2d]
            node3_layer1 = node_id_maps[layer][node3_2d]
            
            node1_layer2 = node_id_maps[layer + 1][node1_2d]
            node2_layer2 = node_id_maps[layer + 1][node2_2d]
            node3_layer2 = node_id_maps[layer + 1][node3_2d]
            
            max_elem_id += 1
            elements_3d.append((max_elem_id, node1_layer1, node2_layer1, node3_layer1, node1_layer2, node2_layer2, node3_layer2))
    
    content = []
    left = [[],[],[]]
    right = [[],[],[]]
    top = [[],[],[]]
    bottom = [[],[],[]]
    lastNodeid = -1
    lastElemid = -1
    leftdownid = 1
    minx,miny,minz = -16.5,-16.5,0.0
    maxx,maxy,maxz = 16.5,16.5,2.0
    
    # 写入节点
    content.append('*NODE, nset=real_nodes\n')
    for node_id, (x, y, z) in sorted(nodes_3d.items()):
        content.append(f'{node_id},{x:.6f},{y:.6f},{z:.6f}\n')
        lastNodeid = node_id
        if x<=minx:
            minx = x
            leftdownid = node_id
        if y<=miny:
            miny = y
            leftdownid = node_id
        if z<=minz:
            minz = z
            leftdownid = node_id

        if abs(x-minx)<1e-6:
            if abs(z-minz)<1e-6:
                left[0].append([y,node_id])
            elif abs(z-maxz)<1e-6:
                left[2].append([y,node_id])
            else:
                left[1].append([y,node_id])                
        elif abs(x-maxx)<1e-6:
            if abs(z-minz)<1e-6:
                right[0].append([y,node_id])
            elif abs(z-maxz)<1e-6:
                right[2].append([y,node_id])
            else:
                right[1].append([y,node_id])
        
        if abs(y-miny)<1e-6:
            if abs(z-minz)<1e-6:
                bottom[0].append([x,node_id])
            elif abs(z-maxz)<1e-6:
                bottom[2].append([x,node_id])
            else:
                bottom[1].append([x,node_id])
        elif abs(y-maxy)<1e-6:
            if abs(z-minz)<1e-6:
                top[0].append([x,node_id])
            elif abs(z-maxz)<1e-6:
                top[2].append([x,node_id])
            else:
                top[1].append([x,node_id])        

    content.append('*ELEMENT, type=C3D6, ELSET=real_elements\n')
    for elem in sorted(elements_3d, key=lambda x: x[0]):
        content.append(f'{elem[0]},{elem[1]},{elem[3]},{elem[2]},{elem[4]},{elem[6]},{elem[5]}\n')
        lastElemid = elem[0]

    content.append('*NODE, nset=imag_nodes\n')
    for node_id, (x, y, z) in sorted(nodes_3d.items()):
        content.append(f'{node_id+lastNodeid},{x:.6f},{y:.6f},{z:.6f}\n')

    # 周期性边界节点集合
    for i in range(3):
        left[i].sort(key=lambda x: x[0])
        right[i].sort(key=lambda x: x[0])
        top[i].sort(key=lambda x: x[0])
        bottom[i].sort(key=lambda x: x[0])

    xn = 0
    yn = 0
    for tag in range(3):
        for index,_ in enumerate(left[tag]):
            content.append(f'*Nset, nset=real_left{tag}_{index}\n')
            content.append(f'{left[tag][index][1]}\n')
            xn = index
        content.append(f'*Nset, nset=real_left{tag}_end\n')
        content.append(f'{left[tag][-1][1]}\n')
        
        for index,_ in enumerate(right[tag]):
            content.append(f'*Nset, nset=real_right{tag}_{index}\n')
            content.append(f'{right[tag][index][1]}\n')
        content.append(f'*Nset, nset=real_right{tag}_end\n')
        content.append(f'{right[tag][-1][1]}\n')
        
        for index,_ in enumerate(top[tag]):
            content.append(f'*Nset, nset=real_top{tag}_{index}\n')
            content.append(f'{top[tag][index][1]}\n')
            yn = index
        content.append(f'*Nset, nset=real_top{tag}_end\n')
        content.append(f'{top[tag][-1][1]}\n')
        
        for index,_ in enumerate(bottom[tag]):
            content.append(f'*Nset, nset=real_bottom{tag}_{index}\n')
            content.append(f'{bottom[tag][index][1]}\n')
        content.append(f'*Nset, nset=real_bottom{tag}_end\n')
        content.append(f'{bottom[tag][-1][1]}\n')
            
            
        for index,_ in enumerate(left[tag]):
            content.append(f'*Nset, nset=imag_left{tag}_{index}\n')
            content.append(f'{left[tag][index][1]+lastNodeid}\n')
        content.append(f'*Nset, nset=imag_left{tag}_end\n')
        content.append(f'{left[tag][-1][1]+lastNodeid}\n')
        
        for index,_ in enumerate(right[tag]):
            content.append(f'*Nset, nset=imag_right{tag}_{index}\n')
            content.append(f'{right[tag][index][1]+lastNodeid}\n')
        content.append(f'*Nset, nset=imag_right{tag}_end\n')
        content.append(f'{right[tag][-1][1]+lastNodeid}\n')
        
        for index,_ in enumerate(top[tag]):
            content.append(f'*Nset, nset=imag_top{tag}_{index}\n')
            content.append(f'{top[tag][index][1]+lastNodeid}\n')
        content.append(f'*Nset, nset=imag_top{tag}_end\n')
        content.append(f'{top[tag][-1][1]+lastNodeid}\n')
        
        for index,_ in enumerate(bottom[tag]):
            content.append(f'*Nset, nset=imag_bottom{tag}_{index}\n')
            content.append(f'{bottom[tag][index][1]+lastNodeid}\n')
        content.append(f'*Nset, nset=imag_bottom{tag}_end\n')
        content.append(f'{bottom[tag][-1][1]+lastNodeid}\n')

        content.append('*Nset, nset=real_fix\n')
        content.append(f'{left[0][0][1]}\n')
        content.append('*Nset, nset=imag_fix\n')
        content.append(f'{left[0][0][1]+lastNodeid}\n')
    
    content.append('*Element, type=c3d6, elset=imag_elements\n')
    for elem in sorted(elements_3d, key=lambda x: x[0]):
        content.append(f'{elem[0]+lastElemid},{elem[1]+lastNodeid},{elem[3]+lastNodeid},{elem[2]+lastNodeid},{elem[4]+lastNodeid},{elem[6]+lastNodeid},{elem[5]+lastNodeid}\n')
    
    content.append('*Solid Section, elset=real_elements, material=ALUMINUM\n')
    content.append(',\n')  
    content.append('*Solid Section, elset=Imag_Elements, material=ALUMINUM\n')
    content.append(',\n')


    nodeElemInp = f'{output_file_name.split(".")[0]}_node_element.inp'
    with open(os.path.join(os.path.dirname(output_file), nodeElemInp), 'w') as f:
        f.writelines(content)

    content2 = []
    
    content2.append('*Material, name=ALUMINUM\n')
    content2.append('*Density\n')
    content2.append('2.6314e-9,\n')
    content2.append('*Elastic, type=ISO\n')
    content2.append('70.e3, 0.33\n')

    
    content2.append('*Step, name=Frequency, nlgeom=NO, perturbation\n')
    content2.append('*Frequency, eigensolver=Lanczos, acoustic coupling=on, normalization=displacement\n')
    content2.append('20\n')
    content2.append('*Boundary, op=NEW\n')
    content2.append('real_fix, 1, 2\n')
    content2.append('imag_fix, 1, 2\n')    
    content2.append('real_nodes, 3, 3\n')
    content2.append('imag_nodes, 3, 3\n')
    content2.append('*Output, field, variable=PRESELECT\n')
    content2.append('*Output, history, variable=PRESELECT\n')
    content2.append('*End Step\n')
    

    # 周期性边界条件处理
    a = 33
    A = (0.5*a, 0.5*a)
    B = (0.0, 0.0)
    C = (0.5*a, 0.0)
    kpoints = interpolate_triangle_vertices(A, B, C, 6)
        
    for kpi,kpoint in enumerate(kpoints):
        kx,ky = kpoint

        fai_x = kx*a
        fia_y = ky*a

        C1 = math.cos(fai_x)
        S1 = math.sin(fai_x)

        C2 = math.cos(fia_y)
        S2 = math.sin(fia_y)

        fai_corner_ru = kx*a + ky*a
        fai_corner_ud = kx*a
        fai_corner_lu = ky*a

        C_total_ru = math.cos(fai_corner_ru)
        S_total_ru = math.sin(fai_corner_ru)
        C_total_rd = math.cos(fai_corner_ud)
        S_total_rd = math.sin(fai_corner_ud)
        C_total_lu = math.cos(fai_corner_lu)
        S_total_lu = math.sin(fai_corner_lu)

        # 实部方程系数：(-cos(φ), +sin(φ))
        # 虚部方程系数：(-sin(φ), -cos(φ))        
        content1 = [f'*Include, input={nodeElemInp}\n']
        # 右上角
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right0_end, 1, 1.0, real_fix, 1, {-C_total_ru:.6f}, imag_fix, 1, {S_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right0_end, 1, 1.0, real_fix, 1, {-S_total_ru:.6f}, imag_fix, 1, {-C_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right0_end, 2, 1.0, real_fix, 2, {-C_total_ru:.6f}, imag_fix, 2, {S_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right0_end, 2, 1.0, real_fix, 2, {-S_total_ru:.6f}, imag_fix, 2, {-C_total_ru:.6f}\n')

        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right1_end, 1, 1.0, real_fix, 1, {-C_total_ru:.6f}, imag_fix, 1, {S_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right1_end, 1, 1.0, real_fix, 1, {-S_total_ru:.6f}, imag_fix, 1, {-C_total_ru:.6f}\n')    
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right1_end, 2, 1.0, real_fix, 2, {-C_total_ru:.6f}, imag_fix, 2, {S_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right1_end, 2, 1.0, real_fix, 2, {-S_total_ru:.6f}, imag_fix, 2, {-C_total_ru:.6f}\n')

        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right2_end, 1, 1.0, real_fix, 1, {-C_total_ru:.6f}, imag_fix, 1, {S_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right2_end, 1, 1.0, real_fix, 1, {-S_total_ru:.6f}, imag_fix, 1, {-C_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right2_end, 2, 1.0, real_fix, 2, {-C_total_ru:.6f}, imag_fix, 2, {S_total_ru:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right2_end, 2, 1.0, real_fix, 2, {-S_total_ru:.6f}, imag_fix, 2, {-C_total_ru:.6f}\n')

        # 右下角
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right0_0, 1, 1.0, real_fix, 1, {-C_total_rd:.6f}, imag_fix, 1, {S_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right0_0, 1, 1.0, real_fix, 1, {-S_total_rd:.6f}, imag_fix, 1, {-C_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right0_0, 2, 1.0, real_fix, 2, {-C_total_rd:.6f}, imag_fix, 2, {S_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right0_0, 2, 1.0, real_fix, 2, {-S_total_rd:.6f}, imag_fix, 2, {-C_total_rd:.6f}\n')

        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right1_0, 1, 1.0, real_fix, 1, {-C_total_rd:.6f}, imag_fix, 1, {S_total_rd:.6f}\n')   
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right1_0, 1, 1.0, real_fix, 1, {-S_total_rd:.6f}, imag_fix, 1, {-C_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right1_0, 2, 1.0, real_fix, 2, {-C_total_rd:.6f}, imag_fix, 2, {S_total_rd:.6f}\n')   
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right1_0, 2, 1.0, real_fix, 2, {-S_total_rd:.6f}, imag_fix, 2, {-C_total_rd:.6f}\n')
        
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right2_0, 1, 1.0, real_fix, 1, {-C_total_rd:.6f}, imag_fix, 1, {S_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right2_0, 1, 1.0, real_fix, 1, {-S_total_rd:.6f}, imag_fix, 1, {-C_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_right2_0, 2, 1.0, real_fix, 2, {-C_total_rd:.6f}, imag_fix, 2, {S_total_rd:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_right2_0, 2, 1.0, real_fix, 2, {-S_total_rd:.6f}, imag_fix, 2, {-C_total_rd:.6f}\n')
        
        # 左上角
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_left0_end, 1, 1.0, real_fix, 1, {-C_total_lu:.6f}, imag_fix, 1, {S_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_left0_end, 1, 1.0, real_fix, 1, {-S_total_lu:.6f}, imag_fix, 1, {-C_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_left0_end, 2, 1.0, real_fix, 2, {-C_total_lu:.6f}, imag_fix, 2, {S_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_left0_end, 2, 1.0, real_fix, 2, {-S_total_lu:.6f}, imag_fix, 2, {-C_total_lu:.6f}\n')
        
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_left1_end, 1, 1.0, real_fix, 1, {-C_total_lu:.6f}, imag_fix, 1, {S_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_left1_end, 1, 1.0, real_fix, 1, {-S_total_lu:.6f}, imag_fix, 1, {-C_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_left1_end, 2, 1.0, real_fix, 2, {-C_total_lu:.6f}, imag_fix, 2, {S_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_left1_end, 2, 1.0, real_fix, 2, {-S_total_lu:.6f}, imag_fix, 2, {-C_total_lu:.6f}\n')
        
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_left2_end, 1, 1.0, real_fix, 1, {-C_total_lu:.6f}, imag_fix, 1, {S_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_left2_end, 1, 1.0, real_fix, 1, {-S_total_lu:.6f}, imag_fix, 1, {-C_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'real_left2_end, 2, 1.0, real_fix, 2, {-C_total_lu:.6f}, imag_fix, 2, {S_total_lu:.6f}\n')
        content1.append('*Equation\n')
        content1.append('3\n')
        content1.append(f'imag_left2_end, 2, 1.0, real_fix, 2, {-S_total_lu:.6f}, imag_fix, 2, {-C_total_lu:.6f}\n')
        

        for zn in range(0,3):
            # X
            for xi in range(1,xn):
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'real_right{zn}_{xi}, 1, 1.0, real_left{zn}_{xi}, 1, {-C1:.6f}, imag_left{zn}_{xi}, 1, {S1:.6f}\n')
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'imag_right{zn}_{xi}, 1, 1.0, real_left{zn}_{xi}, 1, {-S1:.6f}, imag_left{zn}_{xi}, 1, {-C1:.6f}\n')
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'real_right{zn}_{xi}, 2, 1.0, real_left{zn}_{xi}, 2, {-C1:.6f}, imag_left{zn}_{xi}, 1, {S1:.6f}\n')
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'imag_right{zn}_{xi}, 2, 1.0, real_left{zn}_{xi}, 2, {-S1:.6f}, imag_left{zn}_{xi}, 2, {-C1:.6f}\n')

            # Y       
            for yi in range(1,yn):
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'real_top{zn}_{yi}, 1, 1.0, real_bottom{zn}_{yi}, 1, {-C2:.6f}, imag_bottom{zn}_{yi}, 1, {S2:.6f}\n')
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'imag_top{zn}_{yi}, 1, 1.0, real_bottom{zn}_{yi}, 1, {-S2:.6f}, imag_bottom{zn}_{yi}, 1, {-C2:.6f}\n')
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'real_top{zn}_{yi}, 2, 1.0, real_bottom{zn}_{yi}, 2, {-C2:.6f}, imag_bottom{zn}_{yi}, 1, {S2:.6f}\n')
                content1.append('*Equation\n')
                content1.append('3\n')
                content1.append(f'imag_top{zn}_{yi}, 2, 1.0, real_bottom{zn}_{yi}, 2, {-S2:.6f}, imag_bottom{zn}_{yi}, 2, {-C2:.6f}\n')


        target_output_file = output_file.replace('.inp', f'_{kpi+1}.inp')
        with open(target_output_file, 'w') as f:    
            # f.writelines(content)
            f.writelines(content1)
            f.writelines(content2)
        
    
    print(f"3D模型已生成并保存到 {output_file}")
    print(f"节点数量: {len(nodes_3d)}")
    print(f"单元数量: {len(elements_3d)}")

# 处理sector.txt文件
def process_sector_file(sector_file, output_dir=None):
    base_name = os.path.basename(sector_file)
    idx = base_name.split('_sector.txt')[0]
    
    print(f"\n处理文件：{sector_file}")
    print(f"提取序号：{idx}")
    
    # 读取点序
    points = read_sector_points(sector_file)
    print(f"已读取 {len(points)} 个点")
    
    # 生成2D三角形网格
    output_prefix = f"model_{idx}"
    abaqus_file = create_gmsh_2d_model(points, output_prefix)
    
    clean_file = f"{output_prefix}_2d_clean.inp"
    clean_2d_inp_file(abaqus_file, clean_file)
    
    # 镜像模型
    full_2d_file = f"{output_prefix}_full_2d.inp"
    generate_full_2d_model(clean_file, full_2d_file)
    
    # 三棱柱单元
    if output_dir:
        full_3d_file = os.path.join(output_dir, f"{output_prefix}_c3d6.inp")
    else:
        full_3d_file = f"{output_prefix}_c3d6.inp"
    generate_c3d6_from_2d(full_2d_file, full_3d_file, thickness=2.0, num_layers=2)
    
    files_to_delete = [
        abaqus_file,
        clean_file,
        full_2d_file,
        f"{output_prefix}_2d.msh"
    ]
    
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"已删除文件：{file}")
            except:
                print(f"删除文件 {file} 失败")
    
    print(f"\n处理完成！")
    print(f"最终3D模型文件：{full_3d_file}")
    return full_3d_file

def main(input_param):
    if os.path.isdir(input_param):
        # 如果是路径，遍历所有_sector.txt结尾的文件
        print(f"遍历目录：{input_param}")
        
        # 在目录下创建inp_output文件夹
        inp_dir = os.path.join(input_param, "inp_output")
        if not os.path.exists(inp_dir):
            os.makedirs(inp_dir)
            print(f"已创建inp文件夹：{inp_dir}")
        
        sector_files = []
        
        for root, dirs, files in os.walk(input_param):
            for file in files:
                if file.endswith('_sector.txt'):
                    sector_files.append(os.path.join(root, file))
        
        print(f"找到 {len(sector_files)} 个sector.txt文件")
        
        # with open(os.path.join(input_param, "inplog.txt"), 'w') as f:
        #     pass
        count = len(sector_files)
        # for sector_file in sector_files:
        for findex in range(count,0,-1):
            sector_file = sector_files[findex-1]
            process_sector_file(sector_file, output_dir=inp_dir)
    else:
        # 如果是序号，按当前代码执行
        idx = input_param
        sector_file = f"{idx}_sector.txt"
        
        if not os.path.exists(sector_file):
            if os.path.exists(os.path.join("full_curve_new", sector_file)):
                sector_file = os.path.join("full_curve_new", sector_file)
            else:
                print(f"文件 {sector_file} 不存在")
                return
        
        process_sector_file(sector_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        idx = sys.argv[1]
    else:
        idx = input("请输入: ")
    main(idx)
