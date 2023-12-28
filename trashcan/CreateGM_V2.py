import xml.etree.ElementTree as ET
import networkx as nx
import plotly.graph_objects as go
import colorsys
import numpy as np
import re


def parse_svg(file_path):
    tree = ET.parse(file_path)
    return tree.getroot()


def extract_element_info(element, existing_tags):
    tag_with_namespace = element.tag
    tag_without_namespace = (
        tag_with_namespace.split("}")[1]
        if "}" in tag_with_namespace
        else tag_with_namespace
    )

    # 对非根元素（非SVG元素）添加编号
    if tag_without_namespace != 'svg':
        count = existing_tags.get(tag_without_namespace, 0)
        full_tag = f"{tag_without_namespace}_{count}" if count > 0 else tag_without_namespace
        existing_tags[tag_without_namespace] = count + 1
    else:
        # 根元素直接使用其标签
        full_tag = tag_without_namespace

    attributes = element.attrib
    text_content = element.text.strip() if element.text else None

    # 特殊处理某些属性，如path的'd'属性
    if tag_without_namespace == "path":
        path_data = attributes.get("d", "")
        Pcode, Pnums = parse_path_d_attribute(path_data)
        attributes["Pcode"] = Pcode
        attributes["Pnums"] = Pnums

    return full_tag, attributes, text_content


def parse_path_d_attribute(d_attribute):
    # 匹配命令字母和随后的数值
    path_commands = re.findall(r"([a-zA-Z])([^a-zA-Z]*)", d_attribute)

    Pcode = []  # 存储命令字母
    Pnums = []  # 存储数值序列

    for command, params in path_commands:
        Pcode.append(command)
        # 分割参数，并移除多余的空格
        params_list = [
            param.strip() for param in re.split(r"[ ,]+", params.strip()) if param
        ]
        Pnums.append(params_list)

    return Pcode, Pnums


def build_graph(svg_root):
    G = nx.DiGraph()  # 使用有向图
    existing_tags = {}

    def add_element_to_graph(element, parent_path="svg", level=0, layer="0", inherited_attrs={}, layer_counter=0):
        nonlocal existing_tags
        tag, attributes, text_content = extract_element_info(element, existing_tags)

        # 合并继承的属性和本地属性，继承属性优先级更低
        combined_attributes = {**attributes, **inherited_attrs}

        # 特别处理继承的transform属性
        if 'transform' in inherited_attrs:
        # 假设transform属性可以简单地串联，实际情况可能更复杂
            combined_attributes['transform'] = inherited_attrs['transform'] + " " + attributes.get('transform', '')

         # 构建节点ID
        if parent_path is None:
            # 如果没有父路径，说明是根节点
            node_id = tag
        else:
            element_id = f"{tag}_{attributes.get('id', '')}"
            node_id = f"{parent_path}/{element_id}"

        # 处理可见性
        invisible_elements = {"svg", "g", "defs", "clipPath", "mask", "pattern", "marker", "style"}
        is_visible = tag.split("_")[0] not in invisible_elements

        # 添加节点到图形
        G.add_node(node_id, tag=tag, attributes=combined_attributes, text_content=text_content, level=level, layer=layer, visible=is_visible)

        if parent_path != "svg":
            G.add_edge(parent_path, node_id)

        # 存储上一个兄弟节点ID
        previous_sibling_id = None

        # 继续处理子元素
        new_layer_counter = layer_counter
        for child in reversed(element):  # 逆序处理子元素
            child_layer = f"{layer}_{new_layer_counter}"
            child_id = add_element_to_graph(child, parent_path=node_id, level=level + 1, layer=child_layer, inherited_attrs=combined_attributes, layer_counter=new_layer_counter)
            
            # 在当前节点和上一个节点之间添加边
            if previous_sibling_id:
                G.add_edge(previous_sibling_id, child_id, color="blue", style="solid")
            previous_sibling_id = child_id

            new_layer_counter += 1
        
        return node_id  # 返回当前节点的ID


    for node, data in G.nodes(data=True):
        if data["visible"]:
            parent = next(
                (p for p in G.predecessors(node) if G.nodes[p]["visible"]), None
            )
            if parent:
                G.add_edge(parent, node, color="yellow", style="solid")
                
    add_element_to_graph(svg_root)

    return G


def compute_layout_with_progress(graph, num_steps=100, k=None):
    # 如果未指定k值，则根据节点数量计算
    if k is None:
        k = 1 / np.sqrt(len(graph.nodes()))

    # 使用spring_layout计算初始布局
    pos = nx.spring_layout(graph, k=k, iterations=20)

    # 迭代改进布局
    for _ in range(num_steps):
        pos = nx.spring_layout(graph, pos=pos, k=k, iterations=20)

    # 层级间的垂直间距
    vertical_spacing = 0.1

    # 获取每个层级的节点列表
    layers = {}
    for node, data in graph.nodes(data=True):
        layers.setdefault(data["level"], []).append(node)

    # 调整同一层级的节点使其更靠近，同时确保层级间有足够间距
    for level, nodes in layers.items():
        # 计算该层级的水平位置
        x_positions = [pos[node][0] for node in nodes]
        x_positions.sort()
        min_x, max_x = min(x_positions), max(x_positions)

        # 重新分布该层级的节点
        for i, node in enumerate(nodes):
            new_x = min_x + (max_x - min_x) * i / (
                len(nodes) - 1 if len(nodes) > 1 else 1
            )
            new_y = level * vertical_spacing
            pos[node] = (new_x, new_y)

    return pos


def approximate_bezier_curve(points, num_points=10):
    # 近似贝塞尔曲线
    t_values = np.linspace(0, 1, num_points)
    curve_points = []

    if len(points) == 3: # 二次贝塞尔曲线
        # 使用二次贝塞尔曲线公式
        P0, P1, P2 = points
        for t in t_values:
            point = (1-t)**2 * P0 + 2 * (1-t) * t * P1 + t**2 * P2
            curve_points.append(point)
    elif len(points) == 4: # 立方贝塞尔曲线
        # 使用立方贝塞尔曲线公式
        P0, P1, P2, P3 = points
        for t in t_values:
            point = (1-t)**3 * P0 + 3 * (1-t)**2 * t * P1 + 3 * (1-t) * t**2 * P2 + t**3 * P3
            curve_points.append(point)
    return np.array(curve_points)


def get_path_bbox(d_attribute):
    Pcode, Pnums = parse_path_d_attribute(d_attribute)
    all_points = []

    for command, params in zip(Pcode, Pnums):
        params = [float(p) for p in params]

        if command == 'Q':
            # 二次贝塞尔曲线
            control_points = np.array(params).reshape(-1, 2)
            curve_points = approximate_bezier_curve(control_points)
            all_points.extend(curve_points)
        elif command == 'C':
            # 立方贝塞尔曲线
            control_points = np.array(params).reshape(-1, 2)
            curve_points = approximate_bezier_curve(control_points)
            all_points.extend(curve_points)
        # 其他命令的处理...
        
     # 检查是否有计算出的点
    if not all_points:
        return None  # 或者返回一个特定的空边界框表示

    # 计算边界框
    all_points = np.array(all_points)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    bbox = np.array([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]])
    return bbox


def get_element_bbox(element, parent_transform=np.identity(3)):
    tag = element.tag.split('}')[-1]
    bbox = None

    if tag == 'rect':
        x, y, width, height = map(float, [element.get('x', 0), element.get('y', 0), element.get('width', 0), element.get('height', 0)])
        bbox = np.array([[x, y], [x + width, y], [x, y + height], [x + width, y + height]])

    elif tag == 'circle':
        cx, cy, r = map(float, [element.get('cx', 0), element.get('cy', 0), element.get('r', 0)])
        bbox = np.array([[cx - r, cy - r], [cx + r, cy - r], [cx - r, cy + r], [cx + r, cy + r]])
        
    elif tag == 'path':
        # 处理路径
        d_attribute = element.get('d', '')
        bbox = get_path_bbox(d_attribute)

    # 其他元素类型的处理...

    # 应用transform
    if bbox is not None:
        transform = np.identity(3)
        # 处理transform属性...
        total_transform = np.dot(parent_transform, transform)
        bbox = np.dot(np.hstack((bbox, np.ones((bbox.shape[0], 1)))), total_transform.T)[:,:2]
    
    return bbox


def visualize_graph(graph, pos):
    # 创建不同类型边的集合
    edge_traces = []

    # 分别处理不同类型的边
    edge_types = set(
        (data.get("style", "solid"), data.get("color", "grey"))
        for _, _, data in graph.edges(data=True)
    )
    for style, color in edge_types:
        edge_x = []
        edge_y = []

        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_data = edge[2]
            if edge_data.get("style", "solid") == style and edge_data.get("color", "grey") == color:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.6, color=color),
            hoverinfo="none",
            mode="lines",
            line_shape="spline" if style == "dashed" else "linear",
        )
        edge_traces.append(trace)

    # 处理节点的样式
    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_shape = []
    node_text = []  # 节点名称文本
    node_hover_text = []  # 悬停信息文本

    # 计算最大图层深度
    max_layer = max(int(data["layer"]) for _, data in graph.nodes(data=True))

    # 设定最大最小节点尺寸
    min_size = 10  # 最小节点尺寸
    max_size = 25  # 最大节点尺寸

    # 计算尺寸变化率
    size_rate = (max_size - min_size) / (1 + max_layer)

    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # 从节点的完整标签中提取基本标签（忽略编号）
        base_tag = attrs["tag"].split("_")[0]
        
        # 解析图层信息以计算大小
        layer_info = attrs['layer']
        layer_numbers = [int(n) for n in layer_info.split('_') if n.isdigit()]
        layer_depth = len(layer_numbers)  # 可以使用 sum(layer_numbers) 或其他逻辑

        # 根据基本标签决定形状和颜色
        if base_tag in ["rect"]:
            shape = "square"
        elif base_tag in ["circle"]:
            shape = "circle"
        elif base_tag in ["line", "path"]:
            shape = "circle-open"
        else:
            shape = "circle"

        if not attrs["visible"]:
            shape = "circle"
            color = "lightgrey"  # 不可见元素的颜色
            size = min_size
        else:
            # 可见元素的颜色和大小
            level = attrs["level"]
            color = colorsys.hsv_to_rgb(0.3 * level, 1.0, 1.0)
            color = f"rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})"
            
            # 解析图层信息以计算大小
            layer_info = attrs['layer']
            layer_numbers = [int(n) for n in layer_info.split('_') if n.isdigit()]
            layer_depth = len(layer_numbers)  # 可以使用 sum(layer_numbers) 或其他逻辑
            
             # 计算节点大小
            size = max(max_size - layer_depth * size_rate, min_size)  # 确保大小不小于最小值


        node_size.append(size)
        node_color.append(color)
        node_shape.append(shape)

        # 添加节点标签
        node_text.append(attrs["tag"])

        # 特别处理 text 节点，显示 text_content
        if attrs["tag"] == "text":
            hover_text = attrs.get("text_content", "")
        else:
            hover_text = f"Tag: {attrs['tag']}\n"
            hover_text += "\n".join(
                f"{key}: {val}" for key, val in attrs["attributes"].items()
            )
        node_hover_text.append(hover_text)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",  # 在节点旁添加文本
        text=node_text,  # 节点名称
        hoverinfo="text",  # 悬停显示文本
        hovertext=node_hover_text,  # 悬停信息
        marker=dict(
            size=node_size,
            color=node_color,
            symbol=node_shape,  # 应用node_shape数组
        ),
        textposition="top center",  # 文本位置
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()


if __name__ == "__main__":
    svg_file_path = "./image.svg"
    svg_root = parse_svg(svg_file_path)
    graph = build_graph(svg_root)
    # pos = compute_layout_with_progress(graph)
    # visualize_graph(graph, pos)

# 将输出写入文件
    with open("print.json", "w") as file:
        # 打印并写入图的信息
        file.write(str(svg_file_path).split('/')[1] + " " + str(graph) + "\n\n")

        # 遍历节点并写入信息
        for node in graph.nodes(data=True):
            file.write("Node ID: " + str(node[0]) + "\n")
            file.write("Attributes: " + str(node[1]) + "\n\n")

        # 遍历边并写入信息
        for edge in graph.edges(data=True):
            file.write("Edge: " + str(edge) + "\n\n")
            
        for element in svg_root.iter():
            bbox = get_element_bbox(element)
            if bbox is not None:
                tag = element.tag.split('}')[-1]
                file.write("Location: "+ str(tag) + "\n" + str(bbox) + "\n\n")
