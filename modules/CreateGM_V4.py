import xml.etree.ElementTree as ET
import networkx as nx
import plotly.graph_objects as go
import colorsys
import numpy as np
import re
import json

class SVGParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.graph = nx.DiGraph()
        self.existing_tags = {}


    @staticmethod
    def parse_svg(file_path):
        tree = ET.parse(file_path)
        return tree.getroot()

    @staticmethod
    def default_attributes(tag):
        default_attrs = {
            "rect": {"width": "0", "height": "0", "x": "0", "y": "0", "fill": "black"},
            "circle": {"cx": "0", "cy": "0", "r": "0", "fill": "black"},
            "ellipse": {"cx": "0", "cy": "0", "rx": "0", "ry": "0", "fill": "black"},
            "line": {"x1": "0", "y1": "0", "x2": "0", "y2": "0", "stroke": "black"},
            "polyline": {"points": "", "stroke": "black"},
            "polygon": {"points": "", "fill": "black"},
            "path": {"d": "", "fill": "none"},
            "text": {"x": "0", "y": "0", "fill": "black"},
            "image": {"x": "0", "y": "0", "width": "0", "height": "0"},
            # 添加更多元素类型及其默认属性
        }
        return default_attrs.get(tag, {})

    @staticmethod
    def get_coordinate_attributes(element, tag):
        # 定义各种元素类型的坐标属性
        coordinate_attrs = {
            "circle": ["cx", "cy", "r"],
            "ellipse": ["cx", "cy", "rx", "ry"],
            "rect": ["x", "y", "width", "height"],
            "line": ["x1", "y1", "x2", "y2"],
            "polyline": ["points"],
            "polygon": ["points"],
            "text": ["x", "y"],
            "image": ["x", "y", "width", "height"],
            # 添加其他元素类型
        }

        # 获取元素类型的坐标属性列表
        attrs_list = coordinate_attrs.get(tag, [])

        # 提取坐标属性
        coordinates = {}
        for attr in attrs_list:
            value = element.get(attr, "0")  # 使用 "0" 作为默认值
            coordinates[attr] = value

        return coordinates

    @staticmethod
    def extract_element_info(element, existing_tags):
        tag_with_namespace = element.tag
        tag_without_namespace = tag_with_namespace.split("}")[-1]
    
        if tag_without_namespace != "svg":
            count = existing_tags.get(tag_without_namespace, 0)
            full_tag = (
                f"{tag_without_namespace}_{count}"
                if count > 0
                else tag_without_namespace
            )
            existing_tags[tag_without_namespace] = count + 1
        else:
            full_tag = tag_without_namespace

        attributes = element.attrib
        text_content = element.text.strip() if element.text else None

        # 应用默认属性
        default_attrs = SVGParser.default_attributes(tag_without_namespace)
        attributes = element.attrib.copy()  # 复制原始属性
        for key, value in default_attrs.items():
            attributes.setdefault(key, value)  # 如果属性未在元素中定义，则使用默认值
        
        # 对d属性进行特殊处理
        if tag_without_namespace == "path":
            path_data = attributes.get("d", "")
            Pcode, Pnums = SVGParser.parse_path_d_attribute(path_data)
            attributes["Pcode"] = Pcode
            attributes["Pnums"] = Pnums
            
        # 获取元素类型的坐标属性
        coordinates = SVGParser.get_coordinate_attributes(element, tag_without_namespace)

        # 添加坐标属性到元素属性字典中
        attributes.update(coordinates)
            
        text_content = element.text.strip() if element.text else None
        return full_tag, attributes, text_content


    @staticmethod
    def parse_path_d_attribute(d_attribute):
        path_commands = re.findall(r"([a-zA-Z])([^a-zA-Z]*)", d_attribute)
        Pcode, Pnums = [], []

        for command, params in path_commands:
            Pcode.append(command)
            params_list = [
                param.strip() for param in re.split(r"[ ,]+", params.strip()) if param
            ]
            Pnums.append(params_list)

        return Pcode, Pnums


    @staticmethod
    def approximate_bezier_curve(points, num_points=10):
        t_values = np.linspace(0, 1, num_points)
        curve_points = []

        if len(points) == 3:
            P0, P1, P2 = points
            for t in t_values:
                point = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
                curve_points.append(point)
        elif len(points) == 4:
            P0, P1, P2, P3 = points
            for t in t_values:
                point = (
                    (1 - t) ** 3 * P0
                    + 3 * (1 - t) ** 2 * t * P1
                    + 3 * (1 - t) * t**2 * P2
                    + t**3 * P3
                )
                curve_points.append(point)

        return np.array(curve_points)


    @staticmethod
    def get_path_points(d_attribute):
        Pcode, Pnums = SVGParser.parse_path_d_attribute(d_attribute)
        path_points = []

        for command, params in zip(Pcode, Pnums):
            params = [float(p) for p in params]

            # 处理移动命令
            if command == "M":
                current_point = np.array(params).reshape(-1, 2)[0]
                path_points.append(current_point)

            # 处理直线命令
            elif command == "L":
                for i in range(0, len(params), 2):
                    line_point = np.array(params[i:i+2])
                    path_points.append(line_point)

            # 处理二次贝塞尔曲线
            elif command == "Q":
                control_points = np.array(params).reshape(-1, 2)
                curve_points = SVGParser.approximate_bezier_curve(control_points, num_points=20) # 增加拟合点的数量
                path_points.extend(curve_points)

            # 处理三次贝塞尔曲线
            elif command == "C":
                control_points = np.array(params).reshape(-1, 2)
                curve_points = SVGParser.approximate_bezier_curve(control_points, num_points=20) # 增加拟合点的数量
                path_points.extend(curve_points)

            # 处理路径关闭命令
            elif command == "Z":
                # 如果需要闭合路径，可以考虑添加起始点（M命令的点）到path_points，或者保持当前状态
                pass

        return np.array(path_points)


    def apply_transform(self, bbox, transform):
        # 转换 transform 字典为变换矩阵
        transform_matrix = self.transform_to_matrix(transform)

        # 应用变换矩阵到定界框的每个点
        transformed_bbox = []
        for point in bbox:
            # 转换点为齐次坐标 (x, y, 1)
            point_homogeneous = np.append(point, 1)
            # 应用变换
            transformed_point = np.dot(transform_matrix, point_homogeneous)
            # 变回2D坐标
            transformed_bbox.append(transformed_point[:2])

        return np.array(transformed_bbox)

    
    def transform_to_matrix(self, transform):
        # 创建初始变换矩阵
        transform_matrix = np.identity(3)

        # 应用平移
        transform_matrix[0, 2] = transform["translate"][0]
        transform_matrix[1, 2] = transform["translate"][1]

        # 应用旋转
        angle = np.radians(transform["rotate"])
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        transform_matrix = np.dot(transform_matrix, rotation_matrix)

        # 应用缩放
        scale_matrix = np.array([
            [transform["scale"][0], 0, 0],
            [0, transform["scale"][1], 0],
            [0, 0, 1]
        ])
        transform_matrix = np.dot(transform_matrix, scale_matrix)

        return transform_matrix

    def convert_to_float(self, value):
        try:
            return float(value)
        except ValueError:
            num_part = re.match(r"([0-9\.]+)", value)
            return float(num_part.group(1)) if num_part else 0.0
        
    def get_element_bbox(self, element, parent_transform=np.identity(3)):
        tag = element.tag.split('}')[-1]
        bbox = None

        # 处理不同类型的 SVG 元素以获取其原始定界框
        if tag == "rect":
            x, y, width, height = map(
                self.convert_to_float,
                [
                    element.get("x", 0),
                    element.get("y", 0),
                    element.get("width", 0),
                    element.get("height", 0),
                ],
            )
            bbox = np.array([[x, x+width], [y, y+height], [x+width/2, y+height/2]])

        elif tag == "circle":
            cx, cy, r = map(self.convert_to_float, [element.get("cx", 0), element.get("cy", 0), element.get("r", 0)])
            bbox = np.array([[cx-r, cx+r], [cy-r,cy+r], [cx,cy]])

        elif tag == "path":
            d_attribute = element.get("d", "")
            bbox = SVGParser.get_path_points(d_attribute)

        # 处理线段
        elif tag == "line":
            x1, y1, x2, y2 = map(self.convert_to_float, [element.get("x1", 0), element.get("y1", 0), 
                                         element.get("x2", 0), element.get("y2", 0)])
            bbox = np.array([[x1, x2], [y1, y2], [(x1+x2)/2, (y1+y2)/2]])

        # 处理椭圆
        elif tag == "ellipse":
            cx, cy, rx, ry = map(self.convert_to_float, [element.get("cx", 0), element.get("cy", 0), 
                                         element.get("rx", 0), element.get("ry", 0)])
            bbox = np.array([[cx-rx, cx+rx], [cy-ry,cy+ry], [cx,cy]])

        # 处理多边形和折线元素
        elif tag in ["polygon", "polyline"]:
            points = element.get("points", "").strip()
            if points:
                points_array = np.array([[self.convert_to_float(n) for n in point.split(",")] for point in points.split(" ") if point.strip()])
                min_x, min_y = np.min(points_array, axis=0)
                max_x, max_y = np.max(points_array, axis=0)
                bbox = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        elif tag == "text":
            x, y = map(self.convert_to_float, [element.get("x", 0), element.get("y", 0)])
            # 这里假设一个默认的宽度和高度，因为无法精确计算
            width, height = 100, 20  # 默认值，可根据需要调整
            bbox = np.array([[x, y], [x + width, y], [x, y + height], [x + width, y + height]])

        elif tag == "image":
            x, y, width, height = map(self.convert_to_float, [element.get("x", 0), element.get("y", 0),
                                             element.get("width", 0), element.get("height", 0)])
            bbox = np.array([[x, y], [x + width, y], [x, y + height], [x + width, y + height]])

        # 应用解析后的 transform 到定界框
        if bbox is not None and element.get("transform"):
            # 解析元素自身的 transform 属性
            element_transform = self.parse_transform(element.get("transform", ""))

            # 应用解析后的 transform 到定界框
            bbox = self.apply_transform(bbox, element_transform)

        return bbox

    # 添加一个方法来解析 fill 属性
    def parse_fill_attribute(element, inherited_attrs):
        fill = element.get('fill')
        if fill is None and 'fill' in inherited_attrs:
            fill = inherited_attrs['fill']  # 继承父元素的 fill
        return fill if fill is not None else 'black'  # 默认值为黑色

    @staticmethod
    def parse_transform(transform_str):
        transform_dict = {"translate": [0.0, 0.0], "rotate": 0.0, "scale": [1.0, 1.0]}

        translate_match = re.search(r"translate\(([\d\.\-]+)[ ,]*([\d\.\-]+)\)", transform_str)
        rotate_match = re.search(r"rotate\(([\d\.\-]+)\)", transform_str)
        scale_match = re.search(r"scale\(([\d\.\-]+)(?:[ ,]*([\d\.\-]+))?\)", transform_str)

        if translate_match:
            transform_dict["translate"] = [float(translate_match.group(1)), float(translate_match.group(2))]
        
        if rotate_match:
            transform_dict["rotate"] = float(rotate_match.group(1))
        
        if scale_match:
            x_scale = float(scale_match.group(1))
            y_scale = float(scale_match.group(2)) if scale_match.group(2) else x_scale
            transform_dict["scale"] = [x_scale, y_scale]

        return transform_dict


    def build_graph(self, svg_root):
    
        def add_element_to_graph(
            element,
            parent_path='svg',
            level=0,
            layer="0",
            inherited_attrs={},
            layer_counter=0,
        ):
            tag, attributes, text_content = SVGParser.extract_element_info(
                element, self.existing_tags
            )

            combined_attributes = {**attributes, **inherited_attrs}
            # 合并 transform 属性
            if "transform" in attributes or "transform" in inherited_attrs:
                inherited_transform = self.parse_transform(inherited_attrs.get('transform', ''))
                own_transform = self.parse_transform(attributes.get('transform', ''))

                # 计算总的 transform
                total_transform = {
                    "translate": [sum(x) for x in zip(inherited_transform["translate"], own_transform["translate"])],
                    "rotate": inherited_transform["rotate"] + own_transform["rotate"],
                    "scale": [inherited_transform["scale"][0] * own_transform["scale"][0], inherited_transform["scale"][1] * own_transform["scale"][1]]
                }

                combined_attributes['transform'] = f"translate({total_transform['translate'][0]}, {total_transform['translate'][1]}) rotate({total_transform['rotate']}) scale({total_transform['scale'][0]}, {total_transform['scale'][1]})"

            # 计算bbox并添加到combined_attributes
            bbox = self.get_element_bbox(element, combined_attributes)
            if bbox is not None:
                combined_attributes['bbox'] = bbox.tolist()  # 转换为列表
            node_id = f"{parent_path}/{tag}" if parent_path else tag
            is_visible = tag.split("_")[0] not in [
                "svg",
                "g",
                "defs",
                "clipPath",
                "mask",
                "pattern",
                "marker",
                "style",
            ]
            self.graph.add_node(
                node_id,
                tag=tag,
                attributes=combined_attributes,
                text_content=text_content,
                level=level,
                layer=layer,
                visible=is_visible,
            )

            if parent_path and parent_path != "svg":
                self.graph.add_edge(parent_path, node_id)

            previous_sibling_id = None
            new_layer_counter = layer_counter
            for child in reversed(element):
                child_layer = f"{layer}_{new_layer_counter}"
                child_id = add_element_to_graph(
                    child,
                    parent_path=node_id,
                    level=level + 1,
                    layer=child_layer,
                    inherited_attrs=combined_attributes,
                    layer_counter=new_layer_counter,
                )
                if previous_sibling_id:
                    self.graph.add_edge(
                        previous_sibling_id, child_id, color="blue", style="solid"
                    )
                previous_sibling_id = child_id
                new_layer_counter += 1

            return node_id

        add_element_to_graph(svg_root)


    @staticmethod
    def compute_layout_with_progress(graph, num_steps=100, k=None):
        if k is None:
            k = 1 / np.sqrt(len(graph.nodes()))
        pos = nx.spring_layout(graph, k=k, iterations=20)
        for _ in range(num_steps):
            pos = nx.spring_layout(graph, pos=pos, k=k, iterations=20)

        vertical_spacing = 0.1
        layers = {}
        for node, data in graph.nodes(data=True):
            layers.setdefault(data["level"], []).append(node)

        for level, nodes in layers.items():
            x_positions = [pos[node][0] for node in nodes]
            x_positions.sort()
            min_x, max_x = min(x_positions), max(x_positions)
            for i, node in enumerate(nodes):
                new_x = min_x + (max_x - min_x) * i / (
                    len(nodes) - 1 if len(nodes) > 1 else 1
                )
                new_y = level * vertical_spacing
                pos[node] = (new_x, new_y)

        return pos


    @staticmethod
    def visualize_graph(graph, pos):
        edge_traces = []
        edge_types = set(
            (data.get("style", "solid"), data.get("color", "grey"))
            for _, _, data in graph.edges(data=True)
        )
        for style, color in edge_types:
            edge_x, edge_y = [], []
            for edge in graph.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                if (
                    edge[2].get("style", "solid") == style
                    and edge[2].get("color", "grey") == color
                ):
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=0.6, color=color),
                    hoverinfo="none",
                    mode="lines",
                    line_shape="spline" if style == "dashed" else "linear",
                )
            )

        (
            node_x,
            node_y,
            node_size,
            node_color,
            node_shape,
            node_text,
            node_hover_text,
        ) = [], [], [], [], [], [], []
        max_layer = max(int(data["layer"]) for _, data in graph.nodes(data=True))
        min_size, max_size = 10, 25
        size_rate = (max_size - min_size) / (1 + max_layer)

        for node, attrs in graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            base_tag = attrs["tag"].split("_")[0]
            layer_info = attrs["layer"]
            layer_depth = len([int(n) for n in layer_info.split("_") if n.isdigit()])

            shape = "circle"  # Default shape
            if base_tag == "rect":
                shape = "square"
            elif base_tag == "circle":
                shape = "circle"
            elif base_tag in ["line", "path"]:
                shape = "circle-open"

            if not attrs["visible"]:
                color = "lightgrey"
                size = min_size
            else:
                hsv_color = colorsys.hsv_to_rgb(0.3 * attrs["level"], 1.0, 1.0)
                color = f"rgb({int(hsv_color[0] * 255)}, {int(hsv_color[1] * 255)}, {int(hsv_color[2] * 255)})"
                size = max_size - layer_depth * size_rate
                size = max(size, min_size)

            node_size.append(size)
            node_color.append(color)
            node_shape.append(shape)
            node_text.append(attrs["tag"])

            hover_text = (
                attrs.get("text_content", "")
                if attrs["tag"] == "text"
                else f"Tag: {attrs['tag']}\n"
                + "\n".join(f"{key}: {val}" for key, val in attrs["attributes"].items())
            )
            node_hover_text.append(hover_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            hoverinfo="text",
            hovertext=node_hover_text,
            marker=dict(size=node_size, color=node_color, symbol=node_shape),
            textposition="top center",
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


    def run(self):
        svg_root = SVGParser.parse_svg(self.file_path)
        self.build_graph(svg_root)
        pos = SVGParser.compute_layout_with_progress(self.graph)
        SVGParser.visualize_graph(self.graph, pos)
        self.write_output()


    def write_output(self):
        output = {
            "DiGraph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "Nodes": {},
                "Edges": []
            }
        }

        # 填充节点数据
        for node, data in self.graph.nodes(data=True):
            node_id = str(node)
            # 格式化attributes为所需的结构
            attributes = {
                "tag": data.get("tag", ""),
                "attributes": data.get("attributes", {}),
                "text_content": data.get("text_content", ""),
                "level": data.get("level", 0),
                "layer": data.get("layer", ""),
                "visible": data.get("visible", True),
            }
            output["DiGraph"]["Nodes"][node_id] = {"Attributes": attributes}

        # 填充边缘数据
        for u, v, data in self.graph.edges(data=True):
            output["DiGraph"]["Edges"].append((u, v, data))

        # 使用json模块写入文件
        with open("GMinfo.json", "w") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)
 
