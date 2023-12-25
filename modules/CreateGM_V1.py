import xml.etree.ElementTree as ET
import networkx as nx
import plotly.graph_objects as go
import colorsys
from tqdm import tqdm

def parse_svg(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def extract_element_info(element):
    tag_with_namespace = element.tag
    tag_without_namespace = tag_with_namespace.split('}')[1] if '}' in tag_with_namespace else tag_with_namespace
    attributes = element.attrib
    text_content = element.text
    return tag_without_namespace, attributes, text_content

def parse_path_d_attribute(d_attribute):  # 对path的d元素属性值进行特殊解析（目前有时会失灵）
    path_data = d_attribute.strip()
    parsed_path_data = []

    commands = path_data.split()
    
    for command in commands:
        code = command[0]
        params = command[1:].split(',')
        parsed_path_data.append({'code': code, 'params': params})

    return parsed_path_data

def build_graph(svg_root, parent_path="svg"):
    G = nx.Graph()

    def add_element_to_graph(element, parent_path="svg", level=0):
        tag, attributes, text_content = extract_element_info(element)
        element_id = f"{tag}_{attributes.get('id', '')}"
        node_id = f"{parent_path}/{element_id}"
    
        # 如果节点已经存在，则添加一个计数器以区分实例
        counter = 1
        while G.has_node(node_id):
            counter += 1
            node_id = f"{parent_path}/{element_id}_{counter}"
    
        G.add_node(node_id, tag=tag, attributes=attributes, text_content=text_content, level=level)
    
        if parent_path != "svg":
            G.add_edge(parent_path, node_id)
    
        if tag == 'path':
            path_data = attributes.get('d', '')
            parsed_path_data = parse_path_d_attribute(path_data)
            G.nodes[node_id]['path_data'] = parsed_path_data
    
        # 添加 tqdm 以显示进度条
        for i, child in enumerate(element):
            add_element_to_graph(child, parent_path=f"{parent_path}/{element_id}", level=level + 1)

    add_element_to_graph(svg_root)

    return G

def compute_layout_with_progress(graph, num_steps=100):
    pos = nx.spring_layout(graph, seed=42)  # 无进度条的初始布局

    # 迭代步骤并显示进度条
    for _ in tqdm(range(num_steps), desc="Layout Progress", unit="step"):
        pos = nx.spring_layout(graph, seed=42, pos=pos, iterations=1)

    return pos

def get_node_color(level):
    # 根据级别生成不同的颜色
    hue = (level * 0.1) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
    color = f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
    return color

def visualize_graph(graph, pos):
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    labels = []
    hover_text = []
    node_colors = []

    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        tag = attrs['tag']
        attributes = attrs['attributes']
        level = attrs['level']

        node_x.append(x)
        node_y.append(y)
        labels.append(f"{attributes}")
        hover_text.append(f"Tag: {tag}<br>Attributes: {attributes}")
        node_colors.append(get_node_color(level))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=labels,
        customdata=hover_text,
        marker=dict(size=10, color=node_colors)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # Add text annotations for node labels
    for node, (x, y) in pos.items():
        fig.add_annotation(
            text=node,
            x=x, y=y,
            xref="x", yref="y",
            font=dict(color='black', size=8),
            showarrow=False,
            align='center'
        )

    fig.update_traces(marker=dict(size=10, line=dict(color='rgb(255,255,255)', width=2)))

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    fig.show()

if __name__ == "__main__":
    svg_file_path = "./map.svg"
    svg_root = parse_svg(svg_file_path)
    graph = build_graph(svg_root)

    pos = compute_layout_with_progress(graph)

    for node, (x, y) in pos.items():
        graph.nodes[node]['pos'] = [x, y]
        
    print(graph)

    visualize_graph(graph, pos)
    
    for node in graph.nodes(data=True):
        print("Node ID:", node[0])
        print("Attributes:", node[1])
        print("\n")

    for edge in graph.edges(data=True):
        print("Edge:", edge)
        print("\n")