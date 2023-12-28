import modules.CreateGM_V4  as CreateGM
import os

# 设定SVG文件所在的目录
directory = "./TEST"

# svg_parser = CreateGM.SVGParser("./TEST/testSVG.svg")
# svg_parser.run()

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".svg"):  # 检查文件扩展名
        svg_file_path = os.path.join(directory, filename)  # 获取完整的文件路径
        print(f"Processing {svg_file_path}...")  # 打印当前处理的文件

        # 创建SVGParser对象并运行
        svg_parser = CreateGM.SVGParser(svg_file_path)
        svg_parser.run()

print("All SVG files processed.")