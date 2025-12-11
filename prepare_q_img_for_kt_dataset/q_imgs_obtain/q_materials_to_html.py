import re
import os


# 安全地替换所有占位符 <key> 为 <img> 标签
def replace_image_placeholders(text, mapping):
    def replacer(match):
        key = match.group(1)
        if key in mapping:
            return f'<img src="{mapping[key]}" alt="{key}">'
        else:
            # 如果找不到，保留原样或报错，这里选择保留
            return match.group(0)
    # 匹配 <xxx> 形式的占位符，但避免匹配HTML标签（这里假设占位符不含斜杠）
    return re.sub(r"<([a-zA-Z0-9_-]+)>", replacer, text)


def text2html(text="xxxx<image0>xxxx", imgname2path={"image0":"image0.png"}, output_file="output.html"):
    text = text.replace("\n", "<br />")
    # 执行替换
    replaced_text = replace_image_placeholders(text, imgname2path)

    # 构建完整 HTML
    html_template = f"""<!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; 
            font-size: 16px;
            line-height: 1.6; 
            padding: 20px; }}
            img {{ max-width: 100%; height: auto; vertical-align: middle; }}
        </style>
    </head>
    <body>
        <div>{replaced_text}</div>
    </body>
    </html>
    """
    if type(output_file) == str and output_file.endswith(".html"):
        dir_saved = os.path.dirname(output_file)
        os.makedirs(dir_saved, exist_ok=True)
        # 写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)

    return html_template


if __name__ == "__main__":
    text = "xxxx\nxxxx\nadad\ncddad66"
    imgname2path = {}
    output_file = "./files_temp/output.html"
    text2html(text, imgname2path, output_file)