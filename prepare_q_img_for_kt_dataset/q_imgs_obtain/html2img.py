import imgkit
# sudo apt-get install wkhtmltopdf
# pip install imgkit -i https://pypi.tuna.tsinghua.edu.cn/simple


def html_to_image_imgkit(html_content, output_path, options=None, path_wkhtmltoimage=None):
    """
    使用imgkit将HTML转换为图片
    
    Args:
        html_content: HTML内容字符串或HTML文件路径
        output_path: 输出图片路径
        options: 转换选项字典
        path_wkhtmltoimage: wkhtmltoimage可执行文件路径（可选）
    """
    # 默认配置
    if options is None:
        options = {
            'format': 'jpg',
            'width': 512,  # 图片宽度
            'height': 512,   # 图片高度
            'quality': 100,  # 图片质量
            'enable-local-file-access': None  # 允许访问本地文件
        }
    wkhtmltoimage = path_wkhtmltoimage if path_wkhtmltoimage else '/usr/bin/wkhtmltoimage'
    # 配置wkhtmltoimage路径（如果需要）
    config = imgkit.config(wkhtmltoimage=wkhtmltoimage)  # Linux/Mac
    # config = imgkit.config(wkhtmltoimage=r'xxx\wkhtmltopdf\bin\wkhtmltoimage.exe')  # Windows
    
    # 转换HTML为图片
    if html_content.endswith('.html'):    # if html_content is a file path
        imgkit.from_file(html_content, output_path, options=options, config=config)
    else:    # if html_content is a string
        imgkit.from_string(html_content, output_path, options=options, config=config)


if __name__ == "__main__":
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f0f0f0; }
            .container { width: 800px; margin: 0 auto; padding: 20px; background: white; }
            h1 { color: #333; text-align: center; }
            p { line-height: 1.6; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>HTML转图片示例</h1>
            <p>这是一个将HTML转换为图片的示例。</p>
            <ul>
                <li>支持CSS样式</li>
                <li>支持JavaScript</li>
                <li>高质量输出</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    html_to_image_imgkit(html_content, ".files_temp/output.png")