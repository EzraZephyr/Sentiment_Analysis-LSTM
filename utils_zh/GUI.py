import tkinter as tk
from tkinter import messagebox
from utils_zh.predict_review import *
from googletrans import Translator

def GUI():

    def click_submit():
    # 当用户点击Submit按钮时 调用此函数
        review = text.get("1.0", tk.END).strip()
        # 获取用户输入的文本内容 从第一行第0个字符开始到结尾全部捕获

        translator = Translator()
        review = translator.translate(review, src='zh-cn', dest='en').text
        # 根据Google的API进行中英转译

        result = predict_input(model, word_to_index, review)
        # 调用自己设定的那个predict_input获取结果

        messagebox.showinfo("情感结果", result)
        # 将获取的结果显示在弹窗中


    model, word_to_index = load_model()
    # 加载模型和词汇表

    root = tk.Tk()
    root.title("影评分析")
    root.geometry("300x250+575+300")
    # 创建主窗口 并且设定title、大小和距离左上角的位置

    text_label = tk.Label(root, text="请输入电影评价:")
    text_label.pack(pady=10)
    # 创建一个标签 提示的一段文本内容 父亲为上面设置的主窗口root
    # 并且设置其垂直边距为10个像素
    # 也可以直接在Label(root, text="Input your review:")后面
    # 添加.pack 但是不推荐 因为这样的话会使其返回'None' 这样以后就
    # 无法处理和调用这个部件了 所以像我这样分开处理是非常推荐的
    # 看着也工整

    text = tk.Text(root, width=50, height=10)
    text.pack(padx=20, pady=10)
    # 创建一个多行文本输入框 宽为50 高为10
    # 水平和垂直边距分别为20和10

    submit_button = tk.Button(root, text="提交", command=click_submit)
    submit_button.pack(pady=10)
    # 创建一个按钮 当点击按钮时调用click_submit函数

    root.mainloop()
    # 启动Tkinter主事件循环 设定的这个窗口开始运行并等待交互
