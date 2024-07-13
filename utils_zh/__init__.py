import nltk

nltk.download('wordnet')
# 下载wordnet数据包 用于词形还原

nltk.download('stopwords')
# 下载停用词数据包 过滤没用的词汇

# 注意 如果下载完之后运行还是提示无法找到这两个数据包的话
# 可以看一下C盘-用户-你的名字-然后打开隐藏文件AppData
# 然后找到Roaming-nltk_data-corpora 找到这两个数据包
# 的压缩zip形式 然后解压到corpora里 就可以了
