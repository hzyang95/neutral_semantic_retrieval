# neutral_semantic_retrieval

```
|-- conf                    # 配置文件所在目录
|-- data                    # 数据
|-- docs                    # 文档目录
|-- neu_sem_retrieval       # 核心代码目录
|-- requirements.txt        # 依赖包说明文件
|-- scripts                 # 运行脚本，数据处理脚本等
|-- tests                   # 测试文件路径
|-- third_party             # 第三方库及子模块路径
|-- .gitignore              # 记录不上传至git的文件，例如.pyc
|-- .flake8                 # 记录代码风格规则
```

## 0. 安装

```
$ git clone https://github.com/hzyang95/neutral_semantic_retrieval.git
$ cd neutral_semantic_retrieval

$ pip install -r requirements.txt
```

## 1. 配置文件修改

```
conf/conf.yaml
```

## 2. 单元测试

```
pytest ./tests/
```
