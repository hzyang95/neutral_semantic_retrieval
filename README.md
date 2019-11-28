# NLP项目模板

```
|-- conf                    # 配置文件所在目录
|-- data                    # 数据所在目录，不上传
|-- docs                    # 文档目录
|-- nlpstarter               # 核心代码目录
|-- requirements.txt        # 依赖包说明文件
|-- scripts                 # 运行脚本，数据处理脚本等
|-- tests                   # 测试文件路径
|-- third_party             # 第三方库及子模块路径
|-- .gitignore              # 记录不上传至git的文件，例如.pyc
|-- .flake8                 # 记录代码风格规则
```

## 0. 安装

```
$ git clone https://gitlab.yunfutech.com/yunfu/starter/nlpstarter.git
$ cd nlpstarter

$ pip install -r requirements.txt
```

## 1. 配置文件修改

```
conf/conf.yaml
```

## 2. 服务启动
```
cd scripts
sh run_train.sh  # 训练
sh run_serve.sh  # 启动服务

测试服务：curl -H "Content-Type:application/json" -X POST -d '{"text": "战斗机"}' localhost:12345/predict
```

## 3. 单元测试

```
pytest ./tests/
```

## 4. 并发测试
```
cd tests/ab_pref_test
run_ab_pref_test.sh
```