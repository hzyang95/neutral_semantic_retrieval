## 代码规范
- 使用Python3作为主力编程语言，如无特殊原因，不要使用其它语言或Python2
- 使用google python style作为代码规范
- 使用Miniconda管理python环境

### 1 代码风格规范
- https://github.com/google/styleguide
- [Google 开源项目风格指南 (中文版)](https://github.com/zh-google-styleguide/zh-google-styleguide)
- 必看文档 http://zh-google-styleguide.readthedocs.io/en/latest/ 

#### 1.2 flake8
- 由Python官方发布的一款辅助检测Python代码是否规范的工具。
- 对下面三个工具的封装：
  - PyFlakes：静态检查Python代码逻辑错误的工具。
  - pep8： 静态检查PEP 8编码风格的工具。
  - Ned Batchelder’s McCabe script：静态分析Python代码复杂度的工具。

#### 1.3 YAPF
- 使用YAPF作为代码自动格式化工具

#### 1.4 命名规范
- 包/命名空间命名
  - mod_submod  # 全部小写，以_间隔
- 类命名
  - ThisClassName  # 每个单词都以大写字母开头
- 函数/方法命名
  - the_method_name  # 全小写单词，以下划线分隔
- 常量命名
  - MAX_VALUE # 全大写，以下划线分隔

#### 1.5 下划线命名约定
规则 | 作用
---- | ---
_xxx | 表示internal，内部使用，不能被from abc import *导入
xxx_ | 表示避免和关键字冲突，如lambda_
__xxx | 更彻底的private，用到了name mangling，会自动加上类名前缀，不能被子类和类外访问
\_\_xxx\_\_ | magic方法或用户控制的命名空间

- interface, _internal, __private

### 2 配置规范
- 使用yaml或json作为配置文件

### 3 测试规范
#### 3.1 单元测试
- 使用pytest做为单元测试框架
- ` pytest ./tests `

#### 3.2 压力测试
- 使用 apache ab（Apache Bench）作为压力测试框架
- [example](https://gitlab.yunfutech.com/yunfu/starter/nlpstarter/tree/master/tests/ab_pref_test)

### 4 日志规范
- 使用如下代码获取日志对象
```
import logging

logger = logging.getLogger(__name__)
```

#### 4.1 使用日志级别
- 主要使用如下4个日志级别，优先级从高到低为：
- ERROR  # 错误
- WARNING  # 警告
- INFO  # 消息，输出对最终用户有实质意义的系统状态
- DEBUG  # 调试信息，仅在调试模式使用

#### 4.2 Exception日志	
Exception中的日志，应记录trace信息，例如：
```
try:
    open("/path/to/filename", "rb")
except Exception, e:
    logger.exception("Failed to open file")
```

#### 4.3 注意事项
- 不要在线上程序使用print
- 不要在生产环境中使用DEBUG日志
- 推荐使用YAML格式配置logging
