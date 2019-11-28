## 代码常见问题

#### 1 使用dict.setdefault()

- 普通写法

``` py
result = {}
for (key, value) in data:
    if key in result:
        result[key].append(value)
    else:
        result[key] = [value]
```

- 更好的写法: 一行完成append

```  python
result = {}
for (key, value) in data:
    result.setdefault(key, []).append(value)
```

#### 2 禁止隐式相对导入

``` python
from __future__ import absolute_import # 对于python2，禁止相对导入
import model                 # 隐式相对导入，Python3 中已经被废弃
from . import model          # 显式相对导入
from nlpstarter import bench  # 此为 绝对导入
``` 

#### 3 不要使用from abc import \*导入
#### 4 使用list comprehension
#### 5 尽量使用自然跨行连接而不是\\

#### 6 使用New-Style Class
- 类必须从object中继承
- 可以使用super函数，而且比老式类多了一些特殊方法，比如__new__方法

```  python
class OldStyleClass():  
    def __init__(self, a):  
        self.a = a
```
```  python
class NewStyleClass(object):  
    def __init__(self, a):  
        self.a = a
```
#### 7 逻辑代码与测试代码分离
不要将测试代码写在你的实现逻辑代码中(doctest 不受此规范约束)，　如
```  python
# core_implemention.py
def core_func(msg):
    print("Core implemention of current module!")
    print("Got: {}".format(msg))

if __name__ == "__main__":
    core_func("test")
```

#### 8 Exception: do not use bare except

``` python
def do_not_raise(user_defined_logic):
    try:
        user_defined_logic()
    except:
        logger.warning("User defined logic raises an exception", exc_info=True)
        # ignore
```

``` py
def do_not_raise(user_defined_logic):
   try:
       user_defined_logic()
   except Exception:          ### <= Notice here ###
       logger.warning("User defined logic raises an exception", exc_info=True)
       # ignore
```
