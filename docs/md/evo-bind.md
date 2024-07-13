
## EVO Bind

- Use `libevo` in Other Program Languages


### 1.1 Bind to other Language

```shell
cd bind
# bind to all language
make
# bind to [Language]
make python
# delete
make clean
```


### 1.2 Python API

We use `pybind11` to bind `pyevo` to Python.

You should install:
```shell
pip install pybind11
cd bind/python
make
make run
```