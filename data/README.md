# 数据目录

运行 `python scripts/prepare_data.py` 会自动查找或下载 SYSU-MM01。手动放置时，把课程链接下载的数据解压到 `SYSU-MM01/`（多嵌套一层也可以），需包含 cam1–cam6 和 exp。

数据内容（包括自动下载记录 `dataset_location.json`）默认忽略，不上传 Git。详见 [数据接口说明](../docs/data_interface.md)。
