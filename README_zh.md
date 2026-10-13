# STI-PSD：水质图像补全与污染源识别

本仓库用于补全不完整的水质时空图像，并用残差网络回归污染源参数。
仓库包含 64 个模拟样本，五种基础补全方法（零填充、线性插值、固定秩、Telea、N-S），
可选的 ResNet 训练与 ADR 代理修正，以及测试和结果导出。

## 运行

在本目录打开终端。建议使用独立 Python 虚拟环境，具体创建方式见英文首页。

```bash
python -m pip install -r requirements.txt
python run.py
```

完成后查看 `outputs/default/summary.csv`。
再次运行请指定另一个输出目录，例如 `python run.py --out outputs/run_2`，避免覆盖已有结果。

需要运行源参数预测和 ADR 代理修正时：

```bash
python -m pip install -r requirements-ml.txt
python run.py --with-adr --train-epochs 2 --out outputs/extended
python -m unittest discover -s tests -v
```

`--train-epochs 2` 用于快速检查流程，正式训练请增大轮数。模型用完整训练图像学习，
随后冻结，对同一测试样本的各补全结果进行预测（Strategy B）。


## 引用与许可

引用信息见 `CITATION.cff`，版本变更见 `CHANGELOG.md`。
代码以 MIT 许可证发布（见 `LICENSE`）。`examples/` 中的样本数据取自 TP_S1_ALL 数据集
（https://doi.org/10.5281/zenodo.21498704），沿用该数据记录的许可。
