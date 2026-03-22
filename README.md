# RAG 小说问答系统

本项目提供了两个基于 LangChain 和 Qwen 模型的 RAG（检索增强生成）实现，用于对小说文本进行智能问答。

## 🚀 功能特点

- **基础 RAG** (`RAGQA.py`)：简单的检索-生成流程，适合快速上手。
- **Agentic RAG** (`AgenticRAGQA.py`)：增强版，包含意图判断、查询改写、向量检索 + 重排序（Rerank），提高答案质量。
- 支持 CPU / GPU 运行，自动检测设备。
- 向量库持久化，避免重复构建。
- 可选的重排序模型，提升检索精准度。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `RAGQA.py` | 最简 RAG 实现：文档切分 → 向量库 → 检索 → 生成 |
| `AgenticRAGQA.py` | Agentic RAG 实现：意图判断 → 查询改写 → 向量检索 → 重排序 → 生成 |
| `novel2.txt` | 示例小说文件（也可以自行准备，UTF-8 编码） |
| `requirements.txt` | 依赖包列表（可手动安装） |

## 🛠️ 环境依赖

Python 3.8+，建议使用虚拟环境。

```bash
pip install torch transformers langchain langchain-community langchain-huggingface langchain-classic chromadb sentence-transformers jieba
```

或一键安装：

```bash
pip install -r requirements.txt
```

> 注意：`sentence-transformers` 用于重排序，如不使用可跳过。

## ⚙️ 配置说明

两个脚本的配置均在文件开头的“配置区域”中，可根据需要修改：

- `FILE_PATH`：小说文件路径（UTF-8 编码）
- `PERSIST_DIR`：向量库持久化目录
- `MODEL_NAME`：大语言模型名称（推荐 `Qwen/Qwen2.5-1.5B-Instruct`）
- `CHUNK_SIZE` / `CHUNK_OVERLAP`：文档切块大小与重叠
- `RETRIEVAL_K`：最终使用的文档块数（Agentic 版中为重排序后数量）
- `SIMILARITY_THRESHOLD`：意图判断的相似度阈值（Agentic 版）

更多参数请参考脚本中的注释。

## 🚀 快速开始

1. 准备一本小说文本文件（UTF-8 编码），例如 `novel.txt`。
2. 修改脚本中的 `FILE_PATH` 为你的文件路径。
3. 运行基础版：

```bash
python RAGQA.py
```

4. 运行增强版：

```bash
python AgenticRAGQA.py
```

首次运行会下载模型（Qwen、Embedding、Reranker 等），请耐心等待。后续运行会直接加载本地缓存。

## 📖 使用示例

启动后，在提示符下输入问题，例如：

```
📖 你的问题（输入 q 退出）: 苏婉清怎么死的？
```

系统会根据意图判断决定是否检索，并给出答案。

## 🧠 实现原理

### 基础 RAG
1. 将小说切分为固定长度的文本块。
2. 使用嵌入模型将文本块转为向量，存入向量库。
3. 用户输入问题 → 转为向量 → 检索最相似的 k 个文本块。
4. 将文本块与问题一起拼接成提示词，交给大语言模型生成答案。

### Agentic RAG
1. **意图判断**：利用向量相似度判断问题是否与小说内容相关，避免对无关问题执行检索。
2. **查询改写**：用大语言模型去除语气词、精简问题，提高检索命中率。
3. **检索**：从向量库召回更多候选（如 10 个）。
4. **重排序**：使用 CrossEncoder 模型对候选文档重新打分，选出最相关的 3 个。
5. **生成**：将筛选后的文本块与问题输入大语言模型，生成最终答案。

## 📌 注意事项

- 小说文件必须为 UTF-8 编码，否则可能乱码。
- 首次运行会下载模型，请确保网络通畅。
- 如显存不足，可将 `DEVICE` 设为 `"cpu"`，或换用更小的模型（如 `Qwen/Qwen2.5-0.5B-Instruct`）。
- 重排序模型 `BAAI/bge-reranker-base` 约 200MB，若网络慢可替换为 `cross-encoder/ms-marco-MiniLM-L-6-v2`（英文为主）。

## 🤝 贡献

共同完善这个项目。

## 📄 许可证

MIT License

---

Happy coding! 📖✨
