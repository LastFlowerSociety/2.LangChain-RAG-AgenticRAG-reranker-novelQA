"""
小说 RAG 问答 - 支持中文 + 向量库持久化
"""

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

# ========== 配置区域 ==========
FILE_PATH = "novel2.txt"            # 小说文件路径
PERSIST_DIR = "./chroma_db"        # 向量库保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 支持中文的轻量模型

print(f"🔧 使用设备: {DEVICE}")

# ========== 1. 加载小说文档 ==========
loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()
print(f"✅ 加载文档完成，共 {len(documents)} 页")

# ========== 2. 文档切块 ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],  # 中文优先按句子分割
    keep_separator=False  # 不保留分隔符，让文本更干净
)
docs = text_splitter.split_documents(documents)
print(f"✂️ 文档切分为 {len(docs)} 个片段")

# ========== 3. 向量库（自动加载/创建） ==========
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": DEVICE}
)

# 检查是否已存在向量库
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # 加载已有向量库
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    print("💾 加载已有向量库完成")
else:
    # 创建新向量库并保存
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIR
    )
    print("💾 新建向量库并保存完成")

# ========== 4. 加载中文LLM ==========
# 使用 Qwen 模型，需要加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
)
# 移到指定设备
if DEVICE == "cuda":
    model = model.cuda()
else:
    model = model.cpu()

# 构建 pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    # device=0 if DEVICE == "cuda" else -1
)
llm = HuggingFacePipeline(pipeline=pipe)

# ========== 5. 构建 LCEL 问答链 ==========
# 提示模板（Qwen 建议使用对话格式）
prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题，直接给出答案，不要添加额外标记：

上下文：{context}

问题：{input}

答案：""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
    vectorstore.as_retriever(search_kwargs={"k": 3}),
    document_chain
)

print("🎉 系统启动成功！开始提问吧~\n")

# ========== 6. 交互式问答 ==========
while True:
    query = input("📖 你的问题（输入 q 退出）: ")
    if query.lower() == "q":
        print("👋 下次见啦，加油~")
        break
    try:
        result = retrieval_chain.invoke({"input": query})
        # 提取生成的文本（可能包含特殊标记，需要简单清理）
        answer = result['answer']
        # 清理所有特殊标记和多余空白
        import re
        answer = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', answer)  # 移除完整对话块
        answer = re.sub(r'<\|im_(start|end)\|>', '', answer)           # 移除残留标签
        answer = answer.strip()
        print(f"✨ 回答：{answer}\n")
    except Exception as e:
        print(f"⚠️ 出错了：{e}\n")
