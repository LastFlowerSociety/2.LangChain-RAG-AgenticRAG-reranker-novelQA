"""
Agentic RAG 简单实现：意图判断 + 查询改写 + 检索增强生成
基于之前 Qwen 模型的代码扩展
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import CrossEncoder

# ========== 配置 ==========
FILE_PATH = "novel2.txt"
PERSIST_DIR = "./chroma_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# reranker 设置
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-base"  # 支持中文，效果较好
# 备选： "cross-encoder/ms-marco-MiniLM-L-6-v2"（英文为主，但也可用）

print(f"🔧 使用设备: {DEVICE}")

# ========== 加载文档和向量库（与之前相同） ==========

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": DEVICE}
)
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print("💾 加载已有向量库")
else:
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        keep_separator=False
    )
    docs = text_splitter.split_documents(documents)
    print(f"✂️ 文档切分为 {len(docs)} 个片段")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    print("💾 新建向量库")

# ========== 加载 Qwen 模型（用作 LLM） ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)
llm = HuggingFacePipeline(pipeline=pipe)
# 加载 reranker（如果使用）
if USE_RERANKER:
    print(f"🔄 正在加载 reranker 模型：{RERANKER_MODEL} ...")
    reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    print("✅ reranker 加载完成")
else:
    reranker = None

# ========== 工具1：意图判断（是否需要检索） ==========


# ========== 配置：相似度阈值（可调整） ==========
# 距离越小表示越相似，根据实际小说内容微调
SIMILARITY_THRESHOLD = 13  # 距离小于此值则认为相关


def need_retrieval(query: str) -> bool:
    # 快速规则过滤（问候语等明显不需要检索的）
    greetings = ["你好", "您好", "hi", "hello", "在吗", "你是谁", "你叫什么", "谢谢", "怎么样"]
    if any(g in query for g in greetings):
        return False

    # 使用向量库判断是否与小说内容相关
    # similarity_search_with_score 返回 (Document, distance) 列表，距离越小越相似
    results = vectorstore.similarity_search_with_score(query, k=1)
    if not results:
        return False  # 空结果，认为不相关

    _, distance = results[0]  # distance 是余弦距离或 L2 距离，范围取决于 embedding
    # 如果距离小于阈值，说明问题与小说中的某段内容很接近，需要检索
    return distance < SIMILARITY_THRESHOLD

# ========== 工具2：查询改写（为检索优化） ==========


def rewrite_query(query: str) -> str:
    # 定义 prompt 模板
    prompt = ChatPromptTemplate.from_template("""
删减问题，去掉多余的语气词，保留核心信息，但不要增删信息。保持问句形式。只输出改写后的句子，不要有其他内容。

原问题：{query}
改写后：""")

    # 构建链：prompt -> llm -> 输出解析器
    chain = prompt | llm | StrOutputParser()

    # 调用链，传入参数
    rewritten = chain.invoke({"query": query}).strip()

    # 清理可能残留的引号或多余文字（模型偶尔还会加）
    import re
    # 去掉首尾引号
    rewritten = rewritten.strip('"').strip("'")
    # 如果模型输出了“改写后：xxx”，提取后面的部分
    if "改写后：" in rewritten:
        rewritten = rewritten.split("改写后：")[-1].strip()
    if "Assistant" in rewritten:
        rewritten = rewritten.split("Assistant")[0].strip()
    # 如果结果为空或太短，返回原问题
    if len(rewritten) < 2:
        return query
    return rewritten
# ========== 工具3：检索增强生成（原有的 RAG 链） ==========
# 这里复用之前的 retrieval_chain，但我们需要根据改写后的查询来检索
# 为了方便，我们单独定义一个检索函数
# 在 retrieve_and_generate 函数内部添加重排序逻辑


def retrieve_and_generate(query: str) -> str:
    # 1. 向量检索，召回更多候选（例如 k=10）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 多召回一些
    candidate_docs = retriever.invoke(query)

    if not candidate_docs:
        return "没有找到相关信息。"

    # 2. 重排序（如果启用）
    if USE_RERANKER and reranker is not None:
        # 准备 (query, passage) 对
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        # 计算分数（分数越高越相关）
        scores = reranker.predict(pairs)
        # 按分数降序排序
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # 取前 3 个最相关的
        top_docs = [candidate_docs[i] for i in sorted_indices[:3]]
    else:
        # 不使用 reranker 时，直接取前 3 个
        top_docs = candidate_docs[:3]

    # 3. 构建上下文
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # 4. 提示词（简洁）
    prompt_template = ChatPromptTemplate.from_template("""
根据以下上下文回答问题，直接给出答案，不要添加额外标记。如果没有结果，就承认不知道：

上下文：{context}

问题：{input}

答案：""")
    formatted_prompt = prompt_template.format(context=context, input=query)

    # 5. 生成
    response = llm.invoke(formatted_prompt)

    # 6. 清理输出
    cleaned = re.sub(r'<\|im_[^>]+\|>', '', response)
    cleaned = re.sub(r'^答案[：:]\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^assistant\s*[：:]\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "抱歉，没有找到相关信息。"
    return cleaned


# ========== 主对话循环 ==========
print("🎉 Agentic RAG 启动！我会先判断是否需要检索，如果需要就改写查询再检索~\n")

while True:
    query = input("📖 你的问题（输入 q 退出）: ")
    if query.lower() == "q":
        break

    try:
        if not need_retrieval(query):
            print("💬 直接回答...")
            # 直接回答的提示也要清理
            direct_prompt = f"请友好地回答用户：{query}"
            answer = llm.invoke(direct_prompt)
            answer = re.sub(r'<\|im_[^>]+\|>', '', answer)
            answer = answer.strip()
            print(f"✨ 回答：{answer}\n")
        else:
            print("🔍 需要检索，正在优化查询...")
            rewritten = rewrite_query(query)
            print(f"✏️ 改写后的查询：{rewritten}")
            answer = retrieve_and_generate(rewritten)
            if "Assistant" in answer:
                answer = answer.split("Assistant")[0].strip()
            print(f"✨ 回答：{answer}\n")
    except Exception as e:
        print(f"⚠️ 出错了：{e}\n")
