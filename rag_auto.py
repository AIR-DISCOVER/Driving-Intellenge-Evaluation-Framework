import os
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 配置信息
base_url = ""
api_key = ""
main_folder_path = "M:\\LLM\\dataset\\collected_data\\" # Modify to actual path


# 初始化模型
def initialize_models():
    embedding_model = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="gpt-4o", temperature=0)
    return embedding_model, llm


# 构建知识库
def build_knowledge_base():
    with open('information.json', 'r') as f:
        merged_data = json.load(f)

    documents = []
    for item in merged_data:
        if "Driver Mindset" in item:
            # 处理驾驶相关的记录
            page_content = f"Context: {item['Context']}\n" \
                           f"Driver Mindset: {item['Driver Mindset']}\n" \
                           f"Driving Decision: {item['Driving Decision']}\n" \
                           f"Driver Action: {item['Driver Action']}\n" \
                           f"Driver Evaluation: {item['Driver Evaluation']}"
        elif "Passenger Mindset" in item:
            # 处理乘客相关的记录
            page_content = f"Context: {item['Context']}\n" \
                           f"Passenger Mindset: {item['Passenger Mindset']}\n" \
                           f"Expectations: {item['Expectations']}\n" \
                           f"Passenger Perception: {item['Passenger Perception']}\n" \
                           f"Passenger Evaluation: {item['Passenger Evaluation']}"
        else:
            # 处理可能的其他情况
            page_content = f"Context: {item['Context']}"

        documents.append(Document(page_content=page_content))

    embedding_model, _ = initialize_models()
    return FAISS.from_documents(documents, embedding_model)


# 阶段评估处理器
class EvaluationPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        _, self.llm = initialize_models()
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def process_folder(self, folder_path):
        folder_name = os.path.basename(folder_path)

        # 自动构建文件路径
        context_file = os.path.join(folder_path, f"{folder_name}.txt")
        desc_file = os.path.join(folder_path, f"{folder_name}_des_new.txt")

        # 检查文件存在性
        if not all(os.path.exists(f) for f in [context_file, desc_file]):
            print(f"跳过文件夹 {folder_name}: 缺少必要文件")
            return

        # 读取文件内容
        with open(context_file, 'r', encoding='utf-8') as f:
            context = f.read()
        with open(desc_file, 'r', encoding='utf-8') as f:
            description = f.read()

        # 分阶段处理
        self._process_stage(folder_path, "操作层面评估", self._stage1_prompt(context, description))
        self._process_stage(folder_path, "策略层面评估", self._stage2_prompt(context, description))
        self._process_stage(folder_path, "战略层面评估", self._stage3_prompt(context, description))

        # 生成最终报告
        self._generate_final_report(folder_path)

    def _process_stage(self, folder_path, stage_name, prompt):
        result = self.qa_chain.run({"query": prompt})
        stage_file = os.path.join(folder_path, f"{stage_name}.txt")

        with open(stage_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {stage_name} ===\n{result}\n")
        print(f"已生成 {stage_name} 结果")

    def _stage1_prompt(self, context, description):
        return f"""基于以下内容进行驾驶操作层面评估：
        {context}
        {description}
        评估维度包括：控制精度、操作连贯性、异常处理能力..."""

    def _stage2_prompt(self, context, description):
        # 读取阶段1结果
        stage1_file = os.path.join(os.path.dirname(context), "操作层面评估.txt")
        with open(stage1_file, 'r') as f:
            stage1_result = f.read()

        return f"""综合以下内容进行策略层面评估：
        {context}
        {description}
        前期评估结果：{stage1_result}
        评估维度包括：社会智能、复杂场景应对..."""

    def _stage3_prompt(self, context, description):
        # 读取阶段2结果
        stage2_file = os.path.join(os.path.dirname(context), "策略层面评估.txt")
        with open(stage2_file, 'r') as f:
            stage2_result = f.read()

        return f"""综合以下内容进行战略层面评估：
        {context}
        {description}
        前期评估结果：{stage2_result}
        评估维度包括：宏观交通视野、风险偏好..."""

    def _generate_final_report(self, folder_path):
        report_content = []
        for stage in ["操作层面评估", "策略层面评估", "战略层面评估"]:
            with open(os.path.join(folder_path, f"{stage}.txt"), 'r') as f:
                report_content.append(f.read())

        with open(os.path.join(folder_path, "综合评估报告.txt"), 'w') as f:
            f.write("\n\n".join(report_content))


# 主执行流程
def main():
    # 构建知识库
    vector_store = build_knowledge_base()

    # 初始化评估管道
    pipeline = EvaluationPipeline(vector_store)

    # 遍历处理所有子文件夹
    for root, dirs, _ in os.walk(main_folder_path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            if os.path.isdir(folder_path):
                print(f"\n正在处理文件夹: {dir_name}")
                pipeline.process_folder(folder_path)


if __name__ == "__main__":
    main()
