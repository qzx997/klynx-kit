import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
basedir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(basedir, ".env"))

try:
    from kbase import parse_pdfs, create_vector_kb, KBManager
except ImportError:
    # 确保能导入 kbase (如果直接运行脚本，当前目录 klynx 会在 path 中)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from kbase import parse_pdfs, create_vector_kb, KBManager

def build_knowledge_base():
    """解析PDF并构建向量知识库"""
    
    # 定位路径 (基于当前文件位置 e:\...\klynx\core\klynx\build_kb.py)
    # klynx_root = e:\...\klynx
    klynx_root = Path(__file__).resolve().parent.parent.parent
    kb_root = klynx_root / "knowledge_db"
    
    # 1. 路径设置
    input_pdf_dir = kb_root / "papers_to_parser"
    # 输出到 structured_kb 下的临时测试目录，避免污染正式数据
    parsed_output_dir = kb_root / "structured_kb" / "test_parsing_output"
    # 向量库输出到 vector_db
    vector_db_dir = kb_root / "vector_db"
    
    print("="*60)
    print("🧪 KBase 模块集成测试 (独立脚本)")
    print("="*60)
    print(f"项目根目录: {klynx_root}")
    print(f"PDF 输入: {input_pdf_dir}")
    print(f"解析输出: {parsed_output_dir}")
    print(f"向量库输出: {vector_db_dir}")
    
    # 2. 执行 PDF 解析
    print("\n[Step 1] 解析 PDF...")
    # 检查输入目录是否有 PDF
    if not input_pdf_dir.exists():
        input_pdf_dir.mkdir(parents=True)
    
    pdfs = list(input_pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  警告: {input_pdf_dir} 中没有 PDF 文件。")
        print("    将跳过解析步骤，尝试使用现有的 quantum_memory 数据构建向量库。")
        # 回退到使用正式数据作为输入源
        parsed_output_dir = kb_root / "structured_kb" / "quantum_memory"
    else:
        stats = parse_pdfs(str(input_pdf_dir), str(parsed_output_dir))
        print(f"✅ 解析完成: {stats}")

    # 3. 创建向量知识库
    print("\n[Step 2] 构建向量知识库...")
    kb_name = "test_quantum_kb"
    
    # 尝试查找本地模型
    local_model_path = kb_root / "all-MiniLM-L6-v2"
    if local_model_path.exists():
        embedding_model = str(local_model_path)
        print(f"   使用本地模型: {embedding_model}")
    else:
        embedding_model = "all-MiniLM-L6-v2"
        print(f"   使用在线模型: {embedding_model}")

    try:
        chroma_path = create_vector_kb(
            name=kb_name,
            input_dir=str(parsed_output_dir),
            output_dir=str(vector_db_dir),
            overwrite=True,  # 测试模式强制覆盖
            embedding_model=embedding_model
        )
        print(f"\n✅ 向量知识库构建成功!")
        print(f"   集合名称: {kb_name}")
        print(f"   存储路径: {chroma_path}")
        
        # 4. 简单验证查询
        print("\n[Step 3] 验证查询...")
        mgr = KBManager()
        mgr.add_kb("test_kb", chroma_path)
        
        query = "quantum memory"
        print(f"   执行查询: '{query}'")
        result = mgr.query(query, top_k=2)
        print(f"   查询结果长度: {len(result)} 字符")
        if "<error>" not in result:
             print("   ✅ 查询成功返回结果")
        else:
             print(f"   ❌ 查询返回错误: {result[:100]}...")

    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)

if __name__ == "__main__":
    build_knowledge_base()
