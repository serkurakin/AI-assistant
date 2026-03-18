from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import chromadb
from chromadb.utils import embedding_functions
#from pypdf import PdfReader
from docling.document_converter import DocumentConverter
from glob import glob
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Папка с PDF
pdf_folder = r"E:\llm_bot\articles"

# Инициализация Docling
converter = DocumentConverter()

# Инициализация LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4o-mini",
    temperature=0
)

all_chunks = []
all_metadatas = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, 
    chunk_overlap=200,
    length_function=len,
    separators=['\n\n', '\n', ' ', '']
)

# функция очистки текста
def clean_extracted_text(text: str) -> str:
    
    # Убираем переносы слов (дефис в конце строки)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Заменяем любые последовательности пробелов и переносов на один пробел
    text = re.sub(r'\s+', ' ', text)
    
    # удаляем управляющие символы (control characters)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    return text.strip()


def extract_key_sections(text: str) -> str:
    """Извлекает только ключевые разделы статьи для суммаризации."""
    # Регулярные выражения для поиска заголовков (регистронезависимо)
    sections = {
        "Abstract": r"(?i)(#+\s*Abstract|#+\s*Аннотация|#+\s*Резюме)",
        "Methods": r"(?i)(#+\s*Materials\s+and\s+Methods|#+\s*Methods|#+\s*Методы\s+и\s+материалы)",
        "Conclusion": r"(?i)(#+\s*Conclusion|#+\s*Conclusions|#+\s*Заключение|#+\s*Выводы)"
    }
    
    extracted_content = ""
    
    for section_name, pattern in sections.items():
        match = re.search(pattern, text)
        if match:
            start_idx = match.start()
            # Берем текст от заголовка и следующие 4000 символов 
            # (или до следующего крупного заголовка)
            section_chunk = text[start_idx : start_idx + 4000]
            extracted_content += f"\n\n--- РАЗДЕЛ {section_name} ---\n{section_chunk}"
            
    # Если ничего не нашли, берем начало и конец статьи
    if len(extracted_content) < 500:
        return text[:5000] + "\n... [разрыв] ...\n" + text[-3000:]

    print(f"Найдено разделов для Summary: {list(sections.keys()) if extracted_content else 'Для Summary ничего не найдено'}")    
    return extracted_content

def generate_summary(full_markdown_text: str) -> str:
    """Генерирует резюме на основе только ключевых разделов."""
    
    # 1. Вырезаем только важное
    content_for_summary = extract_key_sections(full_markdown_text)
    
    prompt = f"""Ты - научный эксперт. 
    Твоя задача - на русском языке составить точное и краткое резюме (Summary) некоторых разделов статьи, которые представлены ниже. 
    Избегай общих фраз и пиши конкретику по этой статье.
    В резюме обязательно укажи: 
    1. Объект исследования в этой статье.
    2. Основная цель работы.
    3. Используемые в работе методы и материалы.
    4. Главный вывод работы.
    
    Текст статьи:
    {content_for_summary}
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Ошибка генерации Summary: {e}")
        return ""


# 1. Читаем файлы по одному
for pdf_path in glob(os.path.join(pdf_folder, "*.pdf")):
    try:
    
        file_name = os.path.basename(pdf_path)
        
        # Конвертируем PDF в структурированный объект
        result = converter.convert(pdf_path)
        
        # Извлекаем текст в формате Markdown (сохраняет таблицы и заголовки)
        full_text = result.document.export_to_markdown()

        if not full_text.strip():
            print(f" В файле {file_name} не удалось извлечь текст")
            continue
            
        # Генерируем SUMMARY (Для всей статьи сразу)
        summary_text = generate_summary(full_text)
        
        if summary_text:
            all_chunks.append(f"ОБЩЕЕ РЕЗЮМЕ СТАТЬИ {file_name}:\n{summary_text}")
            all_metadatas.append({"source": file_name, "type": "summary"})

        # Очистка текста
        full_text = clean_extracted_text(full_text)

        # Отсечение списка литературы
        # Список возможных заголовков раздела литературы
        ref_keywords = ["References", "REFERENCES", "Список литературы", "СПИСОК ЛИТЕРАТУРЫ", "Литература", 
        "ЛИТЕРАТУРА", "LITERATURE CITED", "LITERATURE", "Literature"]
        
        # Ищем самое последнее вхождение одного из ключевых слов
        cut_index = -1
        for kw in ref_keywords:
            last_idx = full_text.rfind(kw) # rfind ищет с конца текста
            if last_idx > cut_index:
                cut_index = last_idx
        
        # Если нашли заголовок "References" и он во второй половине текста
        if cut_index != -1 and cut_index > len(full_text) * 0.5:
            print(f" Отсекаем References в {file_name}")
            full_text = full_text[:cut_index]

        # Разрезаем текст конкретно этого файла
        file_chunks = text_splitter.split_text(full_text)
        
        # Для каждого чанка создаем метаданные с именем файла
        for chunk in file_chunks:
            if chunk and chunk.strip() and len(chunk.strip()) > 50: # игнорируем совсем мелкий мусор, пропускаем пустые чанки
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": file_name,
                    "page_count": result.document.pages_count if hasattr(result.document, 'pages_count') else 0
                })
            
        print(f" {file_name}: {len(file_chunks)} чанков")
    except Exception as e:
        print(f" Ошибка в {os.path.basename(pdf_path)}: {e}")

if not all_chunks:
    print(" Нет текста для базы данных!")
    exit()

# 2. Настройка БД
client = chromadb.PersistentClient(path="./chroma_db")

russian_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Удаляем старую коллекцию, если была
try:
    client.delete_collection(name='articles_database')
    print("Старая коллекция удалена")
except:
    pass

collection = client.create_collection(
    name='articles_database', 
    embedding_function=russian_embedding_function
)

# 3. Добавляем чанки с метаданными
ids = [str(i) for i in range(len(all_chunks))]

collection.add(
    documents=all_chunks,
    metadatas=all_metadatas,
    ids=ids
)

print(f" База готова. Всего чанков: {len(all_chunks)}")
print(f" Источники: {set([m['source'] for m in all_metadatas])}")