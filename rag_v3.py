from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from glob import glob
import os
import re
# Папка с PDF
pdf_folder = r"E:\llm_bot\articles"

all_chunks = []
all_metadatas = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100, 
    length_function=len,
    separators=['\n\n', '\n', '.', '!', '?', ' ', '', ';', ':']
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

# 1. Читаем файлы по одному
for pdf_path in glob(os.path.join(pdf_folder, "*.pdf")):
    try:
        reader = PdfReader(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        # Собираем текст только с непустых страниц
        text_parts = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        
        if not text_parts:
            print(f" В файле {file_name} нет текста")
            continue
            
        full_text = "\n".join(text_parts)
        
        # очистка текста сразу после объединения страниц
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
                    "page_count": len(reader.pages)
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