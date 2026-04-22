from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from abc import ABC, abstractmethod
from operator import itemgetter
from pathlib import Path
import os
import hashlib
from langchain import hub


class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 8
        # 공식 OpenAI API 모델명 (OpenRouter 스타일의 "openai/..." 접두사 없음)
        self.model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
        self.temperature = 0
        self.prompt = "teddynote/rag-prompt"
        self.embeddings = "text-embedding-3-small"
        self.cache_dir = Path(".cache/embeddings")
        self.index_dir = Path(".cache/faiss_index")

    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def _embedding_openai_kwargs(self) -> dict:
        """OpenAI 임베딩. 기본은 공식 API(``base_url`` 미설정 = api.openai.com).

        자체 호스팅·Azure 등 OpenAI 호환 엔드포인트만 쓸 때 ``OPENAI_BASE_URL``을 설정합니다.
        (``EMBEDDING_BASE_URL``은 사용하지 않습니다. 공식 API만 쓰면 .env에서 제거하세요.)
        """
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        kwargs: dict = {
            "model": self.embeddings,
            "api_key": api_key,
        }
        raw_base = os.getenv("OPENAI_BASE_URL")
        if raw_base and str(raw_base).strip():
            kwargs["base_url"] = str(raw_base).strip()
        return kwargs

    def create_embedding(self):
        try:
            # 캐시 디렉토리 생성
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # 기본 임베딩 모델 생성
            underlying_embeddings = OpenAIEmbeddings(**self._embedding_openai_kwargs())

            # 파일 기반 캐시 스토어 생성
            store = LocalFileStore(str(self.cache_dir))

            # 캐시 기반 임베딩 생성 (SHA-256 사용으로 보안 강화)
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings,
                store,
                namespace=self.embeddings,
                key_encoder="sha256",
            )

            return cached_embeddings

        except Exception as e:
            print(f"Warning: Failed to create cached embeddings: {e}")
            print("Falling back to basic OpenAI embeddings without caching")
            return OpenAIEmbeddings(**self._embedding_openai_kwargs())

    def create_vectorstore(self, split_docs):
        try:
            # 인덱스 디렉토리 생성
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # 문서 내용 기반 해시 계산
            doc_contents = "\n".join([doc.page_content for doc in split_docs])
            doc_hash = hashlib.md5(doc_contents.encode()).hexdigest()

            # 해시 파일 경로와 인덱스 파일 경로
            hash_file = self.index_dir / "doc_hash.txt"
            index_path = str(self.index_dir / "faiss_index")

            # 기존 인덱스가 있고 문서가 변경되지 않았는지 확인
            try:
                if (
                    hash_file.exists()
                    and Path(index_path + ".faiss").exists()
                    and hash_file.read_text().strip() == doc_hash
                ):

                    # 기존 인덱스 로드 시도
                    vectorstore = FAISS.load_local(
                        index_path,
                        self.create_embedding(),
                        allow_dangerous_deserialization=True,
                    )
                    print("Loaded existing FAISS index from cache")
                    return vectorstore

            except Exception as e:
                print(f"Warning: Failed to load existing index: {e}")
                print("Creating new index...")

            # 새로운 인덱스 생성
            vectorstore = FAISS.from_documents(
                documents=split_docs, embedding=self.create_embedding()
            )

            # 인덱스와 해시 저장 시도
            try:
                vectorstore.save_local(index_path)
                hash_file.write_text(doc_hash)
                print("FAISS index saved to cache")
            except Exception as e:
                print(f"Warning: Failed to save index to cache: {e}")
                print("Index will not be cached for next use")

            return vectorstore

        except Exception as e:
            print(f"Error: Failed to create vectorstore with caching: {e}")
            print("Falling back to basic FAISS creation without caching")
            return FAISS.from_documents(
                documents=split_docs, embedding=self.create_embedding()
            )

    def create_retriever(self, vectorstore):
        # Cosine Similarity 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        kwargs: dict = {
            "temperature": self.temperature,
            "model": self.model_name,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
        # 채팅만 별도 게이트웨이를 쓸 때(임베딩용 EMBEDDING_BASE_URL과 분리)
        raw_base = os.getenv("OPENAI_BASE_URL")
        if raw_base and str(raw_base).strip():
            kwargs["base_url"] = str(raw_base).strip()
        return ChatOpenAI(**kwargs)

    def create_prompt(self):
        return hub.pull(self.prompt)

    def create_chain(self):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        self.chain = (
            {"question": itemgetter("question"), "context": itemgetter("context")}
            | prompt
            | model
            | StrOutputParser()
        )
        return self
