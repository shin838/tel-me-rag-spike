"""Vector store 공통 인터페이스.

pgvector 와 Qdrant 를 같은 계약으로 감싼다. 비교 실험과 서비스 코드가
스토어 종류를 몰라도 되게 하는 것이 목적이다(본 프로젝트의 Repository 계층에 대응).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Hit:
    faq_id: str
    score: float  # 코사인 유사도 (1.0 에 가까울수록 유사)


class VectorStore(ABC):
    name: str

    @abstractmethod
    def ensure(self, model_key: str, dim: int) -> None:
        """모델별 컬렉션/테이블을 (재)생성한다."""

    @abstractmethod
    def upsert(self, model_key: str, faq_ids: list[str], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def build_index(self, model_key: str) -> float:
        """인덱스를 만들고 소요 ms 를 돌려준다."""

    @abstractmethod
    def search(self, model_key: str, vector: list[float], k: int) -> tuple[list[Hit], float]:
        """상위 k 건과 검색 소요 ms."""

    @abstractmethod
    def count(self, model_key: str) -> int: ...
