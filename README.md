# RAG 기반 한국어 프롬프트 리라이팅 시스템

짧거나 모호한 사용자 입력을 **자연스럽고 구체적인 프롬프트**로 변환하는 RAG + T5 기반 한국어 프롬프트 리라이팅 프로젝트입니다.  

---

## 1. 주요 기능

- 짧거나 모호한 사용자 입력을 자연스럽고 구체적인 프롬프트로 변환
- 데이터셋에서 **유사한 유형의 프롬프트를 검색**하여 리라이팅 품질 향상 (RAG)
- **T5 기반 한국어 생성 모델**을 사용하여 자연스러운 한국어 출력 생성

---

## 2. 프로젝트 구조

```text
project/
├── app.py                          # FastAPI 서버 & 웹 UI 엔드포인트
├── model_handler.py                # RAG + T5 기반 프롬프트 리라이팅 핵심 로직
├── prompt_dataset_input_label22.json  # 프롬프트 데이터셋 (input-label pair)
├── t5_finetuned_result/            # 파인튜닝된 T5 체크포인트 (예: config.json, pytorch_model.bin 등)
├── templates/
│   └── index.html                  # 웹 UI 템플릿
└── static/                         # 정적 리소스 (CSS, JS, 이미지 등)
```

---

## 3. 아키텍처 
전체 데이터 흐름은 다음과 같습니다.
```text
User
  ↓
Web UI (FastAPI + Jinja2)
  - 사용자 입력 처리
  - 결과 출력
  ↓
rag_refine_prompt()  (model_handler.py의 메인 로직)
  ↓
[Step 1] Prompt Retrieval
  - ko-sbert 모델로 입력 임베딩
  - FAISS Index에서 유사 프롬프트 Top-K 검색
  ↓
[Step 2] Prompt Generation (T5)
  - 검색된 프롬프트들을 컨텍스트로 구성
  - Fine-tuned T5 모델로 새로운 프롬프트 생성
  ↓
[Step 3] Post-Processing
  - 특수 토큰 제거
  - 중복 문장 정리
  - 첫 문장 중심으로 간결하게 정제
  ↓
Final Output
```
---
## 4. 핵심 동작 방식
### 4.1 유사 프롬프트 검색 (RAG Retrieval)
 - 임베딩 모델: jhgan/ko-sbert-sts
 - 검색 엔진: FAISS L2 Index
 - 데이터셋(label) 전체를 임베딩하여 사전 구축
 - 사용자 입력을 벡터화 후 Top-K 유사 프롬프트 검색

### 4.2 T5 기반 프롬프트 생성
 - 기본 모델: KETI-AIR/ke-t5-base
 - 추가 파인튜닝된 체크포인트: ./t5_finetuned_result
 - Retrieval 결과 상위 N개(예: 3개)를 컨텍스트로 함께 입력하여 생성 품질 강화

### 4.3 출력 후처리 (Post-Processing)
 - <pad> 등 특수 토큰 제거
 - 불필요한 반복/중복 단어 제거
 - 첫 문장 중심으로 깔끔하게 정제하여 한 문장 위주의 출력 유지

## 5. 실행 방법
 ### 5.1 환경 세팅
```text
pip install -r requirements.txt
```
requirements.txt에는 대략 아래와 같은 패키지가 포함됩니다.
- fastapi
- uvicorn
- sentence-transformers
- transformers
- faiss-cpu
- numpy

### 5.2 서버 실행 및 접속
```
## 필수 요구 사항

- **Python**: 3.11 버전 이상을 권장합니다.

## 설치 방법

1. **Python 설치** (이미 설치된 경우 생략 가능):
   - **Windows** (winget 사용):
     ```powershell
     winget install -e --id Python.Python.3.11
     ```

2. **의존성 패키지 설치**:
   프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다:
   ```bash
   pip install -r requirements.txt.txt
   pip install fastapi uvicorn transformers sentence-transformers
   ```
   *(참고: 원활한 실행을 위해 추가 패키지를 함께 설치합니다).*

1. **Web 디렉토리로 이동**:
   ```bash
   cd Web
   ```

2. **서버 시작**:
   ```bash
   python -m uvicorn app:app --reload
   ```
   *만약 `python` 명령어가 실행되지 않는다면, python 실행 파일의 전체 경로를 사용하세요.*

3. **앱 접속**:
   브라우저를 열고 다음 주소로 접속하세요: [http://127.0.0.1:8000](http://127.0.0.1:8000)

```
```
## 6. 예시 입출력 
```text
요즘 뭐가 잘 팔려?
```
출력 예시
```text
온라인 쇼핑몰에서 판매량이 급증한 인기 상품 3가지를 소개하고, 각 상품의 특징과 주요 구매층을 설명해줘.
```

## 7. 장점 및 활용 포인트
- 사용자의 짧고 모호한 질문을 고품질·구체적·모델 친화적인 프롬프트로 자동 변환
- 대규모 챗봇/AI 서비스의 입력 전처리 & 품질 향상 모듈로 활용 가능
- RAG + 생성(Generation)을 결합하여 유연성 + 신뢰성을 동시에 확보
- 데이터셋 확장 및 재학습을 통해 품질을 지속적으로 개선하기 쉬움

## 8. 향후 개선 
- Retrieval 가중치 조절 또는 Fusion-based RAG(여러 소스 결합) 적용
- Few-shot Prompting 모듈 추가 (예시 기반 리라이팅)
- 응답 스타일(톤, 길이, 포맷 등)을 사용자가 선택할 수 있는 옵션 제공

 




