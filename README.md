-----

## Image Search System (이미지 검색 시스템)

환영합니다\! 이 프로젝트는 **AI 기반의 이미지 검색 시스템**의 기본 구조을 구축하고 배포하는 것을 목표로 합니다. 방대한 이미지 데이터셋에서 유사한 이미지를 찾아내는 기본적인 도구를 제공합니다. 이 시스템은 딥러닝 모델을 활용하여 이미지의 특징을 추출하고, 이를 기반으로 검색 결과를 제공합니다.

---
## 🌟 주요 기능

* **기본적인 이미지 검색 구현:** **ResNet-18 모델**을 활용하여 이미지 특징을 학습하고, 이를 기반으로 간단한 이미지 검색 기능을 만들었습니다.
* **스트림릿(Streamlit) 기반 웹 인터페이스:** `app.py`를 통해 웹 브라우저에서 쉽게 접근하고 사용할 수 있는 **간결한 웹 애플리케이션** 형태를 갖추고 있습니다.
* **사용자 데이터 기반 학습 및 검색:** 직접 수집한 이미지 데이터를 학습시켜 모델을 구성하고, 이를 활용하여 검색 시스템을 구축하는 과정이 담겨 있습니다.

---

### 🚀 시작하기

이 프로젝트를 로컬 환경에서 실행하려면 다음 단계를 따르세요.

#### 📝 사전 준비 사항

  * Python 3.8 이상
  * `pip` (Python 패키지 관리자)
  * Git

#### 📦 설치

1.  **리포지토리 클론:**

    ```bash
    git clone https://github.com/bindobi/imageretrievalsystem.git
    cd imageretrievalsystem
    ```

2.  **필수 라이브러리 설치:**

    `requirements.txt` 파일에 포함된 모든 의존성을 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

#### 🏃‍♂️ 시스템 실행

1.  **데이터셋 준비:** `data_download_python` 스크립트를 사용하여 필요한 이미지 데이터셋을 다운로드하고 `my_dataset` 디렉터리에 저장하세요. 또는 자체 데이터셋을 `my_dataset` 구조에 맞게 준비할 수 있습니다.

    ```bash
    python data_download_python/download_script.py # 예시 (실제 스크립트 이름 확인 필요)
    ```

2.  **모델 훈련 (선택 사항):** 이미 훈련된 `trained_model.pth` 파일이 있다면 이 단계를 건너뛸 수 있습니다. `ai_train.ipynb` Jupyter Notebook을 열어 모델을 직접 훈련시키거나, 기존 모델을 파인튜닝할 수 있습니다.

    ```bash
    jupyter notebook ai_train.ipynb
    ```

    훈련이 완료되면 `trained_model.pth` 파일이 생성되거나 업데이트됩니다.

3.  **애플리케이션 실행:**

    ```bash
    streamlit run app.py
    ```

    명령어를 실행하면 웹 브라우저가 자동으로 열리면서 이미지 검색 시스템 애플리케이션이 나타납니다.

-----

### 📁 프로젝트 구조

```
imageretrievalsystem/
├── data_download_python/   # 데이터셋 다운로드 관련 스크립트
├── my_dataset/             # 이미지 데이터셋 저장 경로
├── ai_train.ipynb          # AI 모델 훈련 및 평가를 위한 Jupyter Notebook
├── app.py                  # Streamlit 기반의 웹 애플리케이션 코드
├── public_function.py      # 재사용 가능한 공통 함수 모듈
├── trained_model.pth       # 훈련된 딥러닝 모델 가중치 파일
├── README.md               # 프로젝트 설명 파일 (현재 보고 계신 파일)
└── requirements.txt        # 프로젝트 실행에 필요한 파이썬 패키지 목록
```
