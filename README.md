# 로컬 LLM 추론 테스트 환경 세팅 가이드

## 1. 필수 프로그램 설치
- [ ] **Python**: [다운로드](https://www.python.org/downloads/)
  - Powershell에서 아래 명령어로 이미 설치되어있는지 확인
    ```Powershell
    python -V
    ```
- [ ] **Git**: [다운로드](https://git-scm.com/install/windows)
  - Powershell에서 아래 명령어로 이미 설치되어있는지 확인
    ```Powershell
    git -v
    ```
- [ ] **Visual Studio Code**: 코드 에디터 [설치](https://code.visualstudio.com/)

## 2. GPU 환경 세팅 (16GB VRAM 기준)
- [ ] **NVIDIA 드라이버**: 최신 버전으로 [업데이트](https://www.nvidia.com/ko-kr/drivers/)
- [ ] **CUDA Toolkit**: 공식 홈페이지에서 [다운로드](https://developer.nvidia.com/cuda-toolkit-archive) 및 설치
- [ ] **cuDNN**: 
  - CUDA 12.x용 cuDNN [다운로드](https://developer.nvidia.com/cudnn) 후 설치
  - cuDNN 설치 경로 하위의 `bin`, `include`, `lib` 폴더 찾기
  - 해당 폴더 내 파일들을 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{버전번호}` 내부의 동일 이름 폴더에 각각 복사/붙여넣기

## 3. 프로젝트 세팅 및 패키지 설치
- [ ] **폴더 생성**: 한글/공백 없는 경로에 작업 폴더 생성 (예: `D:\llm_workspace`)
- [ ] **Git Clone**: Powershell에서 다음 명령어 입력
  ```Powershell
  cd [생성한 폴더 경로]
  git clone https://github.com/waegari/3_5_prompt_test.git
  cd 3_5_prompt_test
  ```
- [ ] **패키지 설치**: Powershell에서 아래 명령어를 실행하여 GPU 가속이 포함된 라이브러리 설치
  ```Powershell
  $env:CMAKE_ARGS="-DGGML_CUDA=on"
  pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
  ```
  
## 4. GGUF 형식의 LLM 준비
- GGUF: Georgi Gerganov Unified Format
  - 모델 가중치 등의 효율적 저장 / 고속 로딩을 위해 만들어진 포맷. [HF docs](https://huggingface.co/docs/hub/gguf)
- [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) 모델을 `3_5_prompt_test/models` 경로에 다운로드

- 모델명 중 'nB'는 learnable parameter의 수를 의미
- BF16, FP16 등 16bit floating point 데이터 타입의 경우 파라미터당 약 2byte (16 bit) 저장 공간 차지
  - 9B 모델: learnable parameter 수가 90억 개
  - BF16 / FP16 모델 크기: 180억 bytes $\simeq$ 18GB
- GGUF가 지원하는 8 bits 양자화된 모델의 경우
  - 90억 bytes $\simeq$ 9GB

## 5. test.py 실행
- Visual Studio Code 실행
- `Open Folder` 버튼 클릭 / 혹은 `Ctrl` + `K` + `O`
- `3_5_prompt_test` 폴더 선택해 열기

- `View` -> `Terminal` 선택 / 혹은 Ctrl + ` 키로 Terminal 열기

- 다음 명령어 입력해 테스트
```Powershell
python test.py
```