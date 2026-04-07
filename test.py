import os
import time
import json
import re
from math import ceil, floor
from llama_cpp import Llama

from send_result import send_json_to_server
from reconstruction_indices import assign_indices_from_reconstruction
from prompts import SUMMARIZATION_PROMPT, RECONSTRUCTION_PROMPT, STT_INPUT_DATA, JOB_ID  # 프롬프트 파일 임포트 # 임포트 부분에 JOB_ID 추가

time1 = time.time()

def json_parse(response):
    raw_content = response["choices"][0]["message"]["content"]

    parsed = None
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw_content)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                parsed = None
    return parsed, raw_content

MODEL_PATH = "models/Qwen3.5-9B-Q8_0.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"파일을 찾을 수 없습니다.")
    print(f"'{MODEL_PATH}' 경로에 파일이 있는지, 파일명이 정확한지 확인해주세요.")
    exit(1)

print(f"loading {MODEL_PATH} on the GPU")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=int(261344*0.75),      
    verbose=False     
)

time2 = time.time()

print("\n model loaded\n")

def get_response(system_prompt, input_data, max_tokens=-1, temperature=0.1):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 텍스트를 처리해줘: \n{input_data}"}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    parsed_data, raw_content = json_parse(response)
    return parsed_data, raw_content, response['usage']

def calc_chunk_size(base:int, text_length:int) -> int:
    q = text_length / base
    divisor = ceil(q) if q < 4 else floor(q)
    chunk_size = ceil(text_length / divisor)
    return chunk_size


final_reconstruction = []
final_keywords = []
current_overview = ""
total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

# 1. 청크 나누기 전, 전체 데이터를 바탕으로 Overview 및 Keywords 생성
print("Generating overview and keywords for the entire text...")

# 전체 텍스트에 대해 한 번만 호출 (프롬프트는 동일하게 사용하거나 필요시 조정 가능)
# 단, 모델의 n_ctx 범위 내에 STT_INPUT_DATA가 들어와야 합니다.
summary_res, summary_raw, summary_usage = get_response(SUMMARIZATION_PROMPT, STT_INPUT_DATA)

if summary_res:
    final_keywords = summary_res.get('keywords', [])
    current_overview = summary_res.get('overview', "")
    
    # 요약 단계에서 발생한 토큰 사용량 합산
    total_usage['prompt_tokens'] += summary_usage['prompt_tokens']
    total_usage['completion_tokens'] += summary_usage['completion_tokens']
    total_usage['total_tokens'] += summary_usage['total_tokens']

# 2. 본문(Reconstruction) 처리를 위한 청크 분할
chunks = []
start_idx = 0
text_len = len(STT_INPUT_DATA)
chunk_size = calc_chunk_size(5000, text_len)

while start_idx < text_len:
    if text_len - start_idx <= chunk_size:
        chunks.append(STT_INPUT_DATA[start_idx:])
        break
    
    end_idx = start_idx + chunk_size
    next_newline = STT_INPUT_DATA.find('\n', end_idx - 1)
    
    if next_newline == -1:
        chunks.append(STT_INPUT_DATA[start_idx:])
        break
    else:
        chunks.append(STT_INPUT_DATA[start_idx:next_newline + 1])
        start_idx = next_newline + 1

# 3. 청크별 추론 (Reconstruction만 추출)
for i, chunk in enumerate(chunks):
    print(f"inference (reconstruction): {i+1}/{len(chunks)}, len(chunk): {len(chunk)}")
    
    parsed_res, raw_content, usage = get_response(RECONSTRUCTION_PROMPT, chunk)
    
    if parsed_res:
        recon_data = parsed_res.get('reconstruction', [])
        if isinstance(recon_data, list):
            final_reconstruction.extend(recon_data)
        elif isinstance(recon_data, dict):
            final_reconstruction.append(recon_data)

    total_usage['prompt_tokens'] += usage['prompt_tokens']
    total_usage['completion_tokens'] += usage['completion_tokens']
    total_usage['total_tokens'] += usage['total_tokens']

# 4. 최종 결과 조립
final_result = {
    "keywords": list(set(final_keywords)), # 중복 제거
    "overview": current_overview,
    "reconstruction": final_reconstruction
}
assign_indices_from_reconstruction(final_result)

time3 = time.time()

print("=====================================")
print(f"작업 ID (Job ID): {JOB_ID}!")  # 이 줄을 추가하세요
print(f"모델 로드: {time2-time1:.2f}초")
print(f"추론: {time3-time2:.2f}초")
print(f"총 소요시간: {time3-time1:.2f}초")
print(f"입력 토큰: {total_usage['prompt_tokens']}")
print(f"출력 토큰: {total_usage['completion_tokens']}")
print(f"토큰 합계: {total_usage['total_tokens']}")
print("-------------------------------------")
print("AI 분석 결과 (JSON):")
print(json.dumps(final_result, ensure_ascii=False, indent=2))
print("=====================================")


# main_task.py (요약 실행 파일)
from send_result import send_json_to_server # 작성하신 코드를 import

# 1. 작업 식별자 결정 (예: 20260406_001)
current_job_id = JOB_ID


# 3. 서버로 전송
success = send_json_to_server(final_result, current_job_id)

if success:
    print("서버 저장 완료")