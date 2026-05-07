import os
import time
import json
import re
from math import ceil, floor
from llama_cpp import Llama

from send_result import send_json_to_server
from reconstruction_indices import assign_indices_from_reconstruction
from prompts import (
    SUMMARIZATION_PROMPT,
    RECONSTRUCTION_PROMPT,
    STT_INPUT_DATA,
    JOB_ID,
    get_summarization_prompt,
    get_aggregation_prompt,
    keyword_count_for_length,
)  # 프롬프트 파일 임포트 # 임포트 부분에 JOB_ID 추가

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

MODEL_PATH = "models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
#"models/Qwen3.5-9B-Q8_0.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"파일을 찾을 수 없습니다.")
    print(f"'{MODEL_PATH}' 경로에 파일이 있는지, 파일명이 정확한지 확인해주세요.")
    exit(1)

print(f"loading {MODEL_PATH} on the GPU")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=int(261344*0.0625),   
    # n_ctx=int(261344*0.75),
    verbose=False     
)

time2 = time.time()

print("\n model loaded\n")


def get_response(system_prompt, input_data, max_tokens=-1, temperature=0.7):
    user_content = f"다음 텍스트를 처리해줘: \n{input_data}"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        # response_format={"type": "json_object"},
    )

    parsed_data, raw_content = json_parse(response)
    return parsed_data, raw_content, response['usage']

def calc_chunk_size(base:int, text_length:int) -> int:
    q = text_length / base
    divisor = ceil(q) if q < 4 else floor(q)
    chunk_size = ceil(text_length / divisor)
    return chunk_size


def split_stt_into_chunks(text: str, base: int = 15000) -> list[str]:
    """재구성·요약 청킹에 동일 규칙 사용 (줄 경계 우선)."""
    chunks: list[str] = []
    start_idx = 0
    text_len = len(text)
    chunk_size = calc_chunk_size(base, text_len)

    while start_idx < text_len:
        if text_len - start_idx <= chunk_size:
            chunks.append(text[start_idx:])
            break

        end_idx = start_idx + chunk_size
        next_newline = text.find("\n", end_idx - 1)

        if next_newline == -1:
            chunks.append(text[start_idx:])
            break
        chunks.append(text[start_idx : next_newline + 1])
        start_idx = next_newline + 1

    return chunks


def build_aggregation_user_input(
    partial_overviews: list[str],
    partial_keyword_lists: list[list[str]],
) -> str:
    parts: list[str] = []
    for i, (ov, kws) in enumerate(zip(partial_overviews, partial_keyword_lists), start=1):
        kw_line = json.dumps(kws, ensure_ascii=False)
        parts.append(f"[청크 {i}]\noverview:\n{ov}\nkeywords:\n{kw_line}")
    return "\n\n".join(parts)


final_reconstruction = []
final_keywords: list[str] = []
current_overview = ""
total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

def count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())

segment_count = count_lines(STT_INPUT_DATA)
print(f"segment_count(non-empty lines): {segment_count}")

# 긴 원고(>15행): 요약·재구성에 동일 청크 리스트 사용
chunks_for_long: list[str] | None = None

# 1. Overview 및 Keywords (짧은 원고: 1회 / 긴 원고: 청크별 → 집계)
if segment_count <= 15:
    print("Generating overview and keywords (single pass, full text)...")
    summary_res, summary_raw, summary_usage = get_response(SUMMARIZATION_PROMPT, STT_INPUT_DATA)
    if summary_res:
        final_keywords = summary_res.get("keywords", []) or []
        current_overview = summary_res.get("overview", "") or ""
        total_usage["prompt_tokens"] += summary_usage["prompt_tokens"]
        total_usage["completion_tokens"] += summary_usage["completion_tokens"]
        total_usage["total_tokens"] += summary_usage["total_tokens"]
else:
    print("Generating overview and keywords (per-chunk map → reduce)...")
    chunks_for_long = split_stt_into_chunks(STT_INPUT_DATA)
    partial_overviews: list[str] = []
    partial_keyword_lists: list[list[str]] = []

    for i, chunk in enumerate(chunks_for_long):
        print(f"  summarization chunk {i+1}/{len(chunks_for_long)}, len={len(chunk)}")
        prompt = get_summarization_prompt(len(chunk))
        summary_res, summary_raw, summary_usage = get_response(prompt, chunk)
        if summary_res:
            partial_keyword_lists.append(summary_res.get("keywords", []) or [])
            partial_overviews.append(summary_res.get("overview", "") or "")
        else:
            partial_keyword_lists.append([])
            partial_overviews.append("")
        total_usage["prompt_tokens"] += summary_usage["prompt_tokens"]
        total_usage["completion_tokens"] += summary_usage["completion_tokens"]
        total_usage["total_tokens"] += summary_usage["total_tokens"]

    if len(chunks_for_long) == 1:
        print("Single summarization chunk detected. Skipping aggregation.")
        final_keywords = partial_keyword_lists[0] if partial_keyword_lists else []
        current_overview = partial_overviews[0] if partial_overviews else ""
    else:
        target_kw = keyword_count_for_length(len(STT_INPUT_DATA))
        agg_prompt = get_aggregation_prompt(target_kw)
        agg_user = build_aggregation_user_input(partial_overviews, partial_keyword_lists)
        merged_res, merged_raw, merged_usage = get_response(agg_prompt, agg_user, temperature=0.2)
        if merged_res:
            final_keywords = merged_res.get("keywords", []) or []
            current_overview = merged_res.get("overview", "") or ""
        total_usage["prompt_tokens"] += merged_usage["prompt_tokens"]
        total_usage["completion_tokens"] += merged_usage["completion_tokens"]
        total_usage["total_tokens"] += merged_usage["total_tokens"]

if segment_count <= 15:
    # transcription 짧으면(15행 이하) 발화 내용 그대로 reconstruction 필드 채움
    print("Short script detected (<=15 lines). Not using LLM for reconstruction.")
    for line in STT_INPUT_DATA.splitlines():
        line = line.strip()
        if not line:
            continue

        # line format: "[0.0] 안녕하세요"
        m = re.match(r"^\[(?P<start>\d+(?:\.\d+)?)\]\s*(?P<content>.*)$", line)
        if not m:
            continue

        final_reconstruction.append(
            {
                "start": float(m.group("start")),
                "content": m.group("content"),
            }
        )

else:
    # transcription이 15행 초과하면 depth-3 forest 구조로 재구성 (요약과 동일 청크)
    assert chunks_for_long is not None
    # 2~3. 청크별 추론 (Reconstruction만 추출)
    for i, chunk in enumerate(chunks_for_long):
        print(f"inference (reconstruction): {i+1}/{len(chunks_for_long)}, len(chunk): {len(chunk)}")

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
print("AI 분석 결과 (JSON):")
print(json.dumps(final_result, ensure_ascii=False, indent=2))
print("-------------------------------------")
print(f"작업 ID (Job ID): {JOB_ID}!")  # 이 줄을 추가하세요
print(f"모델 로드: {time2-time1:.2f}초")
print(f"추론: {time3-time2:.2f}초")
print(f"총 소요시간: {time3-time1:.2f}초")
print(f"입력 토큰: {total_usage['prompt_tokens']}")
print(f"출력 토큰: {total_usage['completion_tokens']}")
print(f"토큰 합계: {total_usage['total_tokens']}")
print("=====================================")


# 1. 작업 식별자 결정 (예: 20260406_001)
current_job_id = JOB_ID


# 로컬에 json 파일로 백업
with open(f"debug_{JOB_ID}.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

# 3. 서버로 전송
model_tag = os.path.splitext(os.path.basename(MODEL_PATH))[0]
success = send_json_to_server(final_result, current_job_id + f"_{model_tag}")

if success:
    print("서버 저장 완료")