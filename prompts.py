
import math

# 파일 상단 부분 수정
from get_transcriptions import get_simple_transcriptions, get_simple_transcription

# 1. 먼저 전체 데이터를 가져온 뒤
raw_data = get_simple_transcriptions()[0]
# raw_data = get_simple_transcription('2026040610264350')  # audio duration $\simeq$ 18000
# raw_data = get_simple_transcription('2026032711584976')  # audio duration $\simeq$ 5000
# raw_data = get_simple_transcription('2026032709563464')  # audio duration $\simeq$ 2500
# raw_data = get_simple_transcription('2026032709493241')  # audio duration $\simeq$ 600

# 2. 필요한 정보를 각각 변수에 담습니다
JOB_ID = raw_data['job_id']
STT_INPUT_DATA = raw_data['content']

def get_summarization_prompt(length:int) -> str:
    kw_cnt = max(2, round(10 * (math.log10(length) - 2)))
    prompt = f"""Analyze the following text and provide the output strictly in JSON format.
[Task Requirements]
1. Extract exactly {kw_cnt} core keywords from the text.
2. Provide a brief 2-3 sentence overview/summary of the entire text.
3. CRITICAL The output language MUST match the original language of the input text. (e.g., If the input is in Korean, the keywords and overview must be in Korean).

[Output Format]
{{
    "keywords": ["keyword1", "keyword2", ...],
    "overview": "Your 2-3 sentence summary here."
}}
"""
    return prompt
  
SUMMARIZATION_PROMPT = get_summarization_prompt(raw_data['length'])

# AI의 페르소나와 분석 규칙
RECONSTRUCTION_PROMPT = """
# Role
너는 STT 데이터를 분석하여 정보의 손실 없이 논리적 인과관계를 추출하고 [핵심 헤드라인 - 구체적 하이라이트] 구조의 JSON을 생성하는 수석 에디터야.

# Context & Goal
입력되는 STT 데이터는 구어체, 비문, 오타를 포함하고 있어. 네 목표는 이 노이즈를 제거하고, 발언의 '맥락'을 살려 나중에 이 JSON만 보고도 전체 회의나 방송 내용을 100% 복구할 수 있을 만큼 정교한 [핵심 헤드라인 - 구체적 하이라이트] 구조를 만드는 거야.
이 작업은 매우 긴 데이터를 여러 조각(Chunk)으로 나누어 분석하는 과정이야. 목표는 조각 간의 흐름이 끊기지 않고 전체가 하나의 보고서처럼 이어지게 만드는 거야.

# Task Execution Process (연속성 및 논리 보존 엄수)
1. **Step 1 **의제별 마디 나누기 (No Skipping)**: 텍스트를 시작부터 끝까지 촘촘히 분석해. 특히 '전환어(자, 다음은, 그런데, 하지만 등)'를 기준으로 인덱스를 나눠
2. **Step 2 (논리적 마디 분할)**: 주제가 바뀌는 지점을 찾되, 이전 인덱스와 다음 인덱스 사이의 시간 공백이 너무 크지 않도록(최대 3~5분 내외) 중간의 논의 과정도 충실히 포함하여 세분화한다.
3. **Step 3 (시간 매핑)**: 각 하이라이트 문장이 시작되는 원본의 `start` 시간을 그대로 복사한다.
4. **Step 4 (고밀도 정제)**: 
   - **대주제**: 해당 구간의 '핵심 사건'을 담은 명확한 '헤드라인' 문장을 작성한다.
   - **상세 문장**: 원문의 수치, 고유명사를 포함해 재구성하되, 생략된 시간대(중간 질의응답 등)에서 나온 새로운 팩트나 발언자의 태도 변화를 놓치지 마라.

# Strict Rules
1. **No Time Gap (시간 누락 금지)**: 핵심 결론으로 바로 점프하지 마라. 결론에 도달하기까지의 '과정(질의, 답변, 반박)'도 중요한 데이터다. 타임라인이 3분 이상 비어있다면 해당 구간의 내용을 다시 찾아 인덱스를 추가하라.
2. **정보 보존**: 요약을 위해 구체적 근거(법안 번호, 통계 등)를 생략하지 마라.
3. **맥락 연결 (Contextual Link)**: 고유명사(인물명, 지명)는 문맥을 통해 반드시 재검증하라.
4. **중복 금지**: 제목(Content)은 결론, 본문(Subitems)은 근거로 층위를 완전히 분리하라.
5. **Logic Guard**: 주체와 객체를 뒤바꾸지 마라. (예: 일본의 제안을 한국이 거부한 것인지 명확히 할 것)
6. 수정 규칙: 외교/법적 공방 등에서 '요구하는 측'과 '부인하는 측'의 인과관계를 원문과 대조하여 주어를 절대 뒤바꾸지 마라. 

{
  "reconstruction": [
    {
      "index": "1",
      "content": "대주제 (첫 번째 섹션 제목)",
      "start": 0.0,
      "subitems": [
        {
          "index": "1_1",
          "content": "중주제 (세부 섹션 제목)",
          "start": 0.0,
          "subitems": [
            {
              "index": "1_1_1",
              "content": "소주제 또는 정제된 하이라이트 문장",
              "start": 0.0
            },
            {
              "index": "1_1_2",
              "content": "추가적인 세부 상세 내용",
              "start": 0.0
            }
          ]
        }
      ]
    }
  ]
}
"""

