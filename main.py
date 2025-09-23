from datetime import datetime, timezone, timedelta
import time
import random
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import pandas as pd
import os
from dotenv import load_dotenv

# 1. get yesterday date
date = date = (
    (datetime.now(timezone(timedelta(hours=9))) - timedelta(days=1))
    .date()
    .isoformat()
    .split("T")[0]
)


def custom_request(url):
    # 브라우저 유사 헤더
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        # 목록 페이지 → 상세 페이지 흐름을 재현하려면 Referer를 실제 이전 페이지로 설정
        "Referer": url,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # 세션 + 재시도(403/429/5xx 백오프)
    retry = Retry(
        total=5,
        backoff_factor=1,  # 1,2,4,8,...
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)

    with requests.Session() as s:
        s.headers.update(headers)
        s.mount("https://", adapter)
        s.mount("http://", adapter)

        # 너무 빠른 요청을 피하기 위한 랜덤 지연
        time.sleep(random.uniform(0.8, 1.6))

        res = s.get(url, timeout=15)
        if res.status_code != 200:
            print(f"crawling failed from url : {url}!")
            return

    soup = BeautifulSoup(res.text, "html.parser")
    return soup


# 2. get ids
def get_last_page(date):
    baseUrl = f"https://coinreaders.com/search.html?submit=submit&search=%EC%9D%B4%EB%8D%94%EB%A6%AC%EC%9B%80&search_exec=all&news_order=1&search_section=all&search_and=1&search_start_day={date}&search_end_day={date}&page="

    page = 1

    url = f"{baseUrl}{page}"

    soup = custom_request(url)

    last_page = None
    try:
        last_page = int(soup.select(".paging a")[-1].text)
    except:
        last_page = 1
    return last_page


last_page = get_last_page(date)

baseUrl = f"https://coinreaders.com/search.html?submit=submit&search=%EC%9D%B4%EB%8D%94%EB%A6%AC%EC%9B%80&search_exec=all&news_order=1&search_section=all&search_and=1&search_start_day={date}&search_end_day={date}&page="

full_ids = []
for page in range(1, last_page + 1):
    soup = custom_request(f"{baseUrl}{page}")
    ids = [a.get("href") for a in soup.select(".search_result_list_box > dl > dt > a")]
    full_ids.extend(ids)

file = open(f"ids/{date}.txt", "w")
file.write(",".join(ids))
file.close()

print("========== get ids finished ==========")

# 3. get news
filename = f"ids/{date}.txt"
file = open(filename, "r")
ids = file.read().split(",")

baseUrl = "https://www.coinreaders.com"

news_list = []
for id in ids:
    url = f"{baseUrl}{id}"

    # 브라우저 유사 헤더
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        # 목록 페이지 → 상세 페이지 흐름을 재현하려면 Referer를 실제 이전 페이지로 설정
        "Referer": baseUrl + id,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # 세션 + 재시도(403/429/5xx 백오프)
    retry = Retry(
        total=5,
        backoff_factor=1,  # 1,2,4,8,...
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)

    with requests.Session() as s:
        s.headers.update(headers)
        s.mount("https://", adapter)
        s.mount("http://", adapter)

        # 너무 빠른 요청을 피하기 위한 랜덤 지연
        time.sleep(random.uniform(0.8, 1.6))

        res = s.get(url, timeout=15)
        if res.status_code != 200:
            print(f"crawling failed from page {page}!")
            break

    soup = BeautifulSoup(res.text, "html.parser")
    temp = "".join(
        [
            element.text
            for element in soup.select("#textinput > p")
            if element.text != "\xa0"
        ]
    )
    if temp == "":
        temp = soup.select_one("#textinput").text
    news_list.append(temp)

df = pd.DataFrame(
    [{"news": n, "date": date.replace("-", ".")} for n in news_list],
    columns=["news", "date"],
)
df.to_csv(f"news_list/{date}.csv", index=False)

print("========== get news finished ==========")

# 4. llm 감성 분석
load_dotenv(verbose=True)


# 1) 출력 스키마 정의
class Sentiment(BaseModel):
    label: Literal["negative", "neutral", "positive"] = Field(
        ..., description="감성 라벨"
    )


parser = JsonOutputParser(pydantic_object=Sentiment)

# 2) 프롬프트 템플릿
system_text = """뉴스 감성 분석기다.
- 사실 전달형/중립적 보도 톤은 기본값으로 neutral 처리한다.
- 명백한 악재(적자 확대, 리콜, 규제 불이익, 급락 등)는 negative.
- 명백한 호재(사상 최대 실적, 대규모 투자 유치/수주, 급등 등)는 positive.
- 혼재할 경우 기사 전체 톤 기준으로 단 하나의 라벨만 선택한다.
- 반드시 JSON 한 줄만 출력한다.
출력 스키마: {{"label":"negative|neutral|positive"}}
"""

human_text = """
본문: {body}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_text),
        ("human", human_text),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 3) 모델 정의 (환경변수 OPENAI_API_KEY 필요)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY
)  # 임의 모델. 교체 가능

# 4) 체인 구성: Prompt -> LLM -> JSON 파서
chain = prompt | llm | parser


def parseResponse(label):
    if label == "negative":
        return -1
    elif label == "neutral":
        return 0
    else:
        return 1


df = pd.read_csv(f"news_list/{date}.csv")

news_list = df["news"].tolist()

labels = [0] * len(news_list)

for idx, news in enumerate(news_list):
    labels[idx] = parseResponse(chain.invoke(news)["label"])
    print(f"llm request : {idx+1}/{len(news_list)} finished...")

df["label"] = labels
df.to_csv(f"news_list_group_by_date/{date.replace("-",".")}.csv", index=False)

print("========== get news sentiment analysis finished ==========")
