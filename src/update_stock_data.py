import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
import time
import pandas as pd
from pykrx import stock

# =========================
# 설정
# =========================
DATA_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "data"
OUTPUT_SUFFIX = "_stock_data.csv"
ENCODING = "utf-8-sig"     # 엑셀 호환
SLEEP_SEC = 0.3            # API 과호출 방지
WINDOW_DAYS_INIT = 370     # 신규 생성 시 과거 1년+α
BACKFILL_CAL_DAYS_FOR_SHORT = 10  # 공매도잔고/비중 지연 공개 보정용 최소 재수집 구간(캘린더 일수)

REQ_COLS = [
    "일자","시가","고가","저가","종가","거래량","등락률",
    "기관 합계","기타법인","개인","외국인 합계","전체",
    "공매도","공매도비중","공매도잔고","공매도잔고비중"
]

KST = tz.gettz("Asia/Seoul")

# pykrx 내부 로그 묵음
for name in ["pykrx", "pykrx.website", "pykrx.website.comm", "pykrx.website.comm.util"]:
    logging.getLogger(name).disabled = True

# =========================
# 유틸
# =========================
def kst_today_date():
    return datetime.now(tz=KST).date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def empty_with_cols(cols):
    data = {}
    for c in cols:
        data[c] = pd.Series(dtype="object") if c == "일자" else pd.Series(dtype="float64")
    return pd.DataFrame(data)

def read_company_list(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"기업 리스트 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                name, ticker = [x.strip() for x in line.split(",", 1)]
            else:
                parts = line.replace("\t", " ").split()
                if len(parts) < 2:
                    logging.warning("기업 라인 파싱 불가: %s", line)
                    continue
                name, ticker = parts[0], parts[1]
            rows.append((name, ticker.zfill(6)))
    return rows

def last_trading_day_by_ohlcv(ticker: str, today):
    start = today - timedelta(days=30)
    df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        start = today - timedelta(days=90)
        df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: 최근 거래 자료 없음")
    return pd.to_datetime(df.index.max()).date()

def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_with_cols(["일자"])
    df = df.copy()
    if df.index.name is None:
        df.index.name = "일자"
    idx = pd.to_datetime(df.index, errors="coerce")
    df.index = idx
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "일자"}, inplace=True)
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def _normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """CSV/수집 데이터 모두 '일자'를 YYYY-MM-DD 문자열로 표준화."""
    if df is None or df.empty or "일자" not in df.columns:
        return df
    df = df.copy()
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def _to_float_clean(s):
    """문자 형태 수치를 안전하게 float로 변환: 쉼표/공백/% 제거"""
    try:
        if pd.isna(s):
            return 0.0
        x = str(s).strip()
        if x.endswith("%"):
            x = x[:-1]
        x = x.replace(",", "").replace(" ", "")
        return float(x)
    except Exception:
        return 0.0

def _pick_first_col(cols, candidates):
    """cols에서 candidates(우선순위 리스트) 중 처음으로 매칭되는 컬럼명 반환"""
    for key in candidates:
        for c in cols:
            if key in c:
                return c
    return None

def rename_investor_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자","기관 합계","기타법인","개인","외국인 합계","전체"])
    mapping = {
        "기관합계":"기관 합계", "외국인합계":"외국인 합계",
        "기관 합계":"기관 합계", "외국인 합계":"외국인 합계",
        "개인":"개인", "기타법인":"기타법인", "전체":"전체"
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    for need in ["기관 합계","기타법인","개인","외국인 합계","전체"]:
        if need not in df.columns:
            df[need] = 0
    return df[["일자","기관 합계","기타법인","개인","외국인 합계","전체"]]

def rename_short_cols(df: pd.DataFrame, is_balance=False) -> pd.DataFrame:
    """
    공매도 관련 표준화.
    - is_balance=False: 거래(볼륨) → ['공매도','공매도비중']
    - is_balance=True : 잔고     → ['공매도잔고','공매도잔고비중']
    ※ 퍼센트/쉼표 등 문자열 전처리 포함
    """
    if df is None or df.empty or "일자" not in df.columns:
        base = ["공매도잔고","공매도잔고비중"] if is_balance else ["공매도","공매도비중"]
        return empty_with_cols(["일자"] + base)

    dfc = df.copy()

    if is_balance:
        # pykrx 잔고 계열에서 흔한 컬럼들: '공매도잔고수량/금액', '잔고수량/금액', '공매도잔고비중'('잔고비중')
        amt_col = _pick_first_col(
            dfc.columns,
            ["공매도잔고", "잔고수량", "잔고금액", "잔고", "BAL_QTY", "BAL_AMT"]
        )
        rto_col = _pick_first_col(
            dfc.columns,
            ["공매도잔고비중", "잔고비중", "BAL_RTO", "비중"]  # '비중'이 여러 개일 수 있어도 우선순위상 뒤로 둠
        )

        dfc["공매도잔고"] = dfc[amt_col].apply(_to_float_clean) if amt_col else 0.0
        dfc["공매도잔고비중"] = dfc[rto_col].apply(_to_float_clean) if rto_col else 0.0

        keep = ["일자","공매도잔고","공매도잔고비중"]
        out = dfc[keep].copy()

    else:
        # 거래(볼륨) 계열: '공매도거래량/대금', '공매도비중'
        amt_col = _pick_first_col(
            dfc.columns,
            ["공매도거래량", "공매도", "거래량", "SV_QTY", "SV_AMT"]
        )
        rto_col = _pick_first_col(
            dfc.columns,
            ["공매도비중", "비중", "SV_RTO"]
        )

        dfc["공매도"] = dfc[amt_col].apply(_to_float_clean) if amt_col else 0.0
        dfc["공매도비중"] = dfc[rto_col].apply(_to_float_clean) if rto_col else 0.0

        keep = ["일자","공매도","공매도비중"]
        out = dfc[keep].copy()

    # 날짜 표준화
    out["일자"] = pd.to_datetime(out["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out

def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REQ_COLS]

# ---------- CSV 파일명 규칙: <이름>_<6자리티커>_stock_data.csv ----------
def csv_path_for(eng_name: str, ticker: str) -> Path:
    return DATA_DIR / f"{eng_name}_{str(ticker).zfill(6)}{OUTPUT_SUFFIX}"

def fetch_block(ticker: str, start_d, end_d) -> pd.DataFrame:
    s, e = yyyymmdd(start_d), yyyymmdd(end_d)
    ohlcv = stock.get_market_ohlcv(s, e, ticker)
    df1 = normalize_date_index(ohlcv)

    inv = stock.get_market_trading_volume_by_date(s, e, ticker)
    df2 = rename_investor_cols(normalize_date_index(inv))

    try:
        sv = stock.get_shorting_volume_by_date(s, e, ticker)
    except Exception:
        sv = pd.DataFrame()
    df3 = rename_short_cols(normalize_date_index(sv), is_balance=False)

    try:
        sb = stock.get_shorting_balance_by_date(s, e, ticker)
    except Exception:
        sb = pd.DataFrame()
    df4 = rename_short_cols(normalize_date_index(sb), is_balance=True)

    df = df1.merge(df2, on="일자", how="left") \
            .merge(df3, on="일자", how="left") \
            .merge(df4, on="일자", how="left")

    # 숫자 변환(퍼센트/쉼표 정규화는 각 rename_*에서 처리 완료)
    df = ensure_all_cols(df)
    for c in [c for c in df.columns if c != "일자"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df.sort_values("일자", ascending=False)

# =========================
# (유지) T·T-1의 공매도잔고/비중을 T-2 값으로 덮어쓰기
# =========================
def propagate_short_balance_from_t2(df: pd.DataFrame) -> pd.DataFrame:
    """
    최신 내림차순 정렬 기준으로,
    - 2행(2거래일 전)의 '공매도잔고/공매도잔고비중' 값을 읽어,
    - 0행(현거래일), 1행(전일)의 두 컬럼을 동일 값으로 덮어쓴다.
    - 2거래일 전 데이터가 없으면(행<3) 변경하지 않음.
    """
    cols = ["공매도잔고", "공매도잔고비중"]
    if df is None or df.empty or not all(c in df.columns for c in cols):
        return df
    df = df.copy()
    try:
        df["__dt__"] = pd.to_datetime(df["일자"], errors="coerce")
        df.sort_values("__dt__", ascending=False, inplace=True)
        df.drop(columns="__dt__", inplace=True)
    except Exception:
        df.sort_values("일자", ascending=False, inplace=True)

    if len(df) >= 3:
        ref = df.iloc[2][cols].values
        for idx in [0, 1]:
            df.iloc[idx, df.columns.get_indexer(cols)] = ref
    return df

# =========================
# 회사별 업데이트
# =========================
def upsert_company(eng_name: str, ticker: str, run_on_holiday: bool):
    out_path = csv_path_for(eng_name, ticker)
    today = kst_today_date()
    end_date = last_trading_day_by_ohlcv(ticker, today)

    # ---- 백필 윈도우 적용: 최근 N일 + last_have - 2일까지 후퇴 ----
    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        base = _normalize_date_col(base)
        last_have = None if base.empty else pd.to_datetime(base["일자"], errors="coerce").dt.date.max()

        start_date_base = (last_have + timedelta(days=1)) if last_have else (end_date - timedelta(days=WINDOW_DAYS_INIT))
        backfill_floor = end_date - timedelta(days=BACKFILL_CAL_DAYS_FOR_SHORT)
        if last_have:
            conservative_floor = last_have - timedelta(days=2)
            backfill_floor = min(backfill_floor, conservative_floor)

        start_date = min(start_date_base, backfill_floor)
    else:
        start_date = end_date - timedelta(days=WINDOW_DAYS_INIT)

    if (end_date < today) and (not run_on_holiday) and (not out_path.exists()):
        logging.info("[%s] 휴장일(run_on_holiday=False) → 신규 생성 스킵", eng_name)
        return False

    if start_date > end_date:
        logging.info("[%s] 최신 상태 (추가 데이터 없음)", eng_name)
        return False

    logging.info("[%s] 재수집 구간: %s ~ %s (티커 %s)", eng_name, start_date, end_date, ticker)
    df = fetch_block(ticker, start_date, end_date)
    df = _normalize_date_col(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        base = _normalize_date_col(base)

        # 병합: base(우선순위 낮음) + df(우선순위 높음)
        base["__pri__"] = 0
        df["__pri__"] = 1
        merged = pd.concat([base, df], ignore_index=True)

        # 최신→과거, 같은 일자는 __pri__가 높은(df) 값이 먼저 오도록
        merged["__dt__"] = pd.to_datetime(merged["일자"], errors="coerce")
        merged.sort_values(["__dt__", "__pri__"], ascending=[False, False], inplace=True, kind="mergesort")

        # 동일 '일자' 중복 제거: 첫 행(=가장 최신 & df 우선)이 남게
        merged.drop_duplicates(subset=["일자"], keep="first", inplace=True)
        merged.drop(columns=["__dt__", "__pri__"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        # T·T-1 ← T-2 값 덮어쓰기
        merged = propagate_short_balance_from_t2(merged)

        # 최종 정렬 및 저장
        merged["__dt__"] = pd.to_datetime(merged["일자"], errors="coerce")
        merged.sort_values("__dt__", ascending=False, inplace=True)
        merged.drop(columns="__dt__", inplace=True)
        merged.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 업데이트 → %s (총 %d행)", eng_name, out_path, len(merged))
    else:
        df = propagate_short_balance_from_t2(df)
        df.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 신규 생성 → %s (총 %d행)", eng_name, out_path, len(df))
    return True

# =========================
# 기업별 JSON + index.html 생성
#  - 단일 index.json 생성 없음
# =========================
def emit_per_ticker_json(companies, rows_limit=None):
    api_dir = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs" / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    cnt = 0
    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception:
            df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if rows_limit:
            df = df.head(int(rows_limit))

        item = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": [str(c) for c in df.columns],
            "rows": df.astype(str).values.tolist(),
            "row_count": int(len(df)),
        }
        out = api_dir / f"{name}_{str(ticker).zfill(6)}.json"
        out.write_text(json.dumps(item, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        cnt += 1
    logging.info("기업별 JSON 생성: %d개", cnt)

def emit_index_html(companies, rows_limit=None):
    import html as _html
    from string import Template  # ← f-string 중괄호 이스케이프 문제 방지를 위해 Template 사용
    docs_dir = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    generated = datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S %Z")

    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception:
            df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if rows_limit:
            df = df.head(int(rows_limit))

        columns = [str(c) for c in df.columns]
        rows = df.astype(str).values.tolist()

        thead = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
        tbody = "\n".join(
            "<tr>" + "".join(f"<td>{_html.escape(v)}</td>" for v in row) + "</tr>" for row in rows
        )
        sec_id = f"{name}_{str(ticker).zfill(6)}"

        # 🔹 차트용 섹션별 inline JSON (CORS 회피, 기존 구조/주석 유지)
        payload = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": [str(c).strip() for c in columns],  # 안전: 컬럼명 trim
            "rows": rows,  # 문자열 그대로(표와 동일 소스)
        }
        json_raw = json.dumps(payload, ensure_ascii=False)
        json_safe = json_raw.replace("</", "<\\/")  # </script> 차단

        # 🔹 표 + 차트 2개(세로 스택) + 섹션별 데이터 스크립트
        sections.append(f"""
<section id="{_html.escape(sec_id)}">
  <h2>{_html.escape(name)} ({str(ticker).zfill(6)})</h2>
  <div class="scroll">
    <table>
      <thead><tr>{thead}</tr></thead>
      <tbody>
      {tbody}
      </tbody>
    </table>
  </div>
  <p class="meta">rows: {len(rows)} · source: data/{_html.escape(csv_path.name)} · json: api/{_html.escape(sec_id)}.json</p>

  <div class="charts">
    <div id="chart-price-{_html.escape(sec_id)}" class="chart"></div>
    <div id="chart-flow-{_html.escape(sec_id)}" class="chart"></div>
  </div>

  <script id="data-{_html.escape(sec_id)}" type="application/json">{json_safe}</script>
</section>""")

    def _id_from(sec_html: str) -> str:
        try:
            return sec_html.split('id="', 1)[1].split('"', 1)[0]
        except Exception:
            return "section"

    nav = "".join(f'<a href="#{_id_from(s)}">{_id_from(s)}</a>' for s in sections)

    # 🔹 2개 차트(가격/지표, 수급/공매도)를 그리는 스크립트 포함
    html_template = Template("""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRX 기업별 데이터 테이블</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
  header { margin-bottom: 20px; }
  .meta-top { color:#666; font-size:14px; }
  .nav { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:8px; }
  .nav a { font-size:13px; text-decoration:none; color:#2563eb; }
  section { margin: 32px 0; }
  h2 { font-size: 18px; margin: 12px 0; }
  .scroll { overflow:auto; max-height: 60vh; border:1px solid #e5e7eb; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
  th:first-child, td:first-child { text-align: left; white-space: nowrap; }
  thead th { position: sticky; top:0; background:#fafafa; }
  .meta { color:#666; font-size:12px; }
  .charts { width: 100%; display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
  .chart { width: 100%; height: 560px; border:1px solid #e5e7eb; }
</style>
</head>
<body>
<header>
  <h1>KRX 기업별 데이터 테이블</h1>
  <div class="meta-top">생성 시각: $generated · 타임존: Asia/Seoul</div>
  <nav class="nav">$nav</nav>
</header>

$sections

<script>
// ===== 유틸 =====
function SMA(arr,n){const o=Array(arr.length).fill(null);let s=0,q=[];for(let i=0;i<arr.length;i++){const v=+arr[i]||0;q.push(v);s+=v;if(q.length>n)s-=q.shift();if(q.length===n)o[i]=s/n}return o}
function EMA(arr,n){const o=Array(arr.length).fill(null);const k=2/(n+1);let p=null;for(let i=0;i<arr.length;i++){const v=+arr[i]||0;p=(p==null)?v:v*k+p*(1-k);o[i]=p}return o}
function STD(arr,n){const o=Array(arr.length).fill(null);let q=[];for(let i=0;i<arr.length;i++){const v=+arr[i]||0;q.push(v);if(q.length>n)q.shift();if(q.length===n){const m=q.reduce((a,b)=>a+b,0)/n;const s2=q.reduce((a,b)=>a+(b-m)*(b-m),0)/n;o[i]=Math.sqrt(s2)}}return o}
function RSI(close,n=14){const o=Array(close.length).fill(null);let g=0,l=0;for(let i=1;i<close.length;i++){const ch=close[i]-close[i-1],G=ch>0?ch:0,L=ch<0?-ch:0;if(i<=n){g+=G;l+=L;if(i===n){const rs=(g/n)/((l/n)||1e-9);o[i]=100-100/(1+rs)}}else{g=(g*(n-1)+G)/n;l=(l*(n-1)+L)/n;const rs=g/(l||1e-9);o[i]=100-100/(1+rs)}}return o}
function MACD(close,f=12,s=26,sg=9){const ef=EMA(close,f),es=EMA(close,s),m=ef.map((v,i)=>v!=null&&es[i]!=null?v-es[i]:null),signal=EMA(m.map(v=>v??0),sg),h=m.map((v,i)=>v!=null&&signal[i]!=null?v-signal[i]:null);return{macd:m,signal,hist:h}}
function bbBands(close,n=20,k=2){const ma=SMA(close,n),sd=STD(close,n),u=ma.map((m,i)=>m!=null&&sd[i]!=null?m+k*sd[i]:null),l=ma.map((m,i)=>m!=null&&sd[i]!=null?m-k*sd[i]:null);return{ma,upper:u,lower:l}}
function nnum(x){if(x==null)return 0;return +String(x).replace(/,/g,'').replace(/\\s+/g,'').replace(/%/g,'')||0}
const str = (x)=> (x==null ? '' : String(x));
const cumsum = (arr)=>{let s=0; return arr.map(v=>{s += (+v||0); return s;});};
const safeMax = (arr)=> Math.max( ...(arr.map(v=>+v||0).filter(v=>isFinite(v)&&!isNaN(v))), 0 );

function toAsc(date, ...series){
  const N = date.length;
  if (N < 2) return [date, ...series];
  if (date[0] <= date[N-1]) return [date, ...series];
  const rev = a => a.slice().reverse();
  return [rev(date), ...series.map(rev)];
}

function idxOf(cols, primary, alts=[]){
  const i=cols.indexOf(primary);
  if(i>-1) return i;
  for(const a of alts){ const j=cols.indexOf(a); if(j>-1) return j; }
  return -1;
}

function showError(secId,msg){
  for (const side of ['chart-price-','chart-flow-']){
    const el = document.getElementById(side+secId);
    if (el) el.innerHTML = '<div style="padding:12px;color:#b91c1c;font-size:13px">'+msg+'</div>';
  }
}

// ===== 렌더링 =====
function renderOne(secId){
  const tag=document.getElementById('data-'+secId);
  if(!tag){ showError(secId,'섹션 데이터가 없습니다.'); return; }
  let j=null; try{ j=JSON.parse(tag.textContent); }catch(e){ showError(secId,'섹션 데이터 파싱 실패: '+e); return; }

  const cols=(j.columns||[]).map(c=>String(c).trim());

  const iDate=idxOf(cols,'일자',['\\ufeff일자','DATE','date']),
        iOpen=idxOf(cols,'시가',['Open','open']),
        iHigh=idxOf(cols,'고가',['High','high']),
        iLow =idxOf(cols,'저가',['Low','low']),
        iClose=idxOf(cols,'종가',['Close','close']),
        iVol =idxOf(cols,'거래량',['Volume','volume']),
        iFor =idxOf(cols,'외국인 합계',['외국인합계','외인합계']),
        iInst=idxOf(cols,'기관 합계',['기관합계']),
        iShortR =idxOf(cols,'공매도비중',['공매도 비중','공매도 거래량 비중','비중','(공매도)비중']),
        iShortBR=idxOf(cols,'공매도잔고비중',['공매도 잔고 비중','공매도잔고비중(%)','공매도잔고 비중(%)','잔고비중','잔고 비중','공매도잔고비율','잔고비율']);

  if([iDate,iOpen,iHigh,iLow,iClose].some(i=>i<0)){ showError(secId,'필수 컬럼 누락'); return; }
  const rows=j.rows||[]; if(!rows.length){ showError(secId,'시계열 행이 없습니다.'); return; }

  let date   = rows.map(r=>str(r[iDate]));
  let open   = rows.map(r=>nnum(r[iOpen]));
  let high   = rows.map(r=>nnum(r[iHigh]));
  let low    = rows.map(r=>nnum(r[iLow]));
  let close  = rows.map(r=>nnum(r[iClose]));
  let vol    = (iVol>=0)? rows.map(r=>nnum(r[iVol])): rows.map(_=>0);
  let foreign= (iFor>=0)? rows.map(r=>nnum(r[iFor])): rows.map(_=>0);
  let inst   = (iInst>=0)? rows.map(r=>nnum(r[iInst])): rows.map(_=>0);
  let shortR = (iShortR>=0)? rows.map(r=>nnum(r[iShortR])): rows.map(_=>0);
  let shortBR= (iShortBR>=0)? rows.map(r=>nnum(r[iShortBR])): rows.map(_=>0);

  [date, open, high, low, close, vol, foreign, inst, shortR, shortBR] =
    toAsc(date, open, high, low, close, vol, foreign, inst, shortR, shortBR);

  const ma20=SMA(close,20), ma60=SMA(close,60), ma120=SMA(close,120);
  const bb=bbBands(close,20,2);
  const rsi=RSI(close,14);
  const {macd,signal,hist}=MACD(close,12,26,9);

  // 차트 1: 가격/지표
  const layout1={
    grid:{rows:3,columns:1,pattern:'independent',roworder:'top to bottom'},
    xaxis:{domain:[0,1], rangeslider:{visible:false}, showspikes:true, spikemode:'across'},
    yaxis:{domain:[0.35,1.00], title:'주가 (원)', tickformat:',', showspikes:true},
    xaxis2:{anchor:'y2', showspikes:true},
    yaxis2:{domain:[0.18,0.30], title:'RSI', range:[0,100], tickvals:[30,70], showgrid:true},
    xaxis3:{anchor:'y3', showspikes:true},
    yaxis3:{domain:[0.00,0.15], title:'MACD'},
    legend:{orientation:'h', y:1.02, x:0.5, xanchor:'center'},
    margin:{t:40,l:60,r:40,b:30},
    hovermode:'x unified',
    plot_bgcolor:'#ffffff', paper_bgcolor:'#ffffff'
  };

  const traces1=[
    {type:'candlestick',x:date,open,high,low,close,name:'주가',
     increasing:{line:{color:'#ef4444'}}, decreasing:{line:{color:'#3b82f6'}} },
    {type:'scatter',mode:'lines',x:date,y:ma20,name:'MA20', line:{width:1.5}},
    {type:'scatter',mode:'lines',x:date,y:ma60,name:'MA60', line:{width:1.5}},
    {type:'scatter',mode:'lines',x:date,y:ma120,name:'MA120', line:{width:1.5}},
    {type:'scatter',mode:'lines',x:date,y:bb.upper,name:'BB상단', visible:'legendonly', line:{dash:'dot', width:1}},
    {type:'scatter',mode:'lines',x:date,y:bb.lower,name:'BB하단', visible:'legendonly', line:{dash:'dot', width:1}},
    {type:'scatter',mode:'lines',x:date,y:rsi,name:'RSI(14)',xaxis:'x2',yaxis:'y2'},
    {type:'bar',x:date,y:hist,name:'MACD Hist',xaxis:'x3',yaxis:'y3'},
    {type:'scatter',mode:'lines',x:date,y:macd,name:'MACD',xaxis:'x3',yaxis:'y3'},
    {type:'scatter',mode:'lines',x:date,y:signal,name:'Signal',xaxis:'x3',yaxis:'y3'},
  ];

  Plotly.newPlot('chart-price-'+secId, traces1, layout1, {responsive:true, displaylogo:false});

  // 차트 2: 수급/공매도
  const layout2={
    yaxis:{title:'누적 순매수', tickformat:',', showgrid:true},
    yaxis2:{title:'공매도 비율(%)', overlaying:'y', side:'right',
           range:[0, Math.max(1, Math.max(...shortBR, ...shortR, 0)*1.2)]},
    margin:{t:40,l:60,r:50,b:30},
    hovermode:'x unified',
    legend:{orientation:'h', y:1.08, x:0.5, xanchor:'center'},
    plot_bgcolor:'#ffffff'
  };

  const instCum = cumsum(inst);
  const foreignCum = cumsum(foreign);

  const traces2=[
    {type:'scatter',mode:'lines',x:date,y:instCum,   name:'기관 누적'},
    {type:'scatter',mode:'lines',x:date,y:foreignCum,name:'외국인 누적'},
    {type:'scatter',mode:'lines',x:date,y:shortR,    name:'공매도비중(%)',yaxis:'y2', line:{dash:'dot'}},
    {type:'scatter',mode:'lines',x:date,y:shortBR,   name:'공매도잔고비중(%)',yaxis:'y2'},
  ];

  Plotly.newPlot('chart-flow-'+secId, traces2, layout2, {responsive:true, displaylogo:false});
}

(function main(){
  const ids=Array.from(document.querySelectorAll('section[id]')).map(s=>s.id);
  for(const id of ids){ try{ renderOne(id); }catch(e){ showError(id,'렌더링 오류: '+e); } }
})();
</script>

<footer style="margin-top:40px;color:#666;font-size:12px">
  Published via GitHub Pages · Per-ticker JSON: /api/*.json
</footer>
</body>
</html>""")

    html_doc = html_template.substitute(
        generated=generated,
        nav=nav,
        sections="".join(sections) if sections else "<p>표시할 데이터가 없습니다.</p>",
    )

    (docs_dir / "index.html").write_text(html_doc, encoding="utf-8")
    logging.info("index.html 생성 완료 → %s", docs_dir / "index.html")

# =========================
# 엔트리포인트
# =========================
def main():
    parser = argparse.ArgumentParser(description="KRX 일별 데이터 수집 & CSV 업데이트")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true", help="휴장일에도 실행 (true/false)")
    parser.add_argument("--rows-limit", default=os.getenv("ROWS_LIMIT", "").strip(),
                        help="HTML/JSON 포함 최대 행 수 (빈 값이면 전량)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")
    rows_limit = None if args.rows_limit in ("", "0", "none", "None") else int(args.rows_limit)

    try:
        companies = read_company_list(Path(args.company_list))
    except Exception as e:
        logging.exception("기업 리스트 로딩 실패: %s", e)
        return

    if not companies:
        logging.warning("수집 대상 기업이 없습니다.")
        return

    changed = False
    for name, ticker in companies:
        try:
            time.sleep(SLEEP_SEC)
            updated = upsert_company(name, ticker, run_on_holiday)
            changed = changed or updated
        except Exception as e:
            logging.exception("[%s,%s] 처리 중 오류: %s", name, ticker, e)

    if changed:
        logging.info("변경사항 존재 → 커밋 단계에서 반영됩니다.")
    else:
        logging.info("변경사항 없음.")

    # 단일 index.json은 만들지 않음 → 기업별 JSON + index.html만 생성
    emit_per_ticker_json(companies, rows_limit=rows_limit)
    emit_index_html(companies, rows_limit=rows_limit)

if __name__ == "__main__":
    main()
