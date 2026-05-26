import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as legacy_genai
from google import genai
from google.genai import types
import logging
import traceback
import os
import json
import re
import asyncio
import time
import urllib.request
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from telegram.error import BadRequest
from flask import Flask
from threading import Thread

# --- KONFİGÜRASYON ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WATCHLIST_FILE = "watchlist.json"

# Gemini Kurulumu
legacy_genai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# Loglama
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- VERİ YÖNETİMİ ---
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return []

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --- FIRSAT TARAYICI ----------------------------------------
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# BIST'in en likit 40 hissesi (genişletilebilir)
BIST_SCAN_LIST = [
    # Bankacılık
    "AKBNK","GARAN","ISCTR","YKBNK","HALKB","VAKBN","SKBNK","ALBRK","TSKB","QNBFB",
    # Holding
    "KCHOL","SAHOL","AGHOL","ENKAI","DOHOL","TKFEN","BRISA","MPARK",
    # Savunma & Teknoloji
    "ASELS","LOGO","NETAS","ARENA","INTEM",
    # Havacılık & Ulaşım & Turizm
    "THYAO","PGSUS","TAVHL","CLEBI",
    # Otomotiv & Makine
    "TOASO","FROTO","DOAS","ARCLK","OTKAR","ASUZU",
    # Perakende & Tüketim
    "BIMAS","MGROS","SOKM","MAVI","ULKER","AEFES","CCOLA","TATGD","BANVT",
    # Enerji & Petrokimya
    "TUPRS","PETKM","AKENR","ZOREN","EUPWR","ENJSA",
    # Demir-Çelik & Metal
    "EREGL","KRDMD","BRSAN","ISDMR","SARKY","CEMTS","KARDEM",
    # Cam & Kimya
    "SISE","TRKCM","SODA","ALKIM","BAGFS",
    # Tekstil
    "SASA","KORDS","BRMEN",
    # GYO (Gayrimenkul)
    "EKGYO","ALGYO","ISGYO","TRGYO","VKGYO","OZGYO",
    # İnşaat & Çimento
    "CIMSA","AKCNS","BSOKE","BTCIM","BUCIM",
    # Telekomünikasyon
    "TCELL","TTKOM","NTHOL",
    # Elektronik & Beyaz Eşya
    "VESBE","VESTL",
    # Diğer Likit
    "KOZAA","ALARK","DOHOL","GESAN","YATAS",
    # Küçük ve Orta Ölçekli Büyüme Hisseleri (Mid & Small Cap)
    "MIATK","ALFAS","YEOTK","KCAER","SMRTG","SDTTR","KONTR","REEDR","ASTOR","ODAS",
    "CANTE","HUNER","JANTS","EGEEN","TMSN","ALCAR","PKART","KRONT","FONET","HEKTS"
]

BIST_TICKERS_CACHE = {"time": 0, "list": []}

def get_all_bist_tickers():
    """IS Yatırım web sitesinden dinamik olarak tüm BIST hisse sembollerini çeker."""
    global BIST_TICKERS_CACHE
    if time.time() - BIST_TICKERS_CACHE["time"] < 86400 and BIST_TICKERS_CACHE["list"]: # 24 saat önbellek
        return BIST_TICKERS_CACHE["list"]
        
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            options = soup.find_all('option')
            symbols = []
            for opt in options:
                val = opt.get('value', '').strip()
                if val and 4 <= len(val) <= 6 and val.isalpha() and val.isupper():
                    symbols.append(val)
            if symbols:
                BIST_TICKERS_CACHE["time"] = time.time()
                BIST_TICKERS_CACHE["list"] = list(set(symbols))
                return BIST_TICKERS_CACHE["list"]
    except Exception as e:
        logging.warning(f"BIST listesi çekilemedi: {e}")
    
    return BIST_SCAN_LIST # Fallback


def quick_screen_calc(symbol: str, data: pd.DataFrame) -> dict | None:
    """
    Zaten indirilmiş olan DataFrame verisi (data) üzerinden 
    teknik ve temel kriterlere göre hızla puanlar (0-100).
    İlk geçmesi gerekli zorunlu filtre: RSI < 58 (aşırı alım bölgesinde değil)
    """
    try:
        ticker_sym = f"{symbol}.IS"
        if data.empty or len(data) < 15:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close  = data["Close"]
        high   = data["High"]
        low    = data["Low"]
        volume = data["Volume"]

        # ── RSI (Wilder EMA) ──
        rsi_period = min(14, max(1, len(close)-1))
        delta    = close.diff()
        gain     = delta.where(delta > 0, 0.0)
        loss     = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, 0.001)
        rsi_s    = 100 - (100 / (1 + rs))
        rsi      = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
        rsi_prev = float(rsi_s.iloc[-2]) if len(rsi_s) > 1 and not pd.isna(rsi_s.iloc[-2]) else 50.0

        # ── MACD ──
        exp1     = close.ewm(span=min(12, len(close)), adjust=False).mean()
        exp2     = close.ewm(span=min(26, len(close)), adjust=False).mean()
        macd_s   = exp1 - exp2
        signal_s = macd_s.ewm(span=min(9, len(close)), adjust=False).mean()
        macd_cur = float(macd_s.iloc[-1])
        macd_prv = float(macd_s.iloc[-2]) if len(macd_s) > 1 else macd_cur
        sig_cur  = float(signal_s.iloc[-1])
        sig_prv  = float(signal_s.iloc[-2]) if len(signal_s) > 1 else sig_cur

        # ── SMA200 ──
        sma50    = float(close.rolling(min(50, len(close)), min_periods=1).mean().iloc[-1])
        sma200   = float(close.rolling(min(200, len(close)), min_periods=1).mean().iloc[-1])
        price    = float(close.iloc[-1])

        # ── Hacim oranı ──
        vol_mean = float(volume.rolling(min(20, len(close)), min_periods=1).mean().iloc[-1])
        vol_last = float(volume.iloc[-1])
        vol_ratio = vol_last / vol_mean if vol_mean > 0 else 1.0

        # ─── PUANLAMA ────────────────────────────────────────────────────
        score = 0
        tags  = []

        # 1) RSI → max 30 pt  (ana kriter)
        if rsi < 30:
            score += 25
            tags.append("RSI Aşırı Satım")
        elif 30 <= rsi < 40:
            score += 30
            tags.append("RSI Dönüş Bölgesi")
        elif 40 <= rsi < 50:
            score += 20
            tags.append("RSI Toparlanıyor")
        elif 50 <= rsi < 58:
            score += 10
            tags.append("RSI Nötr")
        # RSI > 58 → zorunlu filtre
        if rsi > 58:
            return None

        # RSI yükseliyorsa (önceki güne göre) +5 bonus
        if rsi > rsi_prev:
            score += 5
            tags.append("RSI Yükseliyor")

        # 2) MACD → max 20 pt
        if macd_prv < sig_prv and macd_cur >= sig_cur:   # tam kesiş
            score += 20
            tags.append("MACD Yukarı Kesiyor")
        elif macd_cur > sig_cur:                          # zaten pozitif
            score += 12
            tags.append("MACD Pozitif")
        elif macd_cur > macd_prv and macd_cur < 0:        # negatif ama yükseliyor
            score += 6
            tags.append("MACD Toparlanıyor")

        # 3) Pozitif RSI uyuşmazlığı → +15 pt
        if len(close) >= 20:
            price_min_10 = float(close.iloc[-10:].min())
            rsi_min_10   = float(rsi_s.iloc[-10:].min())
            price_min_20 = float(close.iloc[-20:-10].min())
            rsi_min_20   = float(rsi_s.iloc[-20:-10].min())
            if price_min_10 < price_min_20 and rsi_min_10 > rsi_min_20:
                score += 15
                tags.append("Pozitif RSI Uyuşmazlığı")

        # 4) Hacim → max 15 pt (eşikler gerçekçi düzeyde)
        if vol_ratio >= 1.5:
            score += 15
            tags.append(f"Yüksek Hacim ({vol_ratio:.1f}x)")
        elif vol_ratio >= 1.0:
            score += 8
            tags.append(f"Normal Hacim ({vol_ratio:.1f}x)")
        elif vol_ratio >= 0.7:
            score += 3
            tags.append(f"Düşük Hacim ({vol_ratio:.1f}x)")

        # 5) Trend → max 10 pt
        if price > sma200:
            score += 10
            tags.append("SMA200 Üstünde")
        elif price > sma50:
            score += 5
            tags.append("SMA50 Üstünde")

        # Minimum eşik: 25 puan (düşük hacimli günlerde bile sonuç verir)
        if score < 25:
            return None

        # ─── TEMEL FİLTRELER (Sadece teknikten geçenlere uygulanır) ───
        try:
            ticker_info = yf.Ticker(ticker_sym).info
            info = ticker_info if isinstance(ticker_info, dict) else {}
            
            # 1. Öz sermaye negatif olmayacak
            bv = info.get("bookValue")
            if bv is not None and isinstance(bv, (int, float)) and bv <= 0:
                return None
                
            # 2. F/K < 25 (Büyüyen küçük hisseleri dışlamamak için limit esnetildi)
            fk = info.get("forwardPE") or info.get("trailingPE")
            if fk is not None and isinstance(fk, (int, float)) and fk >= 25:
                return None
                
            # 3. PD/DD <= 5 (Aşırı şişmemiş hisseler)
            pddd = info.get("priceToBook")
            if pddd is not None and isinstance(pddd, (int, float)) and pddd > 5:
                return None
                
            # 4. Yıllık kar büyümesi >= 20% (0.20)
            eg = info.get("earningsGrowth")
            if eg is not None and isinstance(eg, (int, float)) and eg < 0.20:
                return None
                
            # 5. Hisse Başına Kar (EPS) >= 0 (Zarar eden şirketler elenir)
            eps = info.get("trailingEps")
            if eps is not None and isinstance(eps, (int, float)) and eps < 0:
                return None
                
            # 6. PEG <= 2.0 (Büyüme/Fiyat dengesi)
            peg = info.get("pegRatio")
            if peg is not None and isinstance(peg, (int, float)) and peg > 2.0:
                return None
                
            # 7. Özkaynak Karlılığı (ROE) >= %15
            roe = info.get("returnOnEquity")
            if roe is not None and isinstance(roe, (int, float)) and roe < 0.15:
                return None
                
            # 8. Aktif Karlılık ROA >= %5 (BIST gerçeklerine uygun ve adil seviye)
            roa = info.get("returnOnAssets")
            if roa is not None and isinstance(roa, (int, float)) and roa < 0.05:
                return None

            # Eğer temel veriler uygunsa (veya yfinance eksik verse bile diğerlerini geçmişse)
            # F/K'ya göre bonus puan ver.
            if fk and isinstance(fk, (int, float)) and 0 < fk < 10:
                score += 10
                tags.append(f"F/K={round(fk,1)} Ucuz")

        except Exception as e:
            # Info çekilemezse (timeout vb.), hisseyi eleme ama log at
            logging.warning(f"Temel veri hatası ({symbol}): {e}")

        return {
            "symbol": symbol,
            "score":  score,
            "rsi":    round(rsi, 1),
            "macd":   round(macd_cur, 2),
            "vol":    round(vol_ratio, 2),
            "price":  round(price, 2),
            "tags":   tags,
        }
    except Exception as e:
        logging.warning(f"Tarama hesaplama hatası ({symbol}): {e}")
        return None

def quick_screen(symbol: str) -> dict | None:
    """Tekil kullanım destekli wrapper."""
    ticker_sym = f"{symbol}.IS"
    data = yf.download(ticker_sym, period="6mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return quick_screen_calc(symbol, data)

def run_market_scan(scan_list: list[str]) -> list[dict]:
    """Tüm BIST listesini tek bir çoklu yfinance çağrısıyla çok daha hızlı tarar."""
    results = []
    if not scan_list:
        return results
        
    try:
        tickers = [f"{sym}.IS" for sym in scan_list]
        # YFinance toplu indirme (Koyeb gibi sınırlı ortamlarda thread sayısını kısıtlıyoruz)
        df = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True, threads=10)
        
        is_multi = isinstance(df.columns, pd.MultiIndex)
        
        for sym in scan_list:
            ticker_sym = f"{sym}.IS"
            try:
                if is_multi:
                    # MultiIndex ('THYAO.IS', 'Close') vb. olduğu için direkt ilgili hisseyi seçebiliriz
                    if ticker_sym not in df.columns.levels[0]:
                        continue
                    stock_data = df[ticker_sym].dropna(how='all')
                else:
                    # Sadece 1 hisse geldiyse dümdüz dataframe'dir
                    stock_data = df.dropna(how='all')
                    
                res = quick_screen_calc(sym, stock_data)
                if res:
                    results.append(res)
            except Exception as e:
                logging.warning(f"Scan loop error ({sym}): {e}")
                
    except Exception as e:
        logging.error(f"Bulk download error: {e}")

    return sorted(results, key=lambda x: x["score"], reverse=True)

def custom_screen_calc(symbol: str, data: pd.DataFrame) -> dict | None:
    try:
        ticker_sym = f"{symbol}.IS"
        if data.empty or len(data) < 15:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close  = data["Close"]

        # RSI Calculation
        rsi_period = min(14, max(1, len(close)-1))
        delta    = close.diff()
        gain     = delta.where(delta > 0, 0.0)
        loss     = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, 0.001)
        rsi_s    = 100 - (100 / (1 + rs))
        rsi      = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

        if not (20 <= rsi <= 40):
            return None

        # Fundamental Filters
        ticker_info = yf.Ticker(ticker_sym).info
        info = ticker_info if isinstance(ticker_info, dict) else {}
        
        # 1. Özsermaye en az 2 TL olacak (sermaye kaybı/erimesi olmaması için alt limit)
        bv = info.get("bookValue")
        if bv is not None and isinstance(bv, (int, float)) and bv < 2.0:
            return None
            
        # 2. FD/FAVÖK (EV/EBITDA) <= 12 (Mevcutsa kontrol edilir, eksikse elenmez)
        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda is not None and isinstance(ev_ebitda, (int, float)) and ev_ebitda > 12.0:
            return None
            
        # 3. PD/DD (Fiyat/Defter Değeri) <= 2.5 (Mevcutsa kontrol edilir, eksikse elenmez)
        pddd = info.get("priceToBook")
        if pddd is not None and isinstance(pddd, (int, float)) and pddd > 2.5:
            return None

        price = float(close.iloc[-1])
        return {
            "symbol": symbol,
            "rsi": round(rsi, 1),
            "bv": round(bv, 2),
            "ev_ebitda": round(ev_ebitda, 2),
            "pddd": round(pddd, 2),
            "price": round(price, 2)
        }
            
    except Exception:
        return None

def run_custom_scan(scan_list: list[str]) -> list[dict]:
    results = []
    if not scan_list:
        return results
    try:
        tickers = [f"{sym}.IS" for sym in scan_list]
        df = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True, threads=10)
        is_multi = isinstance(df.columns, pd.MultiIndex)
        for sym in scan_list:
            ticker_sym = f"{sym}.IS"
            try:
                if is_multi:
                    if ticker_sym not in df.columns.levels[0]:
                        continue
                    stock_data = df[ticker_sym].dropna(how='all')
                else:
                    stock_data = df.dropna(how='all')
                res = custom_screen_calc(sym, stock_data)
                if res:
                    results.append(res)
            except:
                continue
    except:
        pass
    return sorted(results, key=lambda x: x["rsi"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- USTA ANALİZ MANTIĞI ---
def get_macro_context():
    return (
        "Türkiye Makro Durumu: Faizlerin yüksek olduğu bir dönemdeyiz. "
        "Borsaya alternatif getiriler (mevduat) güçlü. Bu ortamda şirketlerin borçluluğu "
        "(borç çevirme kapasitesi) ve nakit akışı hayati önem taşır. "
        "Yabancı yatırımcı takası ve hacimli kırılımlar yönü belirler."
    )

MARKET_CONTEXT_CACHE = {"time": 0, "data": None}

def get_market_context():
    global MARKET_CONTEXT_CACHE
    if time.time() - MARKET_CONTEXT_CACHE["time"] < 3600 and MARKET_CONTEXT_CACHE["data"] is not None:
        return MARKET_CONTEXT_CACHE["data"]
        
    try:
        index_data = yf.download("XU100.IS", period="1mo", interval="1d", progress=False, auto_adjust=True)
        if index_data.empty: return "Veri alinamadi"
        if isinstance(index_data.columns, pd.MultiIndex):
            index_data.columns = index_data.columns.get_level_values(0)
        current_idx = index_data['Close'].iloc[-1]
        prev_idx = index_data['Close'].iloc[-5]
        change = ((current_idx - prev_idx) / prev_idx) * 100
        trend = "BOĞA (Yükseliş)" if change > 0 else "AYI (Düşüş)"
        res = f"BIST100: {trend} (Haftalık Değişim: %{change:.2f})"
        MARKET_CONTEXT_CACHE["time"] = time.time()
        MARKET_CONTEXT_CACHE["data"] = res
        return res
    except:
        return "Endeks verisi alinamadi."

def calculate_indicators(data):
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    rsi_period = min(14, max(1, len(close)-1))
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 0.001)
    rsi_series = 100 - (100 / (1 + rs))
    
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 and not pd.isna(rsi_series.iloc[-2]) else 50.0
    
    sma20_s = close.rolling(window=min(20, len(close)), min_periods=1).mean()
    sma20 = float(sma20_s.iloc[-1])
    sma50 = float(close.rolling(window=min(50, len(close)), min_periods=1).mean().iloc[-1])
    sma200 = float(close.rolling(window=min(200, len(close)), min_periods=1).mean().iloc[-1])
    
    std20_s = close.rolling(window=min(20, len(close)), min_periods=1).std()
    std20 = float(std20_s.iloc[-1]) if not pd.isna(std20_s.iloc[-1]) else 0.0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_s = tr.ewm(alpha=1/rsi_period, min_periods=1, adjust=False).mean()
    atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    
    exp1 = close.ewm(span=min(12, len(close)), adjust=False).mean()
    exp2 = close.ewm(span=min(26, len(close)), adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=min(9, len(close)), adjust=False).mean()
    
    idx_5 = min(5, len(close)-1) if len(close) > 5 else (len(close)-1 if len(close) > 1 else 0)
    idx_20 = min(20, len(close)-1) if len(close) > 20 else (len(close)-1 if len(close) > 1 else 0)
    w_change = ((close.iloc[-1] - close.iloc[-1-idx_5]) / close.iloc[-1-idx_5]) * 100 if idx_5 > 0 else 0.0
    m_change = ((close.iloc[-1] - close.iloc[-1-idx_20]) / close.iloc[-1-idx_20]) * 100 if idx_20 > 0 else 0.0
    
    vol_mean_20 = float(volume.rolling(window=min(20, len(close)), min_periods=1).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1] / vol_mean_20) if vol_mean_20 > 0 else 1.0

    algo_signal = "NÖTR: Mevcut trend yatay/belirsiz."
    if rsi_prev < 35 and rsi >= rsi_prev and vol_ratio >= 1.0:
        algo_signal = "🟢 GÜÇLÜ AL: Dip bölgesinden hacimli dönüş yapıyor! (Aşırı satımdan çıkış)"
    elif rsi > 70 and vol_ratio < 0.9:
        algo_signal = "🔴 DİKKAT: Hacimsiz yükseliş. Kar satışları gelebilir (Geri çekilme riski)."
    elif rsi_prev > 65 and rsi < rsi_prev:
        algo_signal = "🔴 SAT SİNYALİ: Yükseliş gücü (momentum) zayıflıyor."
    elif close.iloc[-1] > sma200 and rsi > 50 and vol_ratio >= 0.8:
        algo_signal = "🟢 TREND POZİTİF: Ana trendin üzerinde güçlü duruş sürüyor."
    elif close.iloc[-1] < sma200 and rsi < 45:
        algo_signal = "🔴 TREND NEGATİF: Ana trendin altında zayıf görünüm devam ediyor."

    return {
        "rsi": round(rsi, 2),
        "rsi_prev": round(rsi_prev, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "bb_upper": round(sma20 + (std20 * 2), 2),
        "bb_lower": round(sma20 - (std20 * 2), 2),
        "atr": round(atr, 2),
        "macd": round(float(macd.iloc[-1]), 2),
        "macd_signal": round(float(signal.iloc[-1]), 2),
        "w_change": round(w_change, 2),
        "m_change": round(m_change, 2),
        "vol_ratio": round(vol_ratio, 2),
        "algo_signal": algo_signal
    }

def sanitize_md(text):
    if not text: return ""
    # Standardize bold (Legacy Markdown uses *bold* not **bold**)
    text = text.replace("**", "*")
    # Replace bullet point characters with safe ones
    text = re.sub(r"^\s*[*+-]\s+", "• ", text, flags=re.MULTILINE)
    # Replace other problematic formatting characters
    text = text.replace("_", "-") 
    text = text.replace("`", "'")
    return text

def get_intraday_signal(symbol: str) -> dict | None:
    try:
        ticker_sym = f"{symbol.upper()}.IS"
        data = yf.download(ticker_sym, period="5d", interval="15m", progress=False, auto_adjust=True)
        if data.empty or len(data) < 35:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data['Close']
        
        # MACD (12, 26, 9)
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)
        rsi_series = 100 - (100 / (1 + rs))
        
        # TRIX (15)
        ema1 = close.ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        trix_series = ema3.pct_change() * 10000

        # Current Values
        price = round(float(close.iloc[-1]), 2)
        macd_val = round(float(macd_line.iloc[-1]), 3)
        sig_val = round(float(signal_line.iloc[-1]), 3)
        macdp_val = round(float(macd_line.iloc[-2]), 3)
        sigp_val = round(float(signal_line.iloc[-2]), 3)
        rsi_val = round(float(rsi_series.iloc[-1]), 2)
        trix_val = round(float(trix_series.iloc[-1]), 3)
        
        # Sinyal Yorumlama
        # MACD: Mavi (MACD), Turuncu (Sinyal).
        decision = "BEKLE 🟡"
        macd_yorum = f"MACD: {macd_val} | Sinyal: {sig_val}"
        if macdp_val <= sigp_val and macd_val > sig_val:
            decision = "AL 🟢 (Mavi Çizgi, Turuncuyu Yukarı Kesti!)"
        elif macdp_val >= sigp_val and macd_val < sig_val:
            decision = "SAT 🔴 (Mavi Çizgi, Turuncuyu Aşağı Kesti!)"
        elif macd_val > sig_val:
            decision = "TUT 🟢 (Mavi Çizgi Üstte, Kısa Vade Trend Pozitif)"
        elif macd_val < sig_val:
            decision = "BEKLE 🔴 (Mavi Çizgi Altta, Kısa Vade Trend Negatif)"

        # TRIX ve RSI Uyarısı
        trix_yorum = "TRIX Güçlü Pozitif Trend" if trix_val > 0 else "TRIX Negatif Trend"
        if rsi_val > 70:
            rsi_yorum = f"RSI: {rsi_val} (AŞIRI ALIM! Kar satışı riski var)"
        elif rsi_val < 30:
            rsi_yorum = f"RSI: {rsi_val} (AŞIRI SATIM! Tepki alımı gelebilir)"
        else:
            rsi_yorum = f"RSI: {rsi_val} (Nötr Bölge)"

        # --- Gemini AI Yorumu Ekleme ---
        tools = [types.Tool(google_search=types.GoogleSearch())]
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            system_instruction="Sen bir 'Scalping' ve 'Day Trade' uzmanısın. Kısa vadeli (15 dakikalık) teknik verileri ve internetteki en son haberleri yorumlayarak anlık işlem kararı verirsin."
        )

        prompt = f"""
        Görev: '{symbol}' hissesi için 15 dakikalık teknik verileri ve internetteki EN GÜNCEL haberleri (son 1 saat içindeki gelişmeler, KAP haberleri) birleştirerek kısa vadeli bir analiz yap.
        
        Teknik Veriler:
        - Fiyat: {price} TL
        - Teknik Karar: {decision}
        - MACD Durumu: {macd_yorum}
        - RSI Durumu: {rsi_yorum}
        - TRIX Durumu: {trix_yorum} ({trix_val})
        
        Senden Beklenen:
        1. Teknik veriler ile internetteki son dakika haberleri arasında bir uyum var mı? (Örn: Teknik 'AL' diyor ama kötü bir haber mi geldi?)
        2. Bu verilere göre 1-2 saatlik periyotta hisse nasıl hareket edebilir?
        3. Çok kısa bir 'Scalper Notu' ekle.
        
        Kural: Çok kısa, net ve vurucu konuş. Karmaşık terimlerden kaçın. Madde işareti olarak sadece '•' kullan.
        'Yatırım tavsiyesi değildir.'
        """
        
        ai_response = client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp-01-21',
            contents=prompt,
            config=generate_content_config
        )
        
        return {
            "price": price,
            "decision": decision,
            "macd_yorum": macd_yorum,
            "rsi_yorum": rsi_yorum,
            "trix_yorum": trix_yorum,
            "trix_val": trix_val,
            "ai_yorum": sanitize_md(ai_response.text)
        }
    except Exception as e:
        logging.warning(f"Intraday Signal Hatası ({symbol}): {e}")
        return None

def get_stock_news(symbol: str) -> str:
    """Hisse ile ilgili en son haber başlıklarını (KAP dahil) Google News üzerinden çeker."""
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+hisse+KAP&hl=tr&gl=TR&ceid=TR:tr"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            from datetime import datetime, timedelta
            from email.utils import parsedate_to_datetime
            now = datetime.now()
            news = []
            for item in root.findall('.//item'):
                title = item.find('title')
                pubDate = item.find('pubDate')
                if title is not None and pubDate is not None:
                    try:
                        # "Thu, 15 Mar 2026 12:00:00 GMT"
                        date_str = pubDate.text[:-4] # GMT kısmını at
                        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S")
                        if now - dt < timedelta(days=7):
                            date_clean = dt.strftime("%d %b")
                            news.append(f"[{date_clean}] {title.text}")
                    except:
                        continue
                if len(news) >= 4: break
            if news:
                return "\n".join(news)
    except Exception as e:
        logging.warning(f"Haber çekilemedi ({symbol}): {e}")
    return "Piyasa haberi bulunamadı."

def get_stock_analysis(symbol):
    try:
        market_context = get_market_context()
        macro_context = get_macro_context()
        ticker_sym = f"{symbol.upper()}.IS"
        ticker = yf.Ticker(ticker_sym)
        
        info = {}
        try:
            ticker_info = ticker.info
            if isinstance(ticker_info, dict):
                info = ticker_info
        except Exception as e:
            logging.warning(f"Temel veriler alinamadi: {e}")

        # ─── F/K (P/E) ─────────────────────────────────────────────────────────
        fk_not = ""
        trailing_pe = info.get("trailingPE")
        forward_pe  = info.get("forwardPE")
        trailing_eps = info.get("trailingEps")
        info_price   = info.get("currentPrice") or info.get("previousClose")

        if forward_pe and isinstance(forward_pe, (int, float)) and 0 < forward_pe < 200:
            fk = round(forward_pe, 2)
            fk_not = " (İleriye Dönük)"
        elif trailing_pe and isinstance(trailing_pe, (int, float)) and 0 < trailing_pe < 150:
            fk = round(trailing_pe, 2)
            fk_not = " (Geçmiş)"
        elif trailing_eps and info_price and isinstance(trailing_eps, (int, float)) and trailing_eps > 0:
            fk = round(info_price / trailing_eps, 2)
            if fk > 150:
                fk = "Veri Yok (Bozuk EPS)"
                fk_not = ""
            else:
                fk_not = " (Hesaplama)"
        else:
            fk = "Veri Yok"
            fk_not = ""

        if isinstance(fk, (int, float)):
            fk = f"{fk}{fk_not}"

        pddd_raw = info.get("priceToBook")
        if pddd_raw is None or not isinstance(pddd_raw, (int, float)):
            pddd = "Veri Yok"
        else:
            pddd = round(pddd_raw, 2)
            if pddd > 30:
                pddd = f"{pddd} ⚠️(Kur Uyumsuzluğu Riski)"

        ebitda_margin = info.get("ebitdaMargins")
        if ebitda_margin is None or not isinstance(ebitda_margin, (int, float)):
            ebitda_margin = "Veri Yok"
        else:
            ebitda_margin = f"%{round(ebitda_margin * 100, 2)}"

        ev_ebitda_raw = info.get("enterpriseToEbitda")
        if ev_ebitda_raw is None or not isinstance(ev_ebitda_raw, (int, float)):
            ev_ebitda = "Veri Yok"
        elif ev_ebitda_raw > 80 or ev_ebitda_raw < 0:
            ev_ebitda = f"{round(ev_ebitda_raw, 2)} ⚠️(Güvenilmez)"
        else:
            ev_ebitda = round(ev_ebitda_raw, 2)

        eps_g  = info.get("trailingEps", 0)
        bv_g   = info.get("bookValue", 0)
        target_price = info.get("targetMeanPrice")
        adil_deger_str = "Hesaplanamadı"
        try:
            if target_price is not None and isinstance(target_price, (int, float)) and target_price > 0:
                adil_deger_str = f"{round(target_price, 2)} TL (Analist Hedefi)"
            elif eps_g and bv_g and isinstance(eps_g, (int, float)) and isinstance(bv_g, (int, float)) and eps_g > 1 and bv_g > 1:
                adil_raw = round((22.5 * eps_g * bv_g) ** 0.5, 2)
                if adil_raw > 0:
                    adil_deger_str = f"{adil_raw} TL (Graham Hesabı)"
        except:
            pass

        eg_val = info.get("earningsGrowth")
        eg_str = f"%{round(eg_val * 100, 2)}" if isinstance(eg_val, (int, float)) else "Veri Yok"
        roe_val = info.get("returnOnEquity")
        roe_str = f"%{round(roe_val * 100, 2)}" if isinstance(roe_val, (int, float)) else "Veri Yok"
        roa_val = info.get("returnOnAssets") 
        roa_str = f"%{round(roa_val * 100, 2)}" if isinstance(roa_val, (int, float)) else "Veri Yok"
        peg_val = info.get("pegRatio")
        peg_str = str(round(peg_val, 2)) if isinstance(peg_val, (int, float)) else "Veri Yok"
        eps_str = str(round(info.get("trailingEps"), 2)) if isinstance(info.get("trailingEps"), (int, float)) else "Veri Yok"
        bv_str = str(round(info.get("bookValue"), 2)) if isinstance(info.get("bookValue"), (int, float)) else "Veri Yok"

        # ─── ROIC (Sermaye Yatırım Getirisi) ──────────────────────────────────
        roic_val = info.get("returnOnInvestedCapital") or info.get("returnOnAssets")
        roic_str = f"%{round(roic_val * 100, 2)}" if isinstance(roic_val, (int, float)) else "Veri Yok"

        fundamental_context = (
            f"Çarpan Değeri (P/E & P/B): F/K: {fk} | PD/DD: {pddd}\n"
            f"Adil Değer (Fair Value): {adil_deger_str}\n"
            f"Sermaye Yatırım Getirisi (ROIC): {roic_str}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Hisse Başına Kar (EPS): {eps_str} | Öz Sermaye (Defter Değeri): {bv_str}\n"
            f"FAVÖK Marjı: {ebitda_margin} | FD/FAVÖK: {ev_ebitda}\n"
            f"Yıllık Kar Büyümesi: {eg_str} | Özkaynak Karlılığı (ROE): {roe_str} | PEG Oranı: {peg_str}"
        )

        data = yf.download(ticker_sym, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            logging.warning(f"⚠️ {ticker_sym} için fiyat verisi boş döndü.")
            return None
            
        # Robust MultiIndex Handling
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data.columns = data.columns.get_level_values(0)
            else:
                data.columns = data.columns.get_level_values(1)
        
        if 'Close' not in data.columns:
            logging.error(f"❌ {ticker_sym} verisinde 'Close' sütunu bulunamadı. Mevcut: {data.columns}")
            return None

        last_price = float(data['Close'].iloc[-1])
        ind = calculate_indicators(data)
        news_text = get_stock_news(symbol)

        # Google Search Tool Tanımlama
        tools = [
            types.Tool(google_search=types.GoogleSearch())
        ]
        
        # Yeni GenAI Client Kullanımı
        current_date = time.strftime("%d.%m.%Y")
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
            system_instruction=f"Bugün tarih: {current_date}. Sen deneyimli bir Borsa Analistisin. Sadece GÜNCEL (son 24 saat) haberlere odaklanmalısın. Geçmişte kalmış (aylar önceki) bedelsiz, sermaye artırımı gibi olayları sanki bugün olmuş gibi anlatma. Eğer yeni bir haber yoksa 'yeni bir gelişme yok' de."
        )

        prompt = f"""
        Görev: '{symbol}' hissesi için internetten en güncel gelişmeleri (son dakika haberleri, yeni KAP bildirimleri, sektörel yorumlar) de araştırarak BÜTÜNCÜL (Holistik) bir rapor yaz.
        
        ÖNEMLİ (BAŞLANGIÇ): Raporuna mutlaka en başta Çarpan Değeri, Adil Değer ve Sermaye Yatırım Getirisi (ROIC) verilerini net bir şekilde belirterek başla. 
        Mevcut fiyatın Adil Değer'e göre ucuz mu pahalı mı olduğunu ve ROIC'in şirketin verimliliği hakkında ne dediğini (örn: %30 üstü harika, %15 altı zayıf vb.) basitçe yorumla.

        KESİNLİKLE DİKKAT ETMEN GEREKENLER (KURALLAR):
        1. ASLA karmaşık finansal kelimeler (jargon) kullanma. Normal, borsaya yeni başlamış bir insanın anlayacağı kadar BASİT ve SADE anlat.
        2. Teknik terimleri (RSI, MACD vb.) doğrudan yazmak yerine anlamlarını yorumlayarak anlat.
        3. İnsanları teknik verilere boğma. Sadece verilen sayıların iyi mi kötü olduğunu söyle.
        4. İnternetten bu hisse ile ilgili SON 24 SAATTEKİ gelişmeleri kontrol et.
        
        Makro Durum: {macro_context}
        Endeks Durumu: {market_context}
        Piyasa Haberleri ve KAP Bildirimleri (Sana gelenler):
        {news_text}
        Şirketin Temel Analizi:
        {fundamental_context}
        Hisse Teknik Verileri:
        - Fiyat: {last_price} TL (Haftalık: %{ind['w_change']}, Aylık: %{ind['m_change']})
        - Teknikler: RSI:{ind['rsi']}, MACD:{ind['macd']}/{ind['macd_signal']}
        - Ortalamalar: SMA20:{ind['sma20']}, SMA50:{ind['sma50']}, SMA200:{ind['sma200']}
        - Algoritmik Sinyal: {ind['algo_signal']}

        'Yatırım tavsiyesi değildir.' şeklinde bitir.
        
        ÖNEMLİ: Bugünün tarihi {current_date}. Eğer internette bulduğun haberler 1 haftadan eskiyse onları 'güncel haber' gibi sunma.
        
        Analiz Formatın:
        🏢 *ŞİRKETİN DURUMU (BİLANÇO)*: (Sade dille)
        🤖 *ALGORİTMA NE DİYOR?*: {ind['algo_signal']}
        🚀 *GRAFİK VE TREND*: (Yorumla)
        📈 *İŞLEM SEVİYELERİ*: (Alış/Hedef/Stop)
        ⚖️ *RİSK VE PORTFÖY*: (Derecelendir)
        🧠 *SON SÖZ*: (Karar)
        """
        response = client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp-01-21', # Not: gemini-3-flash-preview henüz yaygın olmayabilir, en güçlü thinking modelini kullanıyoruz.
            contents=prompt,
            config=generate_content_config
        )
        
        safe_response = sanitize_md(response.text)
        return {"price": round(last_price, 2), "ind": ind, "analysis": safe_response, "market": market_context}
    except Exception as e:
        logging.error(f"❌ Analiz Hatası ({symbol}): {e}")
        logging.error(traceback.format_exc())
        return None

def get_daily_prediction(symbol):
    try:
        market_context = get_market_context()
        ticker_sym = f"{symbol.upper()}.IS"
        data = yf.download(ticker_sym, period="1mo", interval="1d", progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            logging.warning(f"⚠️ {ticker_sym} için yetersiz veri.")
            return None
            
        # Robust MultiIndex Handling
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data.columns = data.columns.get_level_values(0)
            else:
                data.columns = data.columns.get_level_values(1)

        if 'Close' not in data.columns:
            return None
            
        last_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        prev_high  = float(data['High'].iloc[-2])
        prev_low   = float(data['Low'].iloc[-2])
        
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = (2 * pivot) - prev_low
        r1 = (2 * pivot) - prev_high
        s1 = (2 * pivot) - prev_low
        
        # Günlük Tahmin için de Temel Verileri Çekelim
        try:
            ticker_info = yf.Ticker(ticker_sym).info
            info = ticker_info if isinstance(ticker_info, dict) else {}
            fk = info.get("forwardPE") or info.get("trailingPE") or "Veri Yok"
            pddd = info.get("priceToBook") or "Veri Yok"
            roic = info.get("returnOnInvestedCapital") or info.get("returnOnAssets") or 0.0
            target = info.get("targetMeanPrice") or "Hesaplanamadı"
            
            fundamental_summary = (
                f"Çarpan Değeri: {fk} (F/K), {pddd} (PD/DD) | "
                f"Adil Değer: {target} TL | "
                f"Sermaye Yatırım Getirisi (ROIC): %{round(roic*100,2) if roic else 'Veri Yok'}"
            )
        except:
            fundamental_summary = "Temel veri alınamadı."

        ind = calculate_indicators(data)
        news_text = get_stock_news(symbol)
        
        # Google Search Tool Tanımlama
        tools = [
            types.Tool(google_search=types.GoogleSearch())
        ]
        
        current_date = time.strftime("%d.%m.%Y")
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
            system_instruction=f"Bugün tarih: {current_date}. Sen bir Day Trader'sın. SADECE BUGÜNÜN haberlerine ve anlık fiyat hareketlerine odaklan. Eski bedelsiz/sermaye artışı haberlerini ciddiye alma, onlar fiyatlandı bitti."
        )

        prompt = f"""
        Görev: '{symbol}' hissesi için internetten en güncel piyasa havasını ve haberleri araştırarak GÜNLÜK bir TAHMİN raporu yaz.
        
        ÖNEMLİ: Tahminine başlarken Çarpan Değeri, Adil Değer ve ROIC verilerini göz önünde bulundurarak hissenin bugün için 'pahalı' mı 'ucuz' mu olduğunu kısaca hissettir.
        
        - Fiyat: {last_price} TL
        - Pivot: {round(pivot,2)} TL (Destek: {round(s1,2)} / Direnç: {round(r1,2)})
        - RSI: {ind['rsi']}
        - Temel Durum: {fundamental_summary}
        - Haberler (Sana gelenler): {news_text}
        
        Format:
        🎯 *GÜNLÜK HİSSİYAT*
        🌍 *GÜNCEL HABER ETKİSİ*: (İnternetten bulduğun bugünlük önemli haberlerin etkisini yorumla)
        ⚖️ *İŞLEM SEVİYELERİ*
        🔮 *GÜNÜN TAHMİNİ*
        
        ÖNEMLİ: Madde işaretleri için sadece '•' karakterini kullan. Kalın yazmak için '*'.
        'Yatırım tavsiyesi değildir.' şeklinde bitir.
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp-01-21',
            contents=prompt,
            config=generate_content_config
        )
        
        return {"price": round(last_price, 2), "analysis": sanitize_md(response.text)}
    except Exception as e:
        logging.error(f"❌ Tahmin Hatası ({symbol}): {e}")
        logging.error(traceback.format_exc())
        return None

# --- BOT KOMUTLARI ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔍 Hisse Analiz Et", callback_data='analiz_et'),
         InlineKeyboardButton("📅 Günlük Tahmin", callback_data='tahmin_et')],
        [InlineKeyboardButton("📋 Takip Listem", callback_data='liste_goster'), 
         InlineKeyboardButton("🗓 Haftalık Rapor", callback_data='haftalik_rapor')],
        [InlineKeyboardButton("🔭 Fırsat Tarayıcı", callback_data='hisse_tara'),
         InlineKeyboardButton("🎯 Özel Tarayıcı", callback_data='hisse_ozel_tara')],
        [InlineKeyboardButton("📚 Borsa Sözlüğü", callback_data='bilgi_al'),
         InlineKeyboardButton("⚡ Anlık Sinyal", callback_data='anlik_sinyal')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (
        "🏛 *BÜTÜNCÜL BORSA ASİSTANI*\n\n"
        "Hoş geldiniz. Lütfen yapmak istediğiniz işlemi seçin:"
    )
    if update.message:
        await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')
    else:
        await update.callback_query.message.edit_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')

async def analiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    if not symbol:
        await update.message.reply_text("🚨 Lütfen sembol girin. Örn: `/analiz thyao`", parse_mode='Markdown')
        return
    msg = await update.message.reply_text(f"⏳ *{symbol}* için Analiz yapılıyor...", parse_mode='Markdown')
    res = await asyncio.to_thread(get_stock_analysis, symbol)
    if res:
        keyboard = [[InlineKeyboardButton(f"📌 {symbol} Listeye Ekle", callback_data=f'ekle_{symbol}')],
                    [InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
        text = f"🏆 *RAPOR: {symbol}*\n💰 Fiyat: {res['price']} TL\n━━━━━━━━━━━━━━━━━━\n{res['analysis']}"
        try:
            await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        except BadRequest as e:
            if "Can't parse entities" in str(e):
                await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                raise e
    else:
        await msg.edit_text("❌ Veri bulunamadı.")

async def tahmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    if not symbol:
        await update.message.reply_text("🚨 Lütfen sembol girin.", parse_mode='Markdown')
        return
    msg = await update.message.reply_text(f"⏳ *{symbol}* için Tahmin yapılıyor...", parse_mode='Markdown')
    res = await asyncio.to_thread(get_daily_prediction, symbol)
    if res:
        keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
        text = f"📈 *TAHMİN: {symbol}*\n💰 Fiyat: {res['price']} TL\n━━━━━━━━━━━━━━━━━━\n{res['analysis']}"
        try:
            await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        except BadRequest as e:
            if "Can't parse entities" in str(e):
                await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                raise e
    else:
        await msg.edit_text("❌ Tahmin bulunamadı.")

async def tara(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔭 *TARAYICI BAŞLATILDI*...", parse_mode='Markdown')
    scan_list = get_all_bist_tickers()
    results = await asyncio.to_thread(run_market_scan, scan_list)
    if not results:
        await msg.edit_text("📭 Fırsat bulunamadı.")
        return
    lines = ["🎯 *AL FIRSATI RADAR:*\n"]
    for i, r in enumerate(results[:8], 1):
        grade = "🟢" if r['score'] >= 65 else "🟡"
        lines.append(f"{grade} {i}. {r['symbol']} ({r['score']}/100) - {r['price']} TL")
    keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
    try:
        await msg.edit_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    except BadRequest as e:
        if "Can't parse entities" in str(e):
            await msg.edit_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            raise e

async def ozel_tara(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🎯 *ÖZEL TARAYICI BAŞLATILDI*...\n(Metrikler: RSI 20-40, Özsermaye x2-x3, FAVÖK<=10, PD/DD<=1.5)", parse_mode='Markdown')
    scan_list = get_all_bist_tickers()
    results = await asyncio.to_thread(run_custom_scan, scan_list)
    if not results:
        await msg.edit_text("📭 Bu özel kriterlere uyan hisse bulunamadı.")
        return
    lines = ["🔥 *ÖZEL KRİTERLERE UYANLAR:*\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"✅ {i}. {r['symbol']} ({r['price']} TL) | RSI:{r['rsi']}, BV:{r['bv']}, PD/DD:{r['pddd']}, EV/EBITDA:{r['ev_ebitda']}")
    keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
    try:
        await msg.edit_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    except BadRequest as e:
        if "Can't parse entities" in str(e):
            await msg.edit_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            raise e

async def sinyal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    if not symbol:
        await update.message.reply_text("🚨 Lütfen sembol girin. Örn: `/sinyal thyao`", parse_mode='Markdown')
        return
    msg = await update.message.reply_text(f"⏳ *{symbol}* için 15 dakikalık Anlık Sinyaller analiz ediliyor...", parse_mode='Markdown')
    res = await asyncio.to_thread(get_intraday_signal, symbol)
    if res:
        keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
        text = (
            f"⚡ *ANLIK SİNYAL: {symbol}*\n"
            f"💰 Fiyat: {res['price']} TL\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *KARAR:* {res['decision']}\n\n"
            f"📊 *Detaylar (15 dk grafik):*\n"
            f"• {res['macd_yorum']}\n"
            f"• {res['rsi_yorum']}\n"
            f"• {res['trix_yorum']} ({res['trix_val']})\n\n"
            f"🧠 *YAPAY ZEKA YORUMU:*\n{res['ai_yorum']}"
        )
        try:
            await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        except BadRequest as e:
            if "Can't parse entities" in str(e):
                await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                raise e
    else:
        await msg.edit_text("❌ Yeterli anlık veri bulunamadı. (Mesai saatleri dışında veya sorunlu hisse)")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == 'analiz_et':
        await query.message.reply_text("🔍 Hisse kodunu yazın (Örn: THYAO).")
    elif data == 'tahmin_et':
        await query.message.reply_text("📅 `/tahmin HİSSE` yazın.")
    elif data == 'liste_goster':
        watchlist = load_watchlist()
        text = "📋 *LİSTE:*\n" + "\n".join([f"- {s}" for s in watchlist]) if watchlist else "Liste boş."
        await query.message.reply_text(text, parse_mode='Markdown')
    elif data == 'haftalik_rapor':
        watchlist = load_watchlist()
        if not watchlist:
            await query.message.reply_text("Liste boş!")
            return
        await query.message.reply_text("📉 *Haftalık Tarama...*")
        for s in watchlist:
            res = await asyncio.to_thread(get_stock_analysis, s)
            if res:
                text = f"📍 *{s}*\n{res['analysis']}"
                try:
                    await query.message.reply_text(text, parse_mode='Markdown')
                except BadRequest as e:
                    if "Can't parse entities" in str(e):
                        await query.message.reply_text(text)
                    else:
                        raise e
    elif data == 'bilgi_al':
        await query.message.reply_text("📚 *Borsa Sözlüğü aktif.*", parse_mode='Markdown')
    elif data == 'anlik_sinyal':
        await query.message.reply_text("⚡ Lütfen `/sinyal HISSE` şeklinde komut gönderin (Örn: `/sinyal THYAO`).", parse_mode='Markdown')
    elif data.startswith('ekle_'):
        symbol = data.split('_')[1]
        watchlist = load_watchlist()
        if symbol not in watchlist:
            watchlist.append(symbol)
            save_watchlist(watchlist)
            await query.message.reply_text(f"✅ {symbol} eklendi.")
    elif data == 'hisse_tara':
        await tara(update, context)
    elif data == 'hisse_ozel_tara':
        await ozel_tara(update, context)
    elif data == 'ana_menu':
        await start(update, context)

async def handle_regular_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        text = update.message.text.strip().upper()
        if len(text) <= 6 and text.isalpha():
            context.args = [text]
            await analiz(update, context)

# --- RENDER KEEP ALIVE (FLASK) ---
flask_app = Flask('')

@flask_app.route('/')
def home():
    return "Bot aktif ve çalışıyor!"

def run_flask():
    port = int(os.getenv("PORT", 8080))
    flask_app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_flask)
    t.start()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --- ANA ÇALIŞTIRICI ----------------------------------------
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    keep_alive()  # Web sunucusunu başlatır
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analiz", analiz))
    app.add_handler(CommandHandler("tahmin", tahmin))
    app.add_handler(CommandHandler("tara", tara))
    app.add_handler(CommandHandler("ozeltara", ozel_tara))
    app.add_handler(CommandHandler("sinyal", sinyal))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_regular_text))
    app.add_handler(CallbackQueryHandler(button_handler))
    print("🚀 Bot Aktif!")
    app.run_polling()