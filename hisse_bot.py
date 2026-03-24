import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import logging
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

# --- KONFİGÜRASYON ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WATCHLIST_FILE = "watchlist.json"

# Gemini Kurulumu
genai.configure(api_key=GEMINI_API_KEY)

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
            info = yf.Ticker(ticker_sym).info
            
            # 1. Öz sermaye negatif olmayacak
            bv = info.get("bookValue")
            if bv is not None and isinstance(bv, (int, float)) and bv <= 0:
                return None
                
            # 2. F/K < 10
            fk = info.get("forwardPE") or info.get("trailingPE")
            if fk is not None and isinstance(fk, (int, float)) and fk >= 10:
                return None
                
            # 3. PD/DD <= 10
            pddd = info.get("priceToBook")
            if pddd is not None and isinstance(pddd, (int, float)) and pddd > 10:
                return None
                
            # 4. Yıllık kar büyümesi >= 100% (1.0)
            eg = info.get("earningsGrowth")
            if eg is not None and isinstance(eg, (int, float)) and eg < 1.0:
                return None
                
            # 5. Hisse Başına Kar (EPS) >= 0
            eps = info.get("trailingEps")
            if eps is not None and isinstance(eps, (int, float)) and eps < 0:
                return None
                
            # 6. PEG <= 1
            peg = info.get("pegRatio")
            if peg is not None and isinstance(peg, (int, float)) and peg > 1:
                return None
                
            # 7. Özkaynak Karlılığı (ROE) >= %10
            roe = info.get("returnOnEquity")
            if roe is not None and isinstance(roe, (int, float)) and roe < 0.10:
                return None
                
            # 8. ROIC (Proxy: ROA) >= %30
            roa = info.get("returnOnAssets")
            if roa is not None and isinstance(roa, (int, float)) and roa < 0.30:
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
        # YFinance toplu indirme (grup by ticker sayesinde her hisse kendi OHLCV dataframe'ine sahip olur)
        df = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True, threads=True)
        
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
    text = text.replace("**", "*")
    text = text.replace("_", "-") 
    text = text.replace("`", "'")
    return text

def get_stock_news(symbol: str) -> str:
    """Hisse ile ilgili en son haber başlıklarını (KAP dahil) Google News üzerinden çeker."""
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+hisse+KAP&hl=tr&gl=TR&ceid=TR:tr"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            news = []
            for item in root.findall('.//item')[:4]:
                title = item.find('title')
                pubDate = item.find('pubDate')
                if title is not None and pubDate is not None:
                    # Tarih formatını temizle (Örn: "Thu, 15 Mar 2026 12:00:00 GMT" -> "15 Mar")
                    date_clean = pubDate.text[:-13].replace(',','')
                    news.append(f"[{date_clean}] {title.text}")
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
            info = ticker.info
        except Exception as e:
            logging.warning(f"Temel veriler alinamadi: {e}")

        # ─── F/K (P/E) ─────────────────────────────────────────────────────────
        # Yahoo Finance BIST için trailingEps'i kur/bölünme uyumsuzluğuyla kaydeder.
        # Bu durum trailingPE'yi 300-600+ gibi şişirir.
        # En güvenilir kaynak: forwardPE (analist konsensüsü, TL cinsinden tutarlı).
        # Fallback sırası: forwardPE → trailingPE (< 150 ise) → Manuel (fiyat/eps)
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

        # ─── PD/DD (Price/Book) ────────────────────────────────────────────────
        # bookValue'nun USD yerine TRY olup olmadığı bilinemiyor; büyük sapmalar bayraklanır.
        pddd_raw = info.get("priceToBook")
        if pddd_raw is None or not isinstance(pddd_raw, (int, float)):
            pddd = "Veri Yok"
        else:
            pddd = round(pddd_raw, 2)
            if pddd > 30:
                pddd = f"{pddd} ⚠️(Kur Uyumsuzluğu Riski)"

        # ─── FAVÖK Marjı ───────────────────────────────────────────────────────
        ebitda_margin = info.get("ebitdaMargins")
        if ebitda_margin is None or not isinstance(ebitda_margin, (int, float)):
            ebitda_margin = "Veri Yok"
        else:
            ebitda_margin = f"%{round(ebitda_margin * 100, 2)}"

        # ─── FD/FAVÖK ──────────────────────────────────────────────────────────
        # BIST için yfinance enterpriseToEbitda çoğunlukla bozuk gelir (100+).
        # 80 üstü değerler güvenilmez, bayraklanır.
        ev_ebitda_raw = info.get("enterpriseToEbitda")
        if ev_ebitda_raw is None or not isinstance(ev_ebitda_raw, (int, float)):
            ev_ebitda = "Veri Yok"
        elif ev_ebitda_raw > 80 or ev_ebitda_raw < 0:
            ev_ebitda = f"{round(ev_ebitda_raw, 2)} ⚠️(Güvenilmez)"
        else:
            ev_ebitda = round(ev_ebitda_raw, 2)

        # ─── Graham Adil Değer ─────────────────────────────────────────────────
        # Graham formülü: √(22.5 × EPS × BV). Doğrudan hisse fiyatıyla kıyaslanır.
        # trailingEps bozuk olabileceğinden sonucu kontrol ediyoruz.
        eps_g  = info.get("trailingEps", 0)
        bv_g   = info.get("bookValue", 0)
        target_price = info.get("targetMeanPrice")
        adil_deger_str = "Hesaplanamadı"
        try:
            if target_price is not None and isinstance(target_price, (int, float)) and target_price > 0:
                # Analist hedef fiyatı en güvenilir kaynak
                adil_deger_str = f"{round(target_price, 2)} TL (Analist Hedefi)"
            elif eps_g and bv_g and isinstance(eps_g, (int, float)) and isinstance(bv_g, (int, float)) and eps_g > 1 and bv_g > 1:
                # Graham formülü: √(22.5 × EPS × Defter Değeri)
                # BIST'te trailingEps genellikle bozuk olduğundan (0.93 gibi küçük),
                # sonucu mevcut fiyatla karşılaştırarak makul olup olmadığını kontrol ediyoruz.
                adil_raw = round((22.5 * eps_g * bv_g) ** 0.5, 2)
                # Eğer Graham değeri mevcut fiyatın 1/5'inden küçük çık ise EPS bozuktur → gösterme
                if adil_raw > 0 and (last_price / adil_raw) < 5:
                    adil_deger_str = f"{adil_raw} TL (Graham Hesabı)"
                else:
                    adil_deger_str = "Hesaplanamadı (EPS verisi BIST'te güvenilmez)"
        except:
            pass

        # Yeni eklenen gelişmiş temel metrikler
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

        fundamental_context = (
            f"F/K: {fk} | PD/DD: {pddd} | Hisse Başına Kar (EPS): {eps_str} | Öz Sermaye (Defter Değeri): {bv_str}\n"
            f"FAVÖK Marjı: {ebitda_margin} | FD/FAVÖK: {ev_ebitda} | Adil Değer: {adil_deger_str}\n"
            f"Yıllık Kar Büyümesi: {eg_str} | Özkaynak Karlılığı (ROE): {roe_str} | ROIC (Zımni ROA): {roa_str}\n"
            f"PEG Oranı: {peg_str}"
        )

        data = yf.download(ticker_sym, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        last_price = float(data['Close'].iloc[-1])
        ind = calculate_indicators(data)
        
        news_text = get_stock_news(symbol)

        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Görev: Sen deneyimli ama halk dilinden konuşan, samimi bir Borsa Analistisin. 
        Aşağıdaki verileri kullanarak '{symbol}' hissesi için BÜTÜNCÜL (Holistik) bir rapor yaz.
        
        KESİNLİKLE DİKKAT ETMEN GEREKENLER (KURALLAR):
        1. ASLA karmaşık finansal kelimeler (jargon) kullanma. Normal, borsaya yeni başlamış bir insanın anlayacağı kadar BASİT ve SADE anlat.
        2. "Volatilite", "Momentum", "RSI", "MACD", "SMA", "F/K", "PD/DD" gibi terimleri doğrudan rapora yazmak yerine, bunların ne anlama geldiğini yorumlayarak anlat.
        3. İnsanları teknik verilere boğma. Sadece verilen sayıların iyi mi kötü mü olduğunu söyle.
        4. Sana verilen GERÇEK sayıları referans al.
        5. TEMEL BEKLENTİLER: Bir hissede ideal olarak (F/K < 10), (PD/DD <= 10), (Yıllık Kar Büyümesi >= %100), (ROE >= %10), (ROIC >= %30), (PEG <= 1), Öz sermayenin negatif olmaması ve EPS'in pozitif olması istenir. Ayrıca Fintables Karne Puanı beklentisi ise 12 ve üzeri olması istenir. Verilen bilanço sayılarını bu ideallere kıyaslayarak değerlendir (beklentiyi/ideali karşılayanları methet, karşılamayanları eleştir).
        
        Makro Durum: {macro_context}
        Endeks Durumu: {market_context}
        
        Piyasa Haberleri ve KAP Bildirimleri (Psikolojik & Temel Metin):
        {news_text}
        
        Şirketin Temel Analizi (Bilançosu):
        {fundamental_context}
        
        Hisse Teknik Verileri (BUNLARI HALK DİLİNE ÇEVİR):
        - Fiyat: {last_price} TL (Haftalık: %{ind['w_change']}, Aylık: %{ind['m_change']})
        - Teknikler: Mevcut RSI:{ind['rsi']} (Önceki:{ind['rsi_prev']}), MACD:{ind['macd']}/{ind['macd_signal']}, Hacim Oranı (20 günlüğe göre):{ind['vol_ratio']}x
        - Hareketli Ortalamalar: SMA20:{ind['sma20']}, SMA50:{ind['sma50']}, SMA200:{ind['sma200']} (SMA200 hissenin ana kalesidir)
        - ATR (Günlük Oynaklık): {ind['atr']} TL
        - Algoritmik Sinyal: {ind['algo_signal']}
        - Haberler: {news_text}

        Analiz Formatın (Kalın yazı için SADECE tek '*' kullan, '_' KULLANMA!):
        🏢 *ŞİRKETİN DURUMU (BİLANÇO)*: (Pahalı mı ucuz mu? İşleri nasıl gidiyor? En fazla 2 cümle, sade dille.)
        🤖 *ALGORİTMA NE DİYOR?*: {ind['algo_signal']}
        🚀 *GRAFİK VE TREND*: (Hisse yükselişte mi düşüşte mi? Alıcılar mı güçlü satıcılar mı? Temel göstergeleri halk diliyle yorumla)
        📈 *İŞLEM SEVİYELERİ*:
           - *Alış Bölgesi*: (Hangi fiyatlardan kademeli alınır?)
           - *Hedef (Kâr Al)*: (Kısa-orta vade beklenti)
           - *Stop (Zarar Kes)*: (Hangi fiyatın altına düşerse tehlikeli?)
           *Not: Seviyeleri destek/direnç (SMA) ve fiyata göre mantıklı belirle.*
        ⚖️ *RİSK VE PORTFÖY*: (Bu hisse ne kadar riskli? Portföyün yüzde kaçıyla alınmalı?)
        🧠 *SON SÖZ*: (Açık ve net nihai kararın)

        'Yatırım tavsiyesi değildir.' şeklinde bitir.
        """
        response = model.generate_content(prompt)
        safe_response = sanitize_md(response.text)
        return {"price": round(last_price, 2), "ind": ind, "analysis": safe_response, "market": market_context}
    except Exception as e:
        logging.error(f"Hata ({symbol}): {e}")
        return None

def get_daily_prediction(symbol):
    try:
        market_context = get_market_context()
        ticker_sym = f"{symbol.upper()}.IS"
        data = yf.download(ticker_sym, period="1mo", interval="1d", progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 2: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        last_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else last_price
        prev_high  = float(data['High'].iloc[-2]) if len(data) > 1 else float(data['High'].iloc[-1])
        prev_low   = float(data['Low'].iloc[-2]) if len(data) > 1 else float(data['Low'].iloc[-1])
        
        # Günlük Pivot, Destek ve Direnç hesaplama
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = (2 * pivot) - prev_low
        s1 = (2 * pivot) - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        
        ind = calculate_indicators(data)
        news_text = get_stock_news(symbol)
        
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Görev: Sen bir 'Day Trader' (Günlük Al-Satçı) uzmanısın. Bütün enerjin SADECE bugüne/yarına odaklı.
        '{symbol}' hissesi için tamamen izole olarak, mevcut analiz mantığından bağımsız KISA VADELİ (GÜNLÜK) bir TAHMİN raporu yaz.
        
        KESİNLİKLE KURALLAR:
        1. Hiçbir finansal jargon kullanma, piyasa diliyle ve halkın anlayacağı basit bir dille konuş. Tarafını belli et ('Yükselebilir', 'Düşüş tehlikesi var', 'Dalgalı/Yatay' vs).
        2. Uzun vadeli bilanço veya temel analizlerden (F/K, Milyonluk karlar vs) ASLA bahsetme, senin işin SADECE bugünkü fiyat ve piyasa psikolojisi.
        3. Verdiğim haberleri (KAP verilerini) okuyarak yatırımcı psikolojisini ölç.
        4. Verdiğim hesaplanmış Destek/Direnç ve Pivot noktalarına doğrudan raporunda yer ver.
        
        Endeks Durumu: {market_context}
        
        Güncel Haber ve KAP (Piyasa Duygusu/Psikolojisi):
        {news_text}
        
        Günlük Fiyat ve Göstergeler:
        - Güncel Fiyat: {last_price} TL
        - Dün Kapanış: {prev_close} TL
        - RSI: {ind['rsi']} (Önceki gün: {ind['rsi_prev']})
        - Algoritmik Günlük Durum: {ind['algo_signal']}
        
        Matematiksel Dönüş Noktaları (Klasik Pivot Formülü):
        - Güç Noktası (Pivot): {round(pivot,2)} TL
        - Destekler (S1, S2): {round(s1,2)} TL, {round(s2,2)} TL
        - Dirençler (R1, R2): {round(r1,2)} TL, {round(r2,2)} TL
        
        Rapor Formatın:
        🎯 *GÜNLÜK PİYASA HİSSİYATI*: (Haberlere ve momentum durumuna göre yatırımcı psikolojisi kısa vadede nasıl?)
        ⚖️ *ÖNEMLİ İŞLEM SEVİYELERİ*:
          - 🚧 *Pivot (Denge Çizgisi)*: {round(pivot,2)} TL (Altında veya üstünde kalmasına göre kısa vadeli yönünü yorumla)
          - 🧱 *Destek*: {round(s1,2)} / {round(s2,2)}
          - 🏔 *Direnç*: {round(r1,2)} / {round(r2,2)}
        🔮 *GÜNÜN TAHMİNİ (KISA VADE)*: (Sadece bugün/1 hafta için yön tahmini yap ve net bir şekilde fikrini belirt.)
        """
        response = model.generate_content(prompt)
        return {"price": round(last_price, 2), "analysis": sanitize_md(response.text)}
    except Exception as e:
        logging.error(f"Tahmin hatası ({symbol}): {e}")
        return None

# --- BOT KOMUTLARI ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔍 Hisse Analiz Et", callback_data='analiz_et'),
         InlineKeyboardButton("📅 Günlük Tahmin", callback_data='tahmin_et')],
        [InlineKeyboardButton("📋 Takip Listem", callback_data='liste_goster'), 
         InlineKeyboardButton("🗓 Haftalık Rapor", callback_data='haftalik_rapor')],
        [InlineKeyboardButton("🔭 Fırsat Tarayıcı", callback_data='hisse_tara')],
        [InlineKeyboardButton("📚 Borsa Sözlüğü", callback_data='bilgi_al')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (
        "🏛 *BÜTÜNCÜL BORSA ASİSTANI*\n\n"
        "Hoş geldiniz. Bu bot, yapay zeka ve profesyonel algoritmaları kullanarak bir hisseyi "
        "hem Temel (Bilanço), hem Teknik (Grafik) hem de Makro (Ekonomi) açılardan inceler.\n\n"
        "Lütfen yapmak istediğiniz işlemi seçin:"
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
    msg = await update.message.reply_text(f"⏳ *{symbol}* için Holistik (Temel+Teknik) Analiz yapılıyor...", parse_mode='Markdown')
    res = await asyncio.to_thread(get_stock_analysis, symbol)
    if res:
        keyboard = [[InlineKeyboardButton(f"📌 {symbol} Listeye Ekle", callback_data=f'ekle_{symbol}')],
                    [InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
        text = (f"🏆 *BÜTÜNCÜL PORTFÖY RAPORU: {symbol}*\n🌍 {res['market']}\n💰 Fiyat: {res['price']} TL\n"
                f"━━━━━━━━━━━━━━━━━━\n{res['analysis']}")
        try:
            await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        except Exception as e:
            logging.error(f"Mesaj edit hatası: {e}")
            await msg.edit_text(text)
    else:
        await msg.edit_text("❌ Veri bulunamadı. Lütfen sembolü (örn: THYAO) kontrol edin.")

async def tahmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    if not symbol:
        await update.message.reply_text("🚨 Lütfen sembol girin. Örn: `/tahmin thyao`", parse_mode='Markdown')
        return
    msg = await update.message.reply_text(f"⏳ *{symbol}* için Günlük Pivot/Yön Tahmini yapılıyor...", parse_mode='Markdown')
    res = await asyncio.to_thread(get_daily_prediction, symbol)
    if res:
        keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
        text = f"📈 *GÜNLÜK AL-SAT TAHMİN RAPORU: {symbol}*\n💰 Anlık Fiyat: {res['price']} TL\n━━━━━━━━━━━━━━━━━━\n{res['analysis']}"
        try:
            await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        except Exception as e:
            logging.error(f"Mesaj edit hatası: {e}")
            await msg.edit_text(text)
    else:
        await msg.edit_text("❌ Veri/Tahmin bulunamadı.")

async def tara(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tüm BIST listesini tarayarak fırsat hisselerini listeler."""
    scan_list = get_all_bist_tickers()
    source_label = f"BIST Piyasası ({len(scan_list)} hisse)"
    msg = await update.message.reply_text(
        f"🔭 *FIRSAT TARAYICI*\n"
        f"_{source_label} taranıyor, lütfen bekleyin (60-120 sn)..._",
        parse_mode='Markdown'
    )
    results = await asyncio.to_thread(run_market_scan, scan_list)
    if not results:
        await msg.edit_text("📭 Şu an kriterlerimizi karşılayan güçlü fırsat bulunamadı.")
        return
    lines = ["🎯 *AL FIRSATI RADAR:*\n"]
    for i, r in enumerate(results[:8], 1):
        grade = "🟢 GÜÇLÜ" if r['score'] >= 65 else ("🟡 İYİ" if r['score'] >= 50 else "⚪ İZLE")
        tag_str = " · ".join(r['tags'][:3])
        lines.append(
            f"*{i}. {r['symbol']}* {grade} ({r['score']}/100)\n"
            f"   💰 {r['price']} TL | RSI:{r['rsi']} | Hacim:{r['vol']}x\n"
            f"   📌 {tag_str}\n"
        )
    lines.append("\n_Detaylı analiz için hisse adını yazın._")
    keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
    try:
        await msg.edit_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except Exception as e:
        logging.error(f"Tarayıcı mesaj hatası: {e}")
        await msg.edit_text("\n".join(lines))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == 'analiz_et':
        await query.message.reply_text("🔍 Analiz için sohbete direkt hisse kodunu yazabilirsiniz (Örn: THYAO).", parse_mode='Markdown')
    elif data == 'tahmin_et':
        await query.message.reply_text("📅 Günlük tahmin için sohbete `/tahmin HİSSE` kodunu yazabilirsiniz (Örn: `/tahmin THYAO`).", parse_mode='Markdown')
    elif data == 'liste_goster':
        watchlist = load_watchlist()
        text = "📋 *TAKİP LİSTESİ:*\n" + "\n".join([f"- {s}" for s in watchlist]) if watchlist else "📭 Liste boş."
        await query.message.reply_text(text, parse_mode='Markdown')
    elif data == 'haftalik_rapor':
        watchlist = load_watchlist()
        if not watchlist:
            await query.message.reply_text("Önce /kaydet ile hisse ekleyin!", parse_mode='Markdown')
            return
        await query.message.reply_text("📉 *Haftalık Bütüncül Taraması Başlatıldı...*", parse_mode='Markdown')
        for s in watchlist:
            res = await asyncio.to_thread(get_stock_analysis, s)
            if res:
                try:
                    await query.message.reply_text(f"📍 *{s} GÜNCELLEME*\n{res['analysis']}\n━━━━━━━━━━━━━━━━━━", parse_mode='Markdown')
                except:
                    await query.message.reply_text(f"📍 {s} GÜNCELLEME\n{res['analysis']}\n━━━━━━━━━━━━━━━━━━")
    elif data == 'bilgi_al':
        text = ("📚 *BORSA SÖZLÜĞÜ*\n\n"
                "🏢 *F/K*: Fiyat/Kazanç. Uçuk (Örn: 500+) çıkması şirketin kârının çok az olduğunu veya veri hatası olduğunu (bölünmeler vs.) gösterir.\n"
                "🏢 *PD/DD*: Piyasa/Defter değeri. Şirketin defterdeki ederinin kaç katına satıldığıdır.\n"
                "🏢 *FD/FAVÖK*: Firma Değeri / FAVÖK. Kârlılık ve borcu iyi harmanlayan güvenilir rasyodur. (8-10 altı olumludur).\n"
                "🏢 *Adil Değer*: Benjamin Graham formülüyle hesaplanan matematiksel fiyattır.\n"
                "🔹 *RSI*: Momentum. 70 üstü aşırı alım, 30 altı aşırı satımdır.\n"
                "🔹 *RSI Uyuşmazlığı*: Fiyat düşerken RSI yükseliyorsa ('Pozitif Uyuşmazlık') dönüş yaklaşmış demektir.\n"
                "🤖 *Algoritmik Sinyal*: Farklı göstergelerin aynı anda kesişmesiyle oluşan (Örn: RSI + Hacim) tetikleyicilerdir."
        )
        await query.message.reply_text(text, parse_mode='Markdown')
    elif data.startswith('ekle_'):
        symbol = data.split('_')[1]
        watchlist = load_watchlist()
        if symbol not in watchlist:
            watchlist.append(symbol)
            save_watchlist(watchlist)
            await query.message.reply_text(f"✅ {symbol} listeye eklendi.")
    elif data == 'hisse_tara':
        scan_list = get_all_bist_tickers()        # Her zaman tam piyasa listesi
        source_label = f"BIST Piyasası ({len(scan_list)} hisse)"
        await query.message.reply_text(
            f"🔭 *FIRSAT TARAYICI*\n"
            f"_{source_label} taranıyor, lütfen bekleyin (60-120 sn)..._",
            parse_mode='Markdown'
        )
        results = await asyncio.to_thread(run_market_scan, scan_list)
        if not results:
            await query.message.reply_text("📭 Şu an kriterlerimizi karşılayan güçlü fırsat bulunamadı.")
            return
        lines = ["🎯 *AL FIRSATI RADAR:*\n"]
        for i, r in enumerate(results[:8], 1):
            grade = "🟢 GÜÇLÜ" if r['score'] >= 65 else ("🟡 İYİ" if r['score'] >= 50 else "⚪ İZLE")
            tag_str = " · ".join(r['tags'][:3])
            lines.append(
                f"*{i}. {r['symbol']}* {grade} ({r['score']}/100)\n"
                f"   💰 {r['price']} TL | RSI:{r['rsi']} | Hacim:{r['vol']}x\n"
                f"   📌 {tag_str}\n"
            )
        lines.append("\n_Detaylı analiz için hisse adını yazın._")
        keyboard = [[InlineKeyboardButton("🏠 Ana Menü", callback_data='ana_menu')]]
        try:
            await query.message.reply_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except Exception as e:
            logging.error(f"Tarayıcı mesaj hatası: {e}")
            await query.message.reply_text("\n".join(lines))
    elif data == 'ana_menu':
        await start(update, context)

async def handle_regular_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        text = update.message.text.strip().upper()
        if len(text) <= 6 and text.isalpha():
            context.args = [text]
            await analiz(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analiz", analiz))
    app.add_handler(CommandHandler("tahmin", tahmin))
    app.add_handler(CommandHandler("tara", tara))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_regular_text))
    app.add_handler(CallbackQueryHandler(button_handler))
    print("Milyoner Trader Botu Aktif (Bütüncül Sürüm)!")
    app.run_polling()


    # --- Mevcut kodlarının bittiği yerden itibaren (app.run_polling satırının öncesi) ---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --- KOYEB HEALTH CHECK & FAKE SERVER -----------------------
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import http.server
import socketserver
import threading

def run_health_check_server():
    # Koyeb'in atadığı PORT'u al, yoksa 8000 kullan
    port = int(os.getenv("PORT", 8000))
    handler = http.server.SimpleHTTPRequestHandler
    # 'allow_reuse_address' hata vermemesi için önemli
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"✅ Koyeb Health Check sunucusu {port} portunda aktif.")
        httpd.serve_forever()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --- ANA ÇALIŞTIRICI ----------------------------------------
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    # 1. Koyeb'i kandırmak için sahte sunucuyu ayrı bir kolda (thread) başlat
    threading.Thread(target=run_health_check_server, daemon=True).start()

    # 2. Telegram Botunu kur
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analiz", analiz))
    app.add_handler(CommandHandler("tahmin", tahmin))
    app.add_handler(CommandHandler("tara", tara))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_regular_text))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    print("🚀 Milyoner Trader Botu Aktif (Bütüncül Sürüm)!")
    
    # 3. Botu çalıştır
    app.run_polling()