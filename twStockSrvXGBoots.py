from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import talib
import xgboost as xgb
import joblib
import os
import uvicorn
from sklearn.preprocessing import StandardScaler
import asyncio
from functools import lru_cache
import logging
from io import StringIO
import time

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加入 GZip 壓縮
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Redis 快取設定
# REDIS_URL = "redis://localhost:6379"
REDIS_URL = os.getenv("REDIS_URL","redis://default:sMQGdhDtCmLBcoOgLtLFmbQoREsGJMnr@redis.railway.internal:6379")
redis_client = redis.from_url(REDIS_URL)

# 模型路徑設定
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 快取時間設定（秒）
CACHE_TTL = 300  # 5分鐘
PRICE_UPDATE_THRESHOLD = 0.01  # 價格變動超過1%時強制更新
MIN_UPDATE_INTERVAL = 60  # 最小更新間隔（秒）
MAX_PRICE_CHANGE = 0.1  # 最大允許價格變動（10%）
MIN_HISTORY_DAYS = 5  # 最小歷史數據天數

# 請求限制設定
RATE_LIMIT = "100/minute"  # 每分鐘100個請求


def get_model_path(stock_id: str):
    """獲取指定股票的模型路徑"""
    return {
        "model": os.path.join(MODELS_DIR, f"stock_{stock_id}_xgboost_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, f"stock_{stock_id}_scaler.pkl")
    }

def train_stock_model(stock_id: str, df: pd.DataFrame):
    """訓練指定股票的模型"""
    try:
        # 準備特徵和目標變數
        features, _ = prepare_features(df)
        target = df['Close'].shift(-1).dropna()  # 使用下一天的收盤價作為目標
        
        # 確保特徵和目標的長度一致
        min_length = min(len(features), len(target))
        features = features.iloc[:min_length]
        target = target.iloc[:min_length]
        
        # 移除任何包含 NaN 的行
        valid_indices = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_indices]
        target = target[valid_indices]
        
        # 確保特徵和目標的維度正確
        features = features.values  # 轉換為 numpy array
        target = target.values.reshape(-1)  # 確保目標是一維數組
        
        # 檢查特徵和目標的形狀
        logger.info(f"特徵形狀: {features.shape}, 目標形狀: {target.shape}")
        
        if len(features) < 20:  # 增加最小數據量要求
            raise ValueError(f"股票 {stock_id} 的歷史數據不足（{len(features)}筆），至少需要20筆數據進行訓練")
        
        # 確保特徵和目標的樣本數相同
        if len(features) != len(target):
            raise ValueError(f"特徵和目標的樣本數不一致：特徵 {len(features)} 筆，目標 {len(target)} 筆")
        
        # 分割訓練集和測試集
        train_size = int(len(features) * 0.8)
        X_train = features[:train_size]
        y_train = target[:train_size]
        X_test = features[train_size:]
        y_test = target[train_size:]
        
        # 標準化特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 設定 XGBoost 參數
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # 訓練模型
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # 儲存模型和 scaler
        paths = get_model_path(stock_id)
        joblib.dump(model, paths["model"])
        joblib.dump(scaler, paths["scaler"])
        
        logger.info(f"股票 {stock_id} 的模型訓練完成並儲存，使用 {len(features)} 筆數據")
        return model, scaler
        
    except Exception as e:
        logger.error(f"訓練股票 {stock_id} 的模型時發生錯誤: {str(e)}")
        raise ValueError(f"訓練股票 {stock_id} 的模型時發生錯誤: {str(e)}")

@lru_cache(maxsize=100)
def load_model_and_scaler(stock_id: str):
    """載入指定股票的預訓練模型和 scaler"""
    try:
        paths = get_model_path(stock_id)
        model_path = paths["model"]
        scaler_path = paths["scaler"]

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.info(f"股票 {stock_id} 的模型不存在，開始訓練新模型...")
            # 獲取歷史數據
            df = get_stock_data(stock_id)
            df = calculate_technical_indicators(df)
            # 訓練新模型
            model, scaler = train_stock_model(stock_id, df)
            return model, scaler

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logger.error(f"載入股票 {stock_id} 的模型時發生錯誤: {str(e)}")
        raise ValueError(f"載入股票 {stock_id} 的模型時發生錯誤: {str(e)}")


def calculate_technical_indicators(df):
    """計算技術指標"""
    try:
        # 檢查輸入數據
        if df.empty:
            raise ValueError("輸入的數據為空")
            
        # 創建數據的副本以避免 SettingWithCopyWarning
        df = df.copy()
        
        # 記錄初始數據量
        initial_length = len(df)
        logger.info(f"初始數據量: {initial_length}")
            
        # 確保必要的列存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的數據列: {missing_columns}")
            
        # 確保數據類型正確
        for col in required_columns:
            # 先轉換為 float64
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            # 使用 numpy 的 astype 確保數據類型正確
            df.loc[:, col] = np.array(df[col].values, dtype=np.float64)
        
        # 檢查數據量是否減少
        if len(df) < initial_length:
            logger.warning(f"數據量從 {initial_length} 減少到 {len(df)}，可能因為 NaN 值被移除")
        
        # 檢查數據量是否足夠
        if len(df) < 20:
            raise ValueError(f"數據量不足（{len(df)}筆），至少需要20筆數據進行分析")
            
        # 將數據轉換為 numpy float64 類型並確保是連續的
        close_values = np.array(df['Close'].values, dtype=np.float64)
        volume_values = np.array(df['Volume'].values, dtype=np.float64)
        
        # 確保數據是連續的
        close_values = np.ascontiguousarray(close_values)
        volume_values = np.ascontiguousarray(volume_values)
        
        # 檢查數據是否有效
        if np.isnan(close_values).any() or np.isnan(volume_values).any():
            raise ValueError("數據中包含無效值")
            
        # 記錄數據類型
        logger.info(f"Close 數據類型: {close_values.dtype}")
        logger.info(f"Volume 數據類型: {volume_values.dtype}")
        logger.info(f"Close 數據示例: {close_values[:5]}")
        logger.info(f"Volume 數據示例: {volume_values[:5]}")
        
        # 基本技術指標
        try:
            # 確保輸入數據是 double 類型
            close_values = np.array(close_values, dtype=np.float64)
            volume_values = np.array(volume_values, dtype=np.float64)
            
            # 計算技術指標 - 調整為適合台股的參數
            df.loc[:, 'SMA5'] = talib.SMA(close_values, timeperiod=5)  # 5日均線
            df.loc[:, 'SMA10'] = talib.SMA(close_values, timeperiod=10)  # 10日均線
            df.loc[:, 'SMA20'] = talib.SMA(close_values, timeperiod=20)  # 月線
            df.loc[:, 'SMA60'] = talib.SMA(close_values, timeperiod=60)  # 季線
            df.loc[:, 'RSI'] = talib.RSI(close_values, timeperiod=6)  # 6日RSI更適合台股
            df.loc[:, 'MACD'], df.loc[:, 'MACD_Signal'], _ = talib.MACD(
                close_values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            df.loc[:, 'BB_Upper'], df.loc[:, 'BB_Middle'], df.loc[:, 'BB_Lower'] = talib.BBANDS(
                close_values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            
            # 計算成交量指標
            df.loc[:, 'Volume_SMA5'] = talib.SMA(volume_values, timeperiod=5)
            df.loc[:, 'Volume_Ratio'] = df['Volume'] / df['Volume_SMA5']
            
            # 計算KD指標 - 更適合台股的參數
            df.loc[:, 'K'], df.loc[:, 'D'] = talib.STOCH(
                df['High'], df['Low'], df['Close'],
                fastk_period=9, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            
        except Exception as e:
            logger.error(f"計算技術指標時發生錯誤: {str(e)}")
            logger.error(f"Close 數據示例: {close_values[:5]}")
            logger.error(f"Close 數據類型: {close_values.dtype}")
            logger.error(f"Close 數據形狀: {close_values.shape}")
            logger.error(f"Close 數據是否連續: {close_values.flags['C_CONTIGUOUS']}")
            raise

        # 處理 NaN 值
        for col in df.columns:
            if df[col].isna().any():
                # 記錄 NaN 值的數量
                nan_count = df[col].isna().sum()
                logger.info(f"技術指標 {col} 中有 {nan_count} 個 NaN 值")
                
                # 使用前向填充
                df.loc[:, col] = df[col].ffill()
                # 使用後向填充
                df.loc[:, col] = df[col].bfill()
                # 如果還有 NaN 值，使用該列的中位數填充
                if df[col].isna().any():
                    median_value = df[col].median()
                    if pd.isna(median_value):
                        # 如果中位數也是 NaN，使用該列的第一個非 NaN 值
                        first_valid = df[col].first_valid_index()
                        if first_valid is not None:
                            median_value = df[col].loc[first_valid]
                        else:
                            raise ValueError(f"無法找到有效的填充值 for {col}")
                    df.loc[:, col] = df[col].fillna(median_value)
                    logger.info(f"技術指標 {col} 使用值 {median_value} 填充 NaN 值")

        # 最終檢查數據量
        if len(df) < 20:
            raise ValueError(f"處理後數據量不足（{len(df)}筆），至少需要20筆數據進行分析")

        return df
    except Exception as e:
        logger.error(f"計算技術指標時發生錯誤: {str(e)}")
        raise


def prepare_features(df):
    """準備特徵數據"""
    try:
        # 記錄初始數據量
        initial_length = len(df)
        logger.info(f"準備特徵前數據量: {initial_length}")
        
        # 計算技術指標
        df = calculate_technical_indicators(df)
        
        # 檢查數據量是否減少
        if len(df) < initial_length:
            logger.warning(f"技術指標計算後數據量從 {initial_length} 減少到 {len(df)}")
            raise ValueError(f"技術指標計算後數據量不足（{len(df)}筆），至少需要20筆數據進行分析")
        
        # 選擇特徵列
        feature_columns = [
            'SMA5', 'SMA10', 'SMA20', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_Ratio'
        ]
        
        # 確保所有特徵列都存在
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"缺少必要的特徵列: {missing_features}")
        
        # 提取特徵
        features = df[feature_columns].copy()
        
        # 檢查特徵是否為空
        if features.empty:
            raise ValueError("特徵數據為空")
            
        # 確保特徵數據的類型正確
        for col in feature_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            
        # 處理 NaN 值
        for col in feature_columns:
            # 記錄 NaN 值的數量
            nan_count = features[col].isna().sum()
            if nan_count > 0:
                logger.info(f"特徵 {col} 中有 {nan_count} 個 NaN 值")
                
            # 使用前向填充
            features[col] = features[col].ffill()
            # 使用後向填充
            features[col] = features[col].bfill()
            # 如果還有 NaN 值，使用該列的中位數填充
            if features[col].isna().any():
                median_value = features[col].median()
                if pd.isna(median_value):
                    # 如果中位數也是 NaN，使用該列的第一個非 NaN 值
                    first_valid = features[col].first_valid_index()
                    if first_valid is not None:
                        median_value = features[col].loc[first_valid]
                    else:
                        raise ValueError(f"無法找到有效的填充值 for {col}")
                features[col] = features[col].fillna(median_value)
                logger.info(f"特徵 {col} 使用值 {median_value} 填充 NaN 值")
            
        # 最後一次檢查是否有 NaN 值
        if features.isna().any().any():
            # 記錄哪些列還有 NaN 值
            nan_columns = features.columns[features.isna().any()].tolist()
            raise ValueError(f"特徵數據中仍然包含 NaN 值，受影響的列: {nan_columns}")
            
        # 檢查最終數據量
        if len(features) < 20:
            raise ValueError(f"特徵準備後數據量不足（{len(features)}筆），至少需要20筆數據進行分析")
            
        return features, df
    except Exception as e:
        logger.error(f"準備特徵時發生錯誤: {str(e)}")
        raise


def get_stock_data(stock_id: str):
    """獲取股票數據"""
    try:
        tz = pytz.timezone('Asia/Taipei')
        end_date = datetime.now(tz)
        start_date = end_date - timedelta(days=365)  # 增加為一年歷史數據

        # 判斷是上市還是上櫃股票
        # 上市股票代號範圍：1000-9999
        # 上櫃股票代號範圍：1000-9999
        # 需要根據實際情況判斷
        try:
            stock_num = int(stock_id)
            # 先嘗試上市股票
            ticker_symbol = f"{stock_id}.TW"
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if df.empty:
                # 如果上市股票沒有資料，嘗試上櫃股票
                ticker_symbol = f"{stock_id}.TWO"
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                is_otc = True
            else:
                is_otc = False
                
            logger.info(f"獲取股票 {stock_id} 資料，市場: {'上櫃' if is_otc else '上市'}")
            
        except Exception as e:
            logger.error(f"判斷股票市場時發生錯誤: {str(e)}")
            raise ValueError(f"無法判斷股票 {stock_id} 的市場")

        # 嘗試多次獲取數據
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if df.empty:
                    logger.warning(f"第 {attempt + 1} 次嘗試獲取股票 {stock_id} 數據失敗，數據為空")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 等待2秒後重試
                        continue
                    raise ValueError(f"無法獲取股票 {stock_id} 的資料")

                # 檢查數據量
                if len(df) < 20:
                    logger.warning(f"第 {attempt + 1} 次嘗試獲取股票 {stock_id} 數據，但數據量不足（{len(df)}筆）")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 等待2秒後重試
                        continue
                    raise ValueError(f"股票 {stock_id} 的歷史數據不足（{len(df)}筆），至少需要20筆數據進行分析")

                # 如果數據量足夠，跳出重試循環
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"第 {attempt + 1} 次嘗試獲取股票 {stock_id} 數據失敗: {str(e)}")
                    time.sleep(2)  # 等待2秒後重試
                else:
                    raise

        # 創建數據的副本以避免 SettingWithCopyWarning
        df = df.copy()
        
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d')

        # 確保必要的列存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的數據列: {missing_columns}")

        # 確保數據類型正確
        for col in required_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[:, col] = df[col].astype(np.float64)

        # 移除任何包含 NaN 的行
        df = df.dropna(subset=required_columns)

        # 再次檢查數據量
        if len(df) < 20:
            raise ValueError(f"股票 {stock_id} 的歷史數據不足（{len(df)}筆），至少需要20筆數據進行分析")

        # 記錄數據統計信息
        logger.info(f"股票 {stock_id} 數據統計:")
        logger.info(f"市場: {'上櫃' if is_otc else '上市'}")
        logger.info(f"總數據筆數: {len(df)}")
        logger.info(f"數據日期範圍: {df['Date'].min()} 至 {df['Date'].max()}")
        logger.info(f"收盤價範圍: {df['Close'].min()} 至 {df['Close'].max()}")
        logger.info(f"成交量範圍: {df['Volume'].min()} 至 {df['Volume'].max()}")
        for col in required_columns:
            logger.info(f"{col}: {df[col].dtype}")

        return df
    except Exception as e:
        logger.error(f"獲取股票 {stock_id} 資料時發生錯誤: {str(e)}")
        raise ValueError(f"獲取股票 {stock_id} 資料時發生錯誤: {str(e)}")


def generate_prediction(df, model, scaler):
    """生成預測和交易建議"""
    try:
        # 記錄初始數據量
        initial_length = len(df)
        logger.info(f"生成預測前數據量: {initial_length}")
        
        if len(df) < MIN_HISTORY_DAYS:
            raise ValueError(f"歷史數據不足（{len(df)}筆），至少需要{MIN_HISTORY_DAYS}筆數據進行分析")

        # 使用完整的數據集準備特徵
        features, _ = prepare_features(df)
        
        if features.empty:
            raise ValueError("無法準備特徵數據")

        # 標準化特徵
        features_scaled = scaler.transform(features)

        # 使用 XGBoost 預測
        # 將特徵轉換為 numpy array
        features_array = np.array(features_scaled, dtype=np.float32)
        # 直接使用 numpy array 進行預測
        predicted_price = float(model.predict(features_array)[-1])  # 使用最後一筆預測
        current_price = float(df.iloc[-1]['Close'])

        # 計算預期漲跌幅
        expected_return = float((predicted_price - current_price) / current_price * 100)

        # 限制預測價格變動範圍 - 調整為台股的漲跌幅限制
        if abs(expected_return) > 10:  # 台股漲跌幅限制為10%
            historical_returns = df['Close'].pct_change().dropna()
            avg_return = historical_returns.mean() * 100
            if abs(avg_return) > 10:
                avg_return = 10 if avg_return > 0 else -10
            expected_return = avg_return
            predicted_price = current_price * (1 + expected_return / 100)
            logger.warning(f"預測價格變動過大，使用歷史平均變動: {expected_return:.2f}%")

        # 技術指標綜合分析
        technical_score = 0
        technical_indicators = []

        # RSI 分析 - 調整為台股的參數
        rsi = float(df.iloc[-1]['RSI'])
        if rsi < 20:  # 台股超賣區間較低
            technical_score += 1
            technical_indicators.append("RSI處於超賣區域")
        elif rsi > 80:  # 台股超買區間較高
            technical_score -= 1
            technical_indicators.append("RSI處於超買區域")
        else:
            technical_indicators.append("RSI處於正常區域")

        # KD 分析
        k = float(df.iloc[-1]['K'])
        d = float(df.iloc[-1]['D'])
        if k < 20 and d < 20:  # KD低檔黃金交叉
            technical_score += 1
            technical_indicators.append("KD處於低檔黃金交叉")
        elif k > 80 and d > 80:  # KD高檔死亡交叉
            technical_score -= 1
            technical_indicators.append("KD處於高檔死亡交叉")
        else:
            technical_indicators.append("KD處於正常區域")

        # MACD 分析
        macd = float(df.iloc[-1]['MACD'])
        macd_signal = float(df.iloc[-1]['MACD_Signal'])
        if macd > macd_signal:
            technical_score += 1
            technical_indicators.append("MACD呈現黃金交叉")
        else:
            technical_score -= 1
            technical_indicators.append("MACD呈現死亡交叉")

        # 布林通道分析
        close_price = float(df.iloc[-1]['Close'])
        bb_upper = float(df.iloc[-1]['BB_Upper'])
        bb_lower = float(df.iloc[-1]['BB_Lower'])
        if close_price > bb_upper:
            technical_score -= 1
            technical_indicators.append("價格突破布林上軌")
        elif close_price < bb_lower:
            technical_score += 1
            technical_indicators.append("價格突破布林下軌")
        else:
            technical_indicators.append("價格在布林通道內")

        # 成交量分析
        volume_ratio = float(df.iloc[-1]['Volume_Ratio'])
        if volume_ratio > 2.0:  # 台股成交量放大標準較高
            technical_score += 1
            technical_indicators.append("成交量明顯放大")
        elif volume_ratio < 0.5:
            technical_score -= 1
            technical_indicators.append("成交量明顯萎縮")

        # 趨勢分析
        sma5 = float(df.iloc[-1]['SMA5'])
        sma20 = float(df.iloc[-1]['SMA20'])
        sma60 = float(df.iloc[-1]['SMA60'])
        if sma5 > sma20 and sma20 > sma60:  # 多頭排列
            technical_score += 2
            technical_indicators.append("均線呈現多頭排列")
        elif sma5 < sma20 and sma20 < sma60:  # 空頭排列
            technical_score -= 2
            technical_indicators.append("均線呈現空頭排列")
        else:
            technical_indicators.append("均線呈現盤整格局")

        # 生成交易建議 - 調整為台股的判斷標準
        if expected_return > 5 and technical_score > 2:  # 台股漲幅較大時才建議強力買入
            advice = "預期強力介入"
            reason = f"預期強勢上漲 {expected_return:.2f}%，技術指標呈現多頭排列：{', '.join(technical_indicators)}"
        elif expected_return > 3 and technical_score > 0:
            advice = "預期介入"
            reason = f"預期溫和上漲 {expected_return:.2f}%，技術面偏多：{', '.join(technical_indicators)}"
        elif expected_return < -5 and technical_score < -2:
            advice = "預期強力退出"
            reason = f"預期大幅下跌 {expected_return:.2f}%，技術指標呈現空頭排列：{', '.join(technical_indicators)}"
        elif expected_return < -3 and technical_score < 0:
            advice = "預期退出"
            reason = f"預期下跌 {expected_return:.2f}%，技術面偏空：{', '.join(technical_indicators)}"
        else:
            advice = "預期觀望"
            reason = f"預期波動幅度較小（{expected_return:.2f}%），技術指標顯示：{', '.join(technical_indicators)}"

        return {
            "預期價格": round(predicted_price, 2),
            "預期漲跌": f"{expected_return:.2f}%",
            "技術分數": int(technical_score),
            "技術分析參考": advice,
            "技術分析說明": reason
        }
    except Exception as e:
        logger.error(f"生成預測時發生錯誤: {str(e)}")
        return None


async def get_cached_stock_data(stock_id: str):
    """從快取獲取股票數據，如果不存在則從 Yahoo Finance 獲取"""
    cache_key = f"stock_data:{stock_id}"
    timestamp_key = f"stock_data_timestamp:{stock_id}"
    
    cached_data = await redis_client.get(cache_key)
    cached_timestamp = await redis_client.get(timestamp_key)
    
    current_time = datetime.now().timestamp()
    
    if cached_data and cached_timestamp:
        try:
            # 檢查是否超過最小更新間隔
            if current_time - float(cached_timestamp) < MIN_UPDATE_INTERVAL:
                json_str = cached_data.decode('utf-8')
                return pd.read_json(StringIO(json_str))
        except Exception as e:
            logger.error(f"解析快取時間戳記時發生錯誤: {str(e)}")
    
    # 獲取新資料
    df = get_stock_data(stock_id)
    try:
        json_str = df.to_json()
        await redis_client.setex(cache_key, CACHE_TTL, json_str)
        await redis_client.setex(timestamp_key, CACHE_TTL, str(current_time))
    except Exception as e:
        logger.error(f"儲存快取資料時發生錯誤: {str(e)}")
    
    return df


async def get_cached_prediction(stock_id: str, df: pd.DataFrame, model, scaler):
    """從快取獲取預測結果，如果不存在則生成新的預測"""
    cache_key = f"prediction:{stock_id}"
    timestamp_key = f"prediction_timestamp:{stock_id}"
    last_price_key = f"last_price:{stock_id}"
    
    cached_prediction = await redis_client.get(cache_key)
    cached_timestamp = await redis_client.get(timestamp_key)
    last_price = await redis_client.get(last_price_key)
    
    current_time = datetime.now().timestamp()
    current_price = float(df.iloc[-1]["Close"])
    
    # 檢查是否需要強制更新
    should_update = False
    if last_price:
        try:
            last_price = float(last_price)
            price_change = abs((current_price - last_price) / last_price)
            if price_change > PRICE_UPDATE_THRESHOLD:
                should_update = True
                logger.info(f"股票 {stock_id} 價格變動超過 {PRICE_UPDATE_THRESHOLD*100}%，強制更新預測")
        except Exception as e:
            logger.error(f"計算股票 {stock_id} 價格變動時發生錯誤: {str(e)}")
    
    if cached_prediction and cached_timestamp and not should_update:
        try:
            if current_time - float(cached_timestamp) < MIN_UPDATE_INTERVAL:
                prediction_str = cached_prediction.decode('utf-8')
                return eval(prediction_str)
        except Exception as e:
            logger.error(f"解析股票 {stock_id} 快取預測時發生錯誤: {str(e)}")
    
    # 生成新預測
    prediction = generate_prediction(df, model, scaler)
    if prediction:
        try:
            prediction_str = str(prediction)
            await redis_client.setex(cache_key, CACHE_TTL, prediction_str)
            await redis_client.setex(timestamp_key, CACHE_TTL, str(current_time))
            await redis_client.setex(last_price_key, CACHE_TTL, str(current_price))
        except Exception as e:
            logger.error(f"儲存股票 {stock_id} 快取預測時發生錯誤: {str(e)}")
    
    return prediction


@app.on_event("startup")
async def startup():
    """應用程式啟動時初始化"""
    await FastAPILimiter.init(redis_client)

@app.get("/predict/{stock_id}", dependencies=[Depends(RateLimiter(times=100, minutes=1))])
async def predict(stock_id: str):
    try:
        # 載入模型和 scaler
        model, scaler = load_model_and_scaler(stock_id)

        # 獲取股票數據
        df = await get_cached_stock_data(stock_id)
        if len(df) < 20:
            raise ValueError(f"股票 {stock_id} 的歷史數據不足（{len(df)}筆），至少需要20筆數據進行分析")

        df = calculate_technical_indicators(df)

        # 生成預測
        prediction = await get_cached_prediction(stock_id, df, model, scaler)

        if prediction is None:
            raise ValueError("無法生成預測")

        result = {
            "stock": stock_id,
            "price": round(float(df.iloc[-1]["Close"]), 2),  # 四捨五入到小數點後兩位
            "suggestion": prediction,
            "update": datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')
        }

        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )

    except ValueError as e:
        logger.error(f"預測股票 {stock_id} 時發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"預測股票 {stock_id} 時發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"預測股票 {stock_id} 時發生錯誤: {str(e)}"
        )

if __name__ == "__main__":
    # SSL 憑證設定
    ssl_keyfile = "ssl/private.key"  # SSL 私鑰路徑
    ssl_certfile = "ssl/certificate.crt"  # SSL 憑證路徑
    
    # 檢查 SSL 憑證是否存在
    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        logger.info("使用 HTTPS 通訊協定")
        ssl_context = (ssl_certfile, ssl_keyfile)
    else:
        logger.warning("未找到 SSL 憑證，使用 HTTP 通訊協定")
        ssl_context = None
    
    # 設定 uvicorn 的 worker 數量
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=2,  # 根據 CPU 核心數調整
        limit_concurrency=1000,  # 最大並行連接數
        timeout_keep_alive=30,  # 保持連接超時時間
        log_level="info",
        ssl_keyfile=ssl_keyfile if ssl_context else None,
        ssl_certfile=ssl_certfile if ssl_context else None
    )