import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pymysql
from flask import Flask, jsonify, render_template
from pymysql.cursors import DictCursor
import yfinance as yf

app = Flask(__name__)

SYMBOLS = {'台積電': '2330.TW', '廣達': '2382.TW'}
PERIOD_DAYS = 365
VERSION = '2026-05-23-v3'


@app.before_request
def _ensure_db():
    app.before_request_funcs[None].remove(_ensure_db)
    try:
        init_db()
    except Exception as e:
        app.logger.error(f'init_db failed: {e}')


def get_conn():
    return pymysql.connect(
        host=os.environ.get('MYSQL_HOST', 'localhost'),
        port=int(os.environ.get('MYSQL_PORT', 3306)),
        user=os.environ.get('MYSQL_USER') or os.environ.get('MYSQL_USERNAME', 'root'),
        password=os.environ.get('MYSQL_PASSWORD', ''),
        database=os.environ.get('MYSQL_DATABASE') or os.environ.get('MYSQL_DB', 'stockdb'),
        charset='utf8mb4',
        cursorclass=DictCursor,
    )


def init_db():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                open_price DOUBLE,
                high_price DOUBLE,
                low_price DOUBLE,
                close_price DOUBLE,
                volume BIGINT,
                UNIQUE KEY uq_symbol_date (symbol, trade_date)
            ) DEFAULT CHARSET=utf8mb4
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS correlation_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                calc_date DATE NOT NULL,
                period_days INT NOT NULL,
                correlation DOUBLE,
                p_value DOUBLE,
                data_points INT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_calc_date (calc_date, period_days)
            ) DEFAULT CHARSET=utf8mb4
        ''')
    conn.commit()
    conn.close()


def _download(symbol, start, end):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(), 'empty dataframe from yfinance'
    # 確保是 flat 欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, c)) for c in df.columns]
    df.index.name = 'Date'
    df = df.reset_index()
    # 找 Close 欄位（不同版本可能大小寫不同）
    col_map = {c.lower(): c for c in df.columns}
    needed = {'date': 'Date', 'open': 'Open', 'high': 'High',
              'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    rename = {}
    for lower, standard in needed.items():
        if lower in col_map and col_map[lower] != standard:
            rename[col_map[lower]] = standard
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in needed.values() if c not in df.columns]
    if missing:
        return pd.DataFrame(), f'missing columns: {missing}, got: {list(df.columns)}'
    df = df[list(needed.values())].copy()
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
    df = df.dropna(subset=['Close'])
    return df, None


def _read_prices(conn, symbol):
    with conn.cursor() as cur:
        cur.execute(
            'SELECT trade_date, close_price FROM stock_prices '
            'WHERE symbol=%s ORDER BY trade_date',
            (symbol,),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=['trade_date', 'close_price'])
    df = pd.DataFrame(rows)
    # 轉成純 Python float，避免 Decimal 型別問題
    df['close_price'] = [float(v) if v is not None else np.nan
                         for v in df['close_price']]
    return df


def _pearsonr(x, y):
    """用 numpy 計算皮爾森相關係數，不依賴 scipy。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    # 近似 p-value
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-15))
    from math import lgamma, exp, sqrt, pi
    def betai(a, b, x):
        if x < 0 or x > 1:
            return 0.0
        if x == 0:
            return 0.0
        if x == 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        return exp(a * np.log(x) + b * np.log(1 - x) - lbeta) / (a)
    p = 2 * betai(0.5 * (n - 2), 0.5, (n - 2) / (t * t + n - 2)) if t != 0 else 1.0
    return r, p


def fetch_and_store():
    end = datetime.now()
    start = end - timedelta(days=PERIOD_DAYS + 30)
    log = []

    conn = get_conn()
    rows_upserted = 0

    for name, sym in SYMBOLS.items():
        df, err = _download(sym, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if err:
            log.append(f'{sym} download error: {err}')
            continue
        log.append(f'{sym} downloaded {len(df)} rows, cols={list(df.columns)}')
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                try:
                    cur.execute(
                        '''
                        INSERT INTO stock_prices
                            (symbol, trade_date, open_price, high_price,
                             low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            open_price  = VALUES(open_price),
                            high_price  = VALUES(high_price),
                            low_price   = VALUES(low_price),
                            close_price = VALUES(close_price),
                            volume      = VALUES(volume)
                        ''',
                        (
                            sym,
                            pd.Timestamp(row['Date']).strftime('%Y-%m-%d'),
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row['Volume']),
                        ),
                    )
                    rows_upserted += 1
                except Exception as e:
                    log.append(f'insert error: {e}')
        conn.commit()

    tsmc_df = _read_prices(conn, '2330.TW')
    quanta_df = _read_prices(conn, '2382.TW')
    log.append(f'DB: tsmc={len(tsmc_df)} quanta={len(quanta_df)}')

    merged = pd.merge(tsmc_df, quanta_df, on='trade_date', suffixes=('_tsmc', '_quanta'))
    merged = merged[merged['close_price_tsmc'].notna() & merged['close_price_quanta'].notna()]
    log.append(f'merged={len(merged)}, dtypes={merged.dtypes.to_dict()}')

    if len(merged) >= 3:
        x = np.array([float(v) for v in merged['close_price_tsmc']])
        y = np.array([float(v) for v in merged['close_price_quanta']])
        corr, pval = _pearsonr(x, y)
        with conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO correlation_log
                    (calc_date, period_days, correlation, p_value, data_points)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    correlation = VALUES(correlation),
                    p_value     = VALUES(p_value),
                    data_points = VALUES(data_points)
                ''',
                (
                    datetime.now().strftime('%Y-%m-%d'),
                    PERIOD_DAYS,
                    round(corr, 6),
                    round(pval, 8) if pval else None,
                    len(merged),
                ),
            )
        conn.commit()
        log.append(f'corr={corr:.4f}')

    conn.close()
    return rows_upserted, log


@app.route('/api/version')
def api_version():
    return jsonify({'version': VERSION, 'time': datetime.now().isoformat()})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/update', methods=['POST'])
def api_update():
    try:
        n, log = fetch_and_store()
        return jsonify({'status': 'ok', 'rows': n, 'log': log})
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/api/chart-data')
def api_chart_data():
    conn = get_conn()

    tsmc_df = _read_prices(conn, '2330.TW')
    quanta_df = _read_prices(conn, '2382.TW')
    merged = pd.merge(tsmc_df, quanta_df, on='trade_date', suffixes=('_tsmc', '_quanta'))
    merged = merged[merged['close_price_tsmc'].notna() & merged['close_price_quanta'].notna()]

    with conn.cursor() as cur:
        cur.execute(
            'SELECT * FROM correlation_log ORDER BY calc_date DESC, id DESC LIMIT 1'
        )
        corr_row = cur.fetchone()

    conn.close()

    if merged.empty:
        return jsonify({
            'dates': [], 'tsmc_price': [], 'quanta_price': [],
            'tsmc_norm': [], 'quanta_norm': [], 'rolling_corr': [],
            'correlation': None, 'p_value': None,
            'data_points': 0, 'last_updated': None,
        })

    tsmc_close = pd.to_numeric(merged['close_price_tsmc'], errors='coerce')
    quanta_close = pd.to_numeric(merged['close_price_quanta'], errors='coerce')

    norm_tsmc = (tsmc_close / tsmc_close.iloc[0] * 100).round(4).tolist()
    norm_quanta = (quanta_close / quanta_close.iloc[0] * 100).round(4).tolist()
    rolling_corr = tsmc_close.rolling(30).corr(quanta_close).fillna(0).round(6).tolist()

    return jsonify({
        'dates': merged['trade_date'].astype(str).tolist(),
        'tsmc_price': tsmc_close.round(2).tolist(),
        'quanta_price': quanta_close.round(2).tolist(),
        'tsmc_norm': norm_tsmc,
        'quanta_norm': norm_quanta,
        'rolling_corr': rolling_corr,
        'correlation': float(corr_row['correlation']) if corr_row else None,
        'p_value': float(corr_row['p_value']) if corr_row and corr_row['p_value'] else None,
        'data_points': int(corr_row['data_points']) if corr_row else 0,
        'last_updated': str(corr_row['calc_date']) if corr_row else None,
    })


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
