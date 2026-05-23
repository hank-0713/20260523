import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pymysql
from flask import Flask, jsonify, render_template
from pymysql.cursors import DictCursor
from scipy import stats
import yfinance as yf

app = Flask(__name__)

SYMBOLS = {'台積電': '2330.TW', '廣達': '2382.TW'}
PERIOD_DAYS = 365


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
                open_price DECIMAL(12,4),
                high_price DECIMAL(12,4),
                low_price DECIMAL(12,4),
                close_price DECIMAL(12,4),
                volume BIGINT,
                UNIQUE KEY uq_symbol_date (symbol, trade_date)
            ) DEFAULT CHARSET=utf8mb4
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS correlation_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                calc_date DATE NOT NULL,
                period_days INT NOT NULL,
                correlation DECIMAL(10,6),
                p_value DECIMAL(10,8),
                data_points INT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_calc_date (calc_date, period_days)
            ) DEFAULT CHARSET=utf8mb4
        ''')
    conn.commit()
    conn.close()


def _download(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index.name = 'Date'
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def fetch_and_store():
    end = datetime.now()
    start = end - timedelta(days=PERIOD_DAYS + 30)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    conn = get_conn()
    rows_upserted = 0

    for _name, sym in SYMBOLS.items():
        df = _download(sym, start_str, end_str)
        if df.empty:
            continue
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                try:
                    cur.execute(
                        '''
                        INSERT INTO stock_prices
                            (symbol, trade_date, open_price, high_price, low_price, close_price, volume)
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
                            row['Date'].strftime('%Y-%m-%d'),
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row['Volume']),
                        ),
                    )
                    rows_upserted += 1
                except Exception:
                    pass
        conn.commit()

    # 計算並記錄相關係數
    tsmc_df = pd.read_sql(
        "SELECT trade_date, close_price FROM stock_prices WHERE symbol='2330.TW' ORDER BY trade_date",
        conn,
    )
    quanta_df = pd.read_sql(
        "SELECT trade_date, close_price FROM stock_prices WHERE symbol='2382.TW' ORDER BY trade_date",
        conn,
    )
    merged = pd.merge(tsmc_df, quanta_df, on='trade_date', suffixes=('_tsmc', '_quanta'))

    if len(merged) >= 3:
        corr, pval = stats.pearsonr(
            merged['close_price_tsmc'].astype(float),
            merged['close_price_quanta'].astype(float),
        )
        with conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO correlation_log (calc_date, period_days, correlation, p_value, data_points)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    correlation  = VALUES(correlation),
                    p_value      = VALUES(p_value),
                    data_points  = VALUES(data_points)
                ''',
                (
                    datetime.now().strftime('%Y-%m-%d'),
                    PERIOD_DAYS,
                    round(float(corr), 6),
                    round(float(pval), 8),
                    len(merged),
                ),
            )
        conn.commit()

    conn.close()
    return rows_upserted


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/update', methods=['POST'])
def api_update():
    try:
        n = fetch_and_store()
        return jsonify({'status': 'ok', 'rows': n})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/chart-data')
def api_chart_data():
    conn = get_conn()

    tsmc_df = pd.read_sql(
        "SELECT trade_date, close_price FROM stock_prices WHERE symbol='2330.TW' ORDER BY trade_date",
        conn,
    )
    quanta_df = pd.read_sql(
        "SELECT trade_date, close_price FROM stock_prices WHERE symbol='2382.TW' ORDER BY trade_date",
        conn,
    )
    merged = pd.merge(tsmc_df, quanta_df, on='trade_date', suffixes=('_tsmc', '_quanta'))

    corr_row = pd.read_sql(
        'SELECT * FROM correlation_log ORDER BY calc_date DESC, id DESC LIMIT 1',
        conn,
    )
    conn.close()

    if merged.empty:
        return jsonify({
            'dates': [], 'tsmc_price': [], 'quanta_price': [],
            'tsmc_norm': [], 'quanta_norm': [], 'rolling_corr': [],
            'correlation': None, 'p_value': None,
            'data_points': 0, 'last_updated': None,
        })

    tsmc_close = merged['close_price_tsmc'].astype(float)
    quanta_close = merged['close_price_quanta'].astype(float)

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
        'correlation': float(corr_row['correlation'].iloc[0]) if not corr_row.empty else None,
        'p_value': float(corr_row['p_value'].iloc[0]) if not corr_row.empty else None,
        'data_points': int(corr_row['data_points'].iloc[0]) if not corr_row.empty else 0,
        'last_updated': str(corr_row['calc_date'].iloc[0]) if not corr_row.empty else None,
    })


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
