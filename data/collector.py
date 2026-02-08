"""
Enhanced Data Collector with multiple sources
"""

import yfinance as yf
import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime, timedelta
import time
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """Collect stock data from multiple sources"""
    
    def __init__(self, db_path: str = "data/stock_data.db"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
        
        # Define Vietnamese stocks with multiple source formats
        self.stock_config = {
            "VIC": {
                "name": "Vingroup",
                "yahoo": ["VIC.HM", "VIC.VN", "VIC"],
                "tradingview": "VIC",
                "vcsc": "VIC"
            },
            "VHM": {
                "name": "Vinhomes",
                "yahoo": ["VHM.HM", "VHM.VN", "VHM"],
                "tradingview": "VHM",
                "vcsc": "VHM"
            },
            "FPT": {
                "name": "FPT Corporation",
                "yahoo": ["FPT.HM", "FPT.VN", "FPT"],
                "tradingview": "FPT",
                "vcsc": "FPT"
            },
            "VCB": {
                "name": "Vietcombank",
                "yahoo": ["VCB.HM", "VCB.VN", "VCB"],
                "tradingview": "VCB",
                "vcsc": "VCB"
            },
            "MWG": {
                "name": "Mobile World",
                "yahoo": ["MWG.HM", "MWG.VN", "MWG"],
                "tradingview": "MWG",
                "vcsc": "MWG"
            }
        }
        
        # VNINDEX alternatives
        self.index_config = {
            "VNINDEX": {
                "name": "VN-Index",
                "yahoo": ["^VNINDEX", "VNI", "VNINDEX.VN"],
                "tradingview": "VNINDEX",
                "vcsc": "VNINDEX"
            }
        }
        
        logger.info(f"Initialized EnhancedDataCollector with {len(self.stock_config)} stocks")
    
    def _init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            source TEXT DEFAULT 'yahoo',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date, source)
        )
        ''')
        
        self.connection.commit()
    
    def try_yahoo_formats(self, symbol_config, start_date=None, end_date=None):
        """Try multiple Yahoo Finance formats"""
        if not start_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        yahoo_formats = symbol_config.get('yahoo', [])
        
        for yahoo_symbol in yahoo_formats:
            try:
                logger.info(f"Trying Yahoo Finance: {yahoo_symbol}")
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    logger.info(f"✅ Yahoo success with {yahoo_symbol}: {len(df)} records")
                    df = df.reset_index()
                    df['symbol'] = symbol_config.get('tradingview', yahoo_symbol.split('.')[0])
                    df['source'] = 'yahoo'
                    
                    # Rename columns
                    df = df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    if 'Adj Close' in df.columns:
                        df = df.rename(columns={'Adj Close': 'adj_close'})
                    else:
                        df['adj_close'] = df['close']
                    
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    
                    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'source']
                    df = df[[col for col in required_cols if col in df.columns]]
                    
                    return df
                else:
                    logger.warning(f"Yahoo {yahoo_symbol}: No data")
                    
            except Exception as e:
                logger.debug(f"Yahoo {yahoo_symbol} failed: {e}")
                continue
        
        return pd.DataFrame()
    
    def download_from_tradingview(self, symbol, start_date=None, end_date=None):
        """Alternative: Download from TradingView (requires API)"""
        # Placeholder for TradingView API
        logger.info(f"TradingView download for {symbol} - Not implemented")
        return pd.DataFrame()
    
    def download_test_data(self, symbol, days=365):
        """Generate test data if no source works"""
        logger.warning(f"⚠️ Using test data for {symbol}")
        
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq='B')
        
        # Generate synthetic data
        np.random.seed(hash(symbol) % 10000)
        base_price = np.random.uniform(10, 100)
        
        returns = np.random.normal(0.0005, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'symbol': symbol,
            'date': dates.strftime('%Y-%m-%d'),
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, days)),
            'low': prices * (1 - np.random.uniform(0, 0.02, days)),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, days),
            'adj_close': prices,
            'source': 'test'
        })
        
        return df
    
    def download_stock_data(self, symbol, config):
        """Main download function with fallback"""
        logger.info(f"Downloading data for {symbol}")
        
        # Try Yahoo first
        df = self.try_yahoo_formats(config)
        
        # If Yahoo fails, try TradingView
        if df.empty:
            df = self.download_from_tradingview(symbol)
        
        # If all else fails, generate test data
        if df.empty:
            df = self.download_test_data(symbol)
        
        return df
    
    def update_all_stocks(self):
        """Update all stocks"""
        all_data = []
        
        # Update VNINDEX
        logger.info("Downloading VNINDEX...")
        vnindex_data = self.download_stock_data("VNINDEX", self.index_config["VNINDEX"])
        if not vnindex_data.empty:
            all_data.append(vnindex_data)
        
        # Update individual stocks
        for symbol, config in self.stock_config.items():
            logger.info(f"Downloading {symbol}...")
            stock_data = self.download_stock_data(symbol, config)
            
            if not stock_data.empty:
                all_data.append(stock_data)
            
            time.sleep(0.5)  # Rate limiting
        
        # Save to database
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['symbol', 'date', 'source'])
            
            # Save to database
            combined_df.to_sql('stock_prices', self.connection, if_exists='append', index=False)
            logger.info(f"✅ Saved {len(combined_df)} records to database")
            
            # Show summary
            summary = combined_df.groupby('symbol').agg({
                'date': ['min', 'max', 'count']
            }).round(2)
            
            print("\n📊 DOWNLOAD SUMMARY:")
            print("=" * 60)
            for symbol in summary.index:
                min_date = summary.loc[symbol, ('date', 'min')]
                max_date = summary.loc[symbol, ('date', 'max')]
                count = summary.loc[symbol, ('date', 'count')]
                print(f"{symbol:8} | {count:4} records | {min_date} to {max_date}")
            print("=" * 60)
            
            return len(combined_df)
        else:
            logger.error("❌ No data downloaded from any source")
            return 0
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()

def main():
    """Main function"""
    print("=" * 60)
    print("ENHANCED DATA COLLECTOR - MULTIPLE SOURCES")
    print("=" * 60)
    
    collector = EnhancedDataCollector()
    
    try:
        print("\n[1/3] Starting data collection...")
        records = collector.update_all_stocks()
        
        print(f"\n[2/3] Total records added: {records}")
        
        # Verify data
        if records > 0:
            query = "SELECT symbol, COUNT(*) as count FROM stock_prices GROUP BY symbol"
            df = pd.read_sql_query(query, collector.connection)
            
            print("\n[3/3] Database verification:")
            for _, row in df.iterrows():
                print(f"  {row['symbol']}: {row['count']} records")
            
            # Save backup
            backup_file = f"data/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            all_data = pd.read_sql_query("SELECT * FROM stock_prices", collector.connection)
            all_data.to_csv(backup_file, index=False)
            print(f"✅ Backup saved to: {backup_file}")
        else:
            print("⚠️ No data was collected")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        collector.close()
    
    print("\n" + "=" * 60)
    print("Data collection completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()