#!/usr/bin/env python3
"""
COMPLETE TRADING BOT - FAITHFUL STRATEGY A/B IMPLEMENTATION
Author: Trading Bot System
Version: 4.0
Description: Implements Strategy A (-1% to +4%) and Strategy B (≤ -1%) exactly as per flowcharts
"""

import asyncio
import csv
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dinoascii
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from rich.align import Align
from rich.box import ROUNDED, SIMPLE
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Trading configuration
SYMBOL = "SOLFDUSD"
TOKEN = "SOL"
PAIR = "FDUSD"
TRADE_AMOUNT_USD = 5.1
MIN_TOKEN_VALUE_FOR_SELL = 5.1
DAILY_DIFF_LOWER_LIMIT = -1.0  # Strategy B threshold
DAILY_DIFF_UPPER_LIMIT = 4.0  # Strategy A threshold

# Files
INDICATORS_CSV = "pinescript_indicators.csv"
TRANSACTIONS_CSV = "transactions.csv"
STATE_FILE = "trading_state.json"
ERROR_LOG_CSV = "trading_errors.csv"

# Safety settings
ORDER_COOLDOWN_SECONDS = 1
MAX_ORDER_RETRIES = 3
ORDER_TIMEOUT_SECONDS = 30
TEST_MODE = False  # ⚠️ SET TO FALSE FOR REAL TRADING ⚠️

# CSV monitoring
CSV_UPDATE_CHECK_INTERVAL = 2  # seconds
CSV_STALE_THRESHOLD = 30  # seconds before considering CSV stale
CSV_BATCH_PROCESSING = True  # Process all new rows at once

# Display settings
STATUS_UPDATE_INTERVAL = 10  # seconds between status displays
VERBOSE_LOGGING = True  # Detailed logging

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
console = Console()

# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================


class TradingMode(Enum):
    """Trading mode based on wallet contents"""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class StrategyType(Enum):
    """Type of wave strategy"""

    MA_200_WAVE = "MA_200_WAVE"
    MA_350_WAVE = "MA_350_WAVE"
    MA_500_WAVE = "MA_500_WAVE"


class StrategyVariant(Enum):
    """Strategy variant based on daily diff"""

    A = "A"  # -1% to +4%
    B = "B"  # ≤ -1%


class Phase(Enum):
    """Current phase in trading cycle"""

    ENTRY_MONITORING = "ENTRY_MONITORING"
    ENTRY_SIGNAL_CONFIRMED = "ENTRY_SIGNAL_CONFIRMED"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_MONITORING = "EXIT_MONITORING"
    STOP_LOSS_ACTIVE = "STOP_LOSS_ACTIVE"


class SignalType(Enum):
    """Types of trading signals"""

    ENTRY_SETUP = "ENTRY_SETUP"
    ENTRY_CONFIRMED = "ENTRY_CONFIRMED"
    EXIT_SIGNAL = "EXIT_SIGNAL"
    STOP_LOSS = "STOP_LOSS"


# ============================================================================
# COLOR FUNCTIONS
# ============================================================================


class Colors:
    """Color definitions for consistent styling"""

    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    INFO = "cyan"
    MAGENTA = "magenta"
    WHITE = "white"
    BLUE = "blue"
    GRAY = "grey70"

    @staticmethod
    def get_strategy_color(variant):
        return "green" if variant == StrategyVariant.A else "red"

    @staticmethod
    def get_mode_color(mode):
        return (
            "green"
            if mode == TradingMode.BUY
            else "red"
            if mode == TradingMode.SELL
            else "yellow"
        )


def print_colored(text, color="white", style="", emoji="", indent=0):
    """Print colored text with emoji and optional indentation"""
    indent_str = " " * indent
    style_tags = ""
    if style == "bold":
        style_tags = "bold "
    elif style == "italic":
        style_tags = "italic "
    elif style == "underline":
        style_tags = "underline "

    console.print(
        f"{indent_str}{emoji} [{style_tags}{color}]{text}[/{style_tags}{color}]"
    )


def print_header(title, color=Colors.INFO, border_char="═", length=80):
    """Print a section header"""
    border = border_char * length
    print_colored(border, color)
    print_colored(f"   {title}", color, "bold", "📌")
    print_colored(border, color)


def print_step(step_num, description, color=Colors.WARNING):
    """Print a step in the process"""
    print_colored(f"   STEP {step_num}: {description}", color, "bold", "⚡")


def print_condition(condition, is_met=True, indent=2):
    """Print a condition with status"""
    status = "✅" if is_met else "❌"
    color = Colors.SUCCESS if is_met else Colors.ERROR
    print_colored(f"{status} {condition}", color, indent=indent)


def print_waiting(waiting_for, indent=2):
    """Print waiting status"""
    print_colored(
        f"⏳ Waiting for: {waiting_for}", Colors.WARNING, "italic", indent=indent
    )


def print_signal(signal_type, description, color=Colors.MAGENTA):
    """Print a trading signal"""
    print_colored(f"\n🎯 {signal_type}: {description}", color, "bold")


def print_transition(from_state, to_state, color=Colors.INFO):
    """Print state transition"""
    print_colored(f"\n🔄 Transition: {from_state} → {to_state}", color, "bold")


# ============================================================================
# DINOSAY VISUALIZATIONS
# ============================================================================


def display_dino(message, dino_type="default", title=""):
    """Display dinosaur with message"""
    try:
        dino = getattr(dinoascii, dino_type)(message)
        if title:
            print_header(title, Colors.INFO)
        console.print(dino)
    except Exception as e:
        print_colored(f"\n🦕 {message}\n", Colors.INFO)


def display_strategy_activation(strategy_variant, daily_diff):
    """Display strategy activation with dino"""
    if strategy_variant == StrategyVariant.A:
        title = "STRATEGY A ACTIVATED"
        range_desc = "-1% to +4%"
        dino_type = "triceratops"
        color = Colors.SUCCESS
    else:
        title = "STRATEGY B ACTIVATED"
        range_desc = "≤ -1%"
        dino_type = "trex"
        color = Colors.ERROR

    display_dino(f"Daily Diff: {daily_diff:.2f}% ({range_desc})", dino_type, title)
    print_colored(
        f"Following {strategy_variant.value} flowchart exactly...", color, "bold"
    )


def display_mode_banner(mode, wallet_info):
    """Display trading mode banner"""
    if mode == TradingMode.BUY:
        title = "BUY MODE ACTIVATED"
        dino_type = "happy"
        color = Colors.SUCCESS
        action = "Looking for entry signals to BUY tokens"
    elif mode == TradingMode.SELL:
        title = "SELL MODE ACTIVATED"
        dino_type = "raptor"
        color = Colors.ERROR
        action = "Looking for entry signals to validate existing position"
    else:
        title = "NEUTRAL MODE"
        dino_type = "sleep"
        color = Colors.WARNING
        action = "Insufficient funds for trading"

    display_dino(title, dino_type)

    print_colored("💰 WALLET STATUS:", color, "bold")
    for asset, info in wallet_info.items():
        print_colored(f"   • {asset}: {info}", Colors.WHITE)

    print_colored(f"\n🎯 ACTION: {action}", color, "bold")

    if mode == TradingMode.SELL:
        print_colored(
            "   ⚡ In SELL mode: Will skip BUY when entry signal occurs",
            Colors.WARNING,
            "italic",
        )
        print_colored(
            "   ⚡ Existing tokens will be validated at entry signal",
            Colors.WARNING,
            "italic",
        )


def display_position_status(action, position_info):
    """Display position opening/closing status"""
    if action == "OPENED":
        dino_type = "happy"
        color = Colors.SUCCESS
        title = "POSITION OPENED"
    else:
        dino_type = "sleep"
        color = Colors.INFO
        title = "POSITION CLOSED"

    display_dino(title, dino_type)

    for key, value in position_info.items():
        print_colored(f"   • {key}: {value}", color)


def display_stop_loss_activation():
    """Display stop loss activation"""
    display_dino("STOP LOSS TRIGGERED!", "trex_roar", "🚨 STOP LOSS ACTIVATED 🚨")
    print_colored(
        "MA_200 ≤ Fibo_23.6% - Activating stop loss flow", Colors.ERROR, "bold"
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_daily_diff(daily_diff_str):
    """Convert daily_diff string to percentage - using precomputed CSV value"""
    if pd.isna(daily_diff_str) or not isinstance(daily_diff_str, str):
        return 0.0

    try:
        clean = daily_diff_str.replace("%", "").replace("+", "").strip()
        return float(clean)
    except:
        return 0.0


def calculate_token_amount(usd_amount, current_price, safety_mgr):
    """Calculate token amount with safety checks"""
    if pd.isna(current_price) or current_price <= 0:
        return 0.0, 0.0

    # Apply safety adjustments if any
    adjusted_usd = usd_amount
    if safety_mgr:
        adjusted_usd = safety_mgr.adjust_trade_amount(usd_amount)

    token_amount = adjusted_usd / current_price
    return token_amount, adjusted_usd


# ============================================================================
# SAFETY MANAGER
# ============================================================================


class SafetyManager:
    """Manages safety checks for orders"""

    def __init__(self, client):
        self.client = client
        self.symbol_info = None
        self.last_order_time = 0

    def get_symbol_info(self):
        """Get symbol info from Binance"""
        try:
            self.symbol_info = self.client.get_symbol_info(SYMBOL)
            return self.symbol_info
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            return None

    def adjust_trade_amount(self, usd_amount):
        """Apply any safety adjustments to trade amount"""
        # Placeholder: No adjustment for now
        return usd_amount

    def can_place_order(self):
        """Check if order can be placed (cooldown, etc.)"""
        if time.time() - self.last_order_time < ORDER_COOLDOWN_SECONDS:
            return False
        return True

    def update_last_order_time(self):
        """Update last order time"""
        self.last_order_time = time.time()


# ============================================================================
# WALLET MANAGER
# ============================================================================


class WalletManager:
    """Manages wallet balances and modes"""

    def __init__(self, client):
        self.client = client

    def get_balance(self, asset, include_locked=True):
        """Get balance for asset"""
        try:
            account = self.client.get_account()
            for balance in account["balances"]:
                if balance["asset"] == asset:
                    free = float(balance["free"])
                    locked = float(balance["locked"])
                    return free + (locked if include_locked else 0)
            return 0.0
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0.0

    def get_token_value_usd(self, token):
        """Get token value in USD"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=f"{token}{PAIR}")
            price = float(ticker["price"])
            balance = self.get_balance(token)
            return balance * price
        except Exception as e:
            logger.error(f"Token value error: {e}")
            return 0.0

    def determine_mode(self):
        """Determine trading mode based on balances"""
        usd_balance = sum(self.get_balance(stable) for stable in [PAIR, "USDT", "USDC"])
        token_value = self.get_token_value_usd(TOKEN)

        if usd_balance >= TRADE_AMOUNT_USD:
            return TradingMode.BUY
        elif token_value >= MIN_TOKEN_VALUE_FOR_SELL:
            return TradingMode.SELL
        return TradingMode.NEUTRAL


# ============================================================================
# TRADE STATE MANAGEMENT
# ============================================================================


class TradeState:
    """Maintains comprehensive trading state"""

    def __init__(self):
        # Core state
        self.mode = TradingMode.NEUTRAL
        self.strategy_variant = None
        self.phase = Phase.ENTRY_MONITORING

        # Position tracking
        self.position_open = False
        self.position_entered_by_signal = False
        self.entry_price = 0.0
        self.entry_time = None
        self.position_size = 0.0
        self.virtual_entry_price = 0.0  # For SELL mode validation

        # Captured levels
        self.captured_fibo_0 = None
        self.captured_fibo_1 = None
        self.captured_fibo_1_dip = None

        # Strategy tracking
        self.active_strategy = None
        self.entry_signal_confirmed = False

        # Detailed waiting conditions
        self.waiting_conditions = {
            # Entry conditions
            "waiting_ma_200_above_fibo_236": False,
            "waiting_ma_350_above_fibo_236": False,
            "waiting_ma_500_above_fibo_236": False,
            # MA_200 Wave exit
            "waiting_ma_100_above_fibo_764": False,
            "waiting_ma_500_above_fibo_764": False,
            "waiting_ma_350_below_ma_500": False,
            # MA_350/500 Wave exit
            "waiting_ma_200_above_fibo_764": False,
            "waiting_lesser_mas_above_ma_200": False,
            "waiting_ma_100_below_ma_500": False,
            "waiting_ma_100_above_new_fibo_1": False,
            "waiting_ma_350_above_new_fibo_1": False,
            "waiting_ma_50_below_fibo_764": False,
            "waiting_rsi_ma50_above_55": False,
            "waiting_rsi_ma50_below_52": False,
            # Strategy B specific
            "waiting_strategy_b_profit_target": False,
            "waiting_strategy_b_reversal": False,
            # Stop loss conditions
            "waiting_ma_100_above_ma_200": False,
            "waiting_ma_100_below_ma_500": False,
            "waiting_ma_200_below_ma_500": False,
            "waiting_rsi_ma50_above_53": False,
            "waiting_rsi_ma50_below_51": False,
        }

        # Data tracking - enhanced for specific row tracking
        self.last_processed_time = None  # Timestamp of last processed row
        self.current_daily_diff = 0.0

        # Performance tracking
        self.stats = {
            "entry_signals_detected": 0,
            "exit_signals_detected": 0,
            "stop_loss_triggers": 0,
            "trades_completed": 0,
            "total_pnl_usd": 0.0,
            "total_pnl_percent": 0.0,
        }

        # Signal history with row timestamps
        self.signal_history = []

        # Managers
        self.safety_mgr = None
        self.wallet_mgr = None
        self.order_executor = None
        self.csv_monitor = None


# Global state
trade_state = TradeState()
client = None
running = True
state_lock = threading.Lock()

# ============================================================================
# CSV MONITORING
# ============================================================================


class CSVMonitor:
    """Monitors and processes CSV updates"""

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.last_modified = 0
        self.last_size = 0
        self.df_cache = pd.DataFrame()
        self.processed_rows = set()  # Track processed timestamps or indices
        self.update_stats = []

    def check_update(self):
        """Check if CSV has been updated"""
        try:
            if not self.csv_path.exists():
                return False, "NOT_FOUND"

            current_mtime = self.csv_path.stat().st_mtime
            current_size = self.csv_path.stat().st_size

            if current_mtime != self.last_modified or current_size != self.last_size:
                self.last_modified = current_mtime
                self.last_size = current_size

                # Record update
                self.update_stats.append(
                    {
                        "timestamp": datetime.now(),
                        "rows_loaded": 0,
                        "age": time.time() - current_mtime,
                    }
                )

                # Keep only last 100 updates
                if len(self.update_stats) > 100:
                    self.update_stats = self.update_stats[-100:]

                return True, "UPDATED"

            return False, "UNCHANGED"

        except Exception as e:
            logger.error(f"CSV check error: {e}")
            return False, "ERROR"

    def load_new_rows(self):
        """Load and return new rows based on timestamps/indices - track specific updates"""
        try:
            # Read CSV
            df = pd.read_csv(self.csv_path)

            # Convert timestamp
            if "Open Time" in df.columns:
                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")

            # Ensure numeric columns - use precomputed values directly
            numeric_cols = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "daily_diff",
                "rsi",
                "rsi_ma50",
                "short002",
                "short007",
                "short21",
                "short50",
                "long100",
                "long200",
                "long350",
                "long500",
                "level_100",
                "level_764",
                "level_618",
                "level_500",
                "level_382",
                "level_236",
                "level_000",
            ]

            for col in numeric_cols:
                if col in df.columns:
                    if col == "daily_diff":
                        # Keep as string for parsing later if needed
                        continue
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Find new rows using timestamp or index, tracking processed
            if self.df_cache.empty:
                new_rows = df
            else:
                # Prefer timestamp for volatility-based updates
                if "Open Time" in df.columns and "Open Time" in self.df_cache.columns:
                    last_time = self.df_cache["Open Time"].max()
                    new_rows = df[df["Open Time"] > last_time].copy()
                    # Mark as processed based on timestamp
                    if not new_rows.empty:
                        new_timestamps = set(
                            new_rows["Open Time"]
                            .dt.strftime("%Y-%m-%d %H:%M:%S")
                            .tolist()
                        )
                        self.processed_rows.update(new_timestamps)
                else:
                    # Fallback to index
                    cache_len = len(self.df_cache)
                    new_rows = (
                        df.iloc[cache_len:] if len(df) > cache_len else pd.DataFrame()
                    )
                    # Track indices if no time
                    new_indices = set(range(cache_len, len(df)))
                    self.processed_rows.update(new_indices)

            # Update cache
            self.df_cache = df

            if not new_rows.empty and self.update_stats:
                self.update_stats[-1]["rows_loaded"] = len(new_rows)

            return new_rows

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def get_stats(self):
        """Get CSV statistics"""
        try:
            if not self.csv_path.exists():
                return {"status": "MISSING", "age": 0, "rows": 0}

            age = time.time() - self.csv_path.stat().st_mtime

            if age < 10:
                status = "ACTIVE"
            elif age < 30:
                status = "SLOW"
            elif age < 60:
                status = "STALE"
            else:
                status = "STALLED"

            return {
                "status": status,
                "age": age,
                "rows": len(self.df_cache),
                "last_update": datetime.fromtimestamp(self.last_modified).strftime(
                    "%H:%M:%S"
                )
                if self.last_modified
                else "Never",
            }

        except Exception as e:
            return {"status": "ERROR", "age": 0, "rows": 0}


# ============================================================================
# DATA PROCESSING
# ============================================================================


def determine_strategy_variant(daily_diff):
    """Determine which strategy to use - using precomputed daily_diff"""
    if DAILY_DIFF_LOWER_LIMIT <= daily_diff <= DAILY_DIFF_UPPER_LIMIT:
        return StrategyVariant.A
    return StrategyVariant.B


def check_entry_setup(row):
    """Check for entry setup conditions - using precomputed values"""
    fibo_236 = row.get("level_236")
    if pd.isna(fibo_236):
        return None

    # Check all other MAs are ≥ Fibo_23.6% using precomputed
    other_mas = ["short002", "short007", "short21", "short50", "long100"]
    for ma in other_mas:
        ma_val = row.get(ma)
        if pd.isna(ma_val) or ma_val < fibo_236:
            return None

    # Check which MA is below Fibo_23.6% using precomputed
    ma_200 = row.get("long200")
    ma_350 = row.get("long350")
    ma_500 = row.get("long500")

    if not pd.isna(ma_200) and ma_200 <= fibo_236:
        return StrategyType.MA_200_WAVE
    elif not pd.isna(ma_350) and ma_350 <= fibo_236:
        return StrategyType.MA_350_WAVE
    elif not pd.isna(ma_500) and ma_500 <= fibo_236:
        return StrategyType.MA_500_WAVE

    return None


def is_entry_condition_met(row, strategy_type):
    """Check if entry condition is fully met - using precomputed values"""
    fibo_236 = row.get("level_236")
    if pd.isna(fibo_236):
        return False

    # All MAs must be ≥ Fibo_23.6% using precomputed
    all_mas = [
        "short002",
        "short007",
        "short21",
        "short50",
        "long100",
        "long200",
        "long350",
        "long500",
    ]

    for ma in all_mas:
        ma_val = row.get(ma)
        if pd.isna(ma_val) or ma_val < fibo_236:
            return False

    return True


# ============================================================================
# ENTRY SIGNAL PROCESSING
# ============================================================================


def process_entry_signal(row, strategy_type):
    """Process entry signal based on trading mode - track specific row trigger"""
    global trade_state

    # Log specific row trigger
    row_time = row.get("Open Time", "Unknown")
    print_header(f"ENTRY SIGNAL DETECTED (Row Time: {row_time})", Colors.MAGENTA)
    print_colored(f"Strategy: {strategy_type.value}", Colors.MAGENTA, "bold")

    # Record signal with row time
    trade_state.stats["entry_signals_detected"] += 1
    trade_state.signal_history.append(
        {
            "time": datetime.now(),
            "row_time": row_time,
            "type": "ENTRY",
            "strategy": strategy_type.value,
            "price": row.get("Close"),
        }
    )

    if trade_state.mode == TradingMode.BUY:
        print_step("1", "BUY MODE: Executing BUY order", Colors.SUCCESS)
        execute_buy_order(row, strategy_type)

    elif trade_state.mode == TradingMode.SELL:
        print_step("1", "SELL MODE: Validating existing position", Colors.WARNING)
        print_colored(
            "   • Entry signal confirms position timing",
            Colors.WARNING,
            "italic",
            indent=2,
        )
        print_colored(
            "   • Skipping BUY (tokens already held)",
            Colors.WARNING,
            "italic",
            indent=2,
        )

        # Validate existing position at this signal
        with state_lock:
            trade_state.position_open = True
            trade_state.position_entered_by_signal = False
            trade_state.virtual_entry_price = row.get("Close", 0)
            trade_state.entry_time = datetime.now()
            trade_state.active_strategy = strategy_type
            trade_state.phase = Phase.EXIT_MONITORING
            trade_state.entry_signal_confirmed = True

            # Get actual token amount from wallet
            if trade_state.wallet_mgr:
                trade_state.position_size = trade_state.wallet_mgr.get_balance(
                    TOKEN, include_locked=False
                )

        print_step(
            "2",
            f"Position validated: {trade_state.position_size:.6f} {TOKEN}",
            Colors.INFO,
        )
        print_colored(
            f"   • Virtual entry price: ${trade_state.virtual_entry_price:.4f}",
            Colors.INFO,
            indent=2,
        )
        print_colored("   • Starting exit monitoring", Colors.INFO, indent=2)

        # Start exit monitoring
        start_exit_monitoring(row)


def execute_buy_order(row, strategy_type):
    """Execute BUY order (only in BUY mode) - using precomputed price"""
    global trade_state

    current_price = row.get("Close")
    if pd.isna(current_price):
        print_colored("Invalid price data", Colors.ERROR, "bold")
        return

    # Calculate token amount
    token_amount, actual_usd = calculate_token_amount(
        TRADE_AMOUNT_USD, current_price, trade_state.safety_mgr
    )

    if token_amount <= 0:
        print_colored("Invalid token calculation", Colors.ERROR, "bold")
        return

    print_step(
        "2", f"Buying {token_amount:.6f} {TOKEN} for ${actual_usd:.2f}", Colors.INFO
    )

    # Execute order
    order_result = trade_state.order_executor.execute_market_order(
        side="BUY", token_amount=token_amount, is_test=TEST_MODE
    )

    if order_result.get("status") in ["SUCCESS", "TEST_SUCCESS"]:
        order = order_result.get("order", {})

        # Update state
        with state_lock:
            trade_state.position_open = True
            trade_state.position_entered_by_signal = True
            trade_state.entry_price = current_price
            trade_state.entry_time = datetime.now()
            trade_state.position_size = token_amount
            trade_state.active_strategy = strategy_type
            trade_state.phase = Phase.EXIT_MONITORING
            trade_state.entry_signal_confirmed = True

            # Reset waiting conditions
            for key in trade_state.waiting_conditions:
                trade_state.waiting_conditions[key] = False

        # Log transaction
        log_binance_transaction(order, "BUY", strategy_type.value)

        # Display success
        display_position_status(
            "OPENED",
            {
                "Tokens": f"{token_amount:.6f} {TOKEN}",
                "Entry Price": f"${current_price:.4f}",
                "Value": f"${actual_usd:.2f}",
                "Strategy": strategy_type.value,
            },
        )

        # Switch to SELL mode
        trade_state.mode = TradingMode.SELL
        print_transition("BUY", "SELL", Colors.INFO)

        # Start exit monitoring
        start_exit_monitoring(row)

    else:
        error_msg = order_result.get("error", "Unknown error")
        print_colored(f"BUY failed: {error_msg}", Colors.ERROR, "bold")


def execute_exit_order(row, exit_reason):
    """Execute SELL order on exit signal - using precomputed price"""
    global trade_state

    if not trade_state.position_open or trade_state.position_size <= 0:
        print_colored("No position to exit", Colors.WARNING)
        return

    current_price = row.get("Close")
    if pd.isna(current_price):
        print_colored("Invalid price data for exit", Colors.ERROR, "bold")
        return

    print_step(
        "EXIT",
        f"Selling {trade_state.position_size:.6f} {TOKEN} at ${current_price:.4f} - Reason: {exit_reason}",
        Colors.ERROR,
    )

    # Calculate P&L
    entry_price = (
        trade_state.entry_price
        if trade_state.position_entered_by_signal
        else trade_state.virtual_entry_price
    )
    entry_value = entry_price * trade_state.position_size
    exit_value = current_price * trade_state.position_size
    pnl_usd = exit_value - entry_value
    pnl_percent = (pnl_usd / entry_value * 100) if entry_value > 0 else 0

    trade_state.stats["total_pnl_usd"] += pnl_usd
    trade_state.stats["total_pnl_percent"] += pnl_percent
    trade_state.stats["exit_signals_detected"] += 1
    trade_state.stats["trades_completed"] += 1

    # Execute order
    order_result = trade_state.order_executor.execute_market_order(
        side="SELL", token_amount=trade_state.position_size, is_test=TEST_MODE
    )

    if order_result.get("status") in ["SUCCESS", "TEST_SUCCESS"]:
        order = order_result.get("order", {})

        # Update state
        with state_lock:
            trade_state.position_open = False
            trade_state.phase = Phase.ENTRY_MONITORING
            trade_state.entry_signal_confirmed = False
            trade_state.active_strategy = None

            # Reset waiting conditions
            for key in trade_state.waiting_conditions:
                trade_state.waiting_conditions[key] = False

        # Log transaction
        log_binance_transaction(
            order,
            "SELL",
            trade_state.active_strategy.value if trade_state.active_strategy else "N/A",
            pnl_percent=pnl_percent,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
        )

        # Display
        display_position_status(
            "CLOSED",
            {
                "Tokens Sold": f"{trade_state.position_size:.6f} {TOKEN}",
                "Exit Price": f"${current_price:.4f}",
                "P&L": f"${pnl_usd:+.2f} ({pnl_percent:+.2f}%)",
                "Reason": exit_reason,
            },
        )

        # Switch to BUY mode
        trade_state.mode = TradingMode.BUY
        print_transition("SELL", "BUY", Colors.INFO)

    else:
        error_msg = order_result.get("error", "Unknown error")
        print_colored(f"SELL failed: {error_msg}", Colors.ERROR, "bold")


# ============================================================================
# EXIT MONITORING - STRATEGY A
# ============================================================================


def start_exit_monitoring(row):
    """Start exit monitoring based on strategy variant - track row time"""
    global trade_state
    with state_lock:
        trade_state.phase = Phase.EXIT_MONITORING
        trade_state.last_processed_time = row.get("Open Time")

    if trade_state.strategy_variant == StrategyVariant.A:
        monitor_strategy_a_exit(row)
    else:
        monitor_strategy_b_exit(row)


def monitor_strategy_a_exit(row):
    """Monitor Strategy A exit conditions - using precomputed"""
    if not trade_state.position_open:
        return

    if trade_state.active_strategy == StrategyType.MA_200_WAVE:
        monitor_ma_200_wave_exit_a(row)
    else:
        monitor_ma_long_wave_exit_a(row)


def monitor_ma_200_wave_exit_a(row):
    """MA_200 Wave exit monitoring for Strategy A - using precomputed"""
    print_header("MA_200 WAVE EXIT MONITORING", Colors.INFO)

    fibo_764 = row.get("level_764")
    ma_100 = row.get("long100")
    ma_500 = row.get("long500")
    ma_350 = row.get("long350")

    if any(pd.isna(x) for x in [fibo_764, ma_100, ma_500, ma_350]):
        return

    # Step 1: Wait for MA_100 & MA_500 ≥ Fibo_76.4%
    ma100_met = ma_100 >= fibo_764
    ma500_met = ma_500 >= fibo_764

    if ma100_met:
        trade_state.waiting_conditions["waiting_ma_100_above_fibo_764"] = True
    if ma500_met:
        trade_state.waiting_conditions["waiting_ma_500_above_fibo_764"] = True

    both_profit_met = (
        trade_state.waiting_conditions["waiting_ma_100_above_fibo_764"]
        and trade_state.waiting_conditions["waiting_ma_500_above_fibo_764"]
    )

    if not both_profit_met:
        print_step("1", "Waiting for profit target", Colors.WARNING)
        print_condition("MA_100 ≥ Fibo_76.4%", ma100_met, indent=4)
        print_condition("MA_500 ≥ Fibo_76.4%", ma500_met, indent=4)
        return

    # Step 2: Wait for MA_350 ≤ MA_500
    ma350_le_500 = ma_350 <= ma_500
    if ma350_le_500:
        trade_state.waiting_conditions["waiting_ma_350_below_ma_500"] = True

    if trade_state.waiting_conditions["waiting_ma_350_below_ma_500"]:
        print_step("2", "Reversal signal met - Executing exit", Colors.SUCCESS)
        execute_exit_order(row, "MA_200_WAVE_TARGET_EXIT")
    else:
        print_step("2", "Profit target reached, waiting for reversal", Colors.SUCCESS)
        print_waiting("MA_350 ≤ MA_500", indent=4)


def monitor_ma_long_wave_exit_a(row):
    """MA_350/500 Wave exit monitoring for Strategy A - Full implementation per flowchart - using precomputed"""
    print_header("MA_LONG WAVE EXIT MONITORING", Colors.INFO)

    fibo_764 = row.get("level_764")
    fibo_236 = row.get("level_236")
    ma_200 = row.get("long200")
    ma_100 = row.get("long100")
    ma_50 = row.get("short50")
    ma_350 = row.get("long350")
    ma_500 = row.get("long500")
    rsi_ma50 = row.get("rsi_ma50")
    lesser_mas = [
        row.get("short002"),
        row.get("short007"),
        row.get("short21"),
        row.get("short50"),
    ]

    if any(pd.isna(x) for x in [fibo_764, fibo_236, ma_200]):
        return

    # Phase 1: Dual monitoring for MA_200 vs Fibo levels
    ma200_ge_764 = ma_200 >= fibo_764
    ma200_le_236 = ma_200 <= fibo_236

    if ma200_le_236:
        # Trigger Stop Loss
        print_signal("STOP LOSS TRIGGER", "MA_200 ≤ Fibo_23.6%")
        display_stop_loss_activation()
        with state_lock:
            trade_state.captured_fibo_0 = row.get("level_000")
            trade_state.phase = Phase.STOP_LOSS_ACTIVE
        trade_state.stats["stop_loss_triggers"] += 1
        return

    if ma200_ge_764:
        # Event A: Proceed to Phase 2
        print_step("PHASE 1", "MA_200 ≥ Fibo_76.4% - Moving to Phase 2", Colors.SUCCESS)
        trade_state.waiting_conditions["waiting_ma_200_above_fibo_764"] = True
    else:
        print_waiting("MA_200 ≥ Fibo_76.4% or ≤ Fibo_23.6%")
        return

    # Phase 2: Check all lesser MAs >= MA_200
    if not trade_state.waiting_conditions["waiting_ma_200_above_fibo_764"]:
        return

    all_lesser_ge_ma200 = all(m >= ma_200 for m in lesser_mas if not pd.isna(m))

    if all_lesser_ge_ma200:
        trade_state.waiting_conditions["waiting_lesser_mas_above_ma_200"] = True
        print_condition("All lesser MAs >= MA_200", True)
        # Proceed to Step 3 Path
        step_3_path_a(row, fibo_764, ma_100, ma_350, ma_500, ma_50, rsi_ma50)
    else:
        print_condition("All lesser MAs >= MA_200", False)
        # Capture Path: Capture Fibo_1 at MA_100 dip
        if pd.notna(ma_100):
            trade_state.captured_fibo_1_dip = (
                ma_100  # Assuming dip capture at current MA_100
            )
            print_step(
                "CAPTURE PATH",
                f"Captured Fibo_1 at MA_100 dip: {trade_state.captured_fibo_1_dip:.4f}",
                Colors.WARNING,
            )
            # Initiate Dual Monitoring for Capture Path
            capture_path_dual_monitoring(row)
        else:
            print_waiting("Valid MA_100 for capture")


def step_3_path_a(row, fibo_764, ma_100, ma_350, ma_500, ma_50, rsi_ma50):
    """Step 3 Path for Strategy A MA_350/500 - using precomputed"""
    # Wait for MA_100 <= MA_500
    ma100_le_500 = ma_100 <= ma_500
    if ma100_le_500:
        trade_state.waiting_conditions["waiting_ma_100_below_ma_500"] = True
        print_condition("MA_100 <= MA_500", True)
    else:
        print_waiting("MA_100 <= MA_500")
        return

    if not trade_state.waiting_conditions["waiting_ma_100_below_ma_500"]:
        return

    # Capture new Fibo_1 at MA_100 dip (use current ma_100 as proxy)
    trade_state.captured_fibo_1 = ma_100
    print_step(
        "CAPTURE NEW",
        f"New Fibo_1 at MA_100: {trade_state.captured_fibo_1:.4f}",
        Colors.INFO,
    )

    # Branch A/B based on MA_500 >= Fibo_76.4%
    ma500_ge_764 = ma_500 >= fibo_764

    if ma500_ge_764:
        # Branch A: Wait for MA_100 >= New Fibo_1
        ma100_ge_new_fibo = ma_100 >= trade_state.captured_fibo_1
        if ma100_ge_new_fibo:
            trade_state.waiting_conditions["waiting_ma_100_above_new_fibo_1"] = True
            print_condition("MA_100 >= New Fibo_1", True)
        else:
            print_waiting("MA_100 >= New Fibo_1")
            return
    else:
        # Branch B: Wait for MA_350 >= New Fibo_1
        ma350_ge_new_fibo = ma_350 >= trade_state.captured_fibo_1
        if ma350_ge_new_fibo:
            trade_state.waiting_conditions["waiting_ma_350_above_new_fibo_1"] = True
            print_condition("MA_350 >= New Fibo_1", True)
        else:
            print_waiting("MA_350 >= New Fibo_1")
            return

    # Common waits after branch
    if (
        trade_state.waiting_conditions["waiting_ma_100_above_new_fibo_1"]
        or trade_state.waiting_conditions["waiting_ma_350_above_new_fibo_1"]
    ):
        # Wait for MA_50 <= Fibo_76.4%
        ma50_le_764 = ma_50 <= fibo_764
        if ma50_le_764:
            trade_state.waiting_conditions["waiting_ma_50_below_fibo_764"] = True
            print_condition("MA_50 <= Fibo_76.4%", True)
        else:
            print_waiting("MA_50 <= Fibo_76.4%")
            return

        if trade_state.waiting_conditions["waiting_ma_50_below_fibo_764"]:
            # Wait for RSI sequence: >=55 then <=52 using precomputed
            rsi_ge_55 = rsi_ma50 >= 55
            if (
                rsi_ge_55
                and not trade_state.waiting_conditions["waiting_rsi_ma50_above_55"]
            ):
                trade_state.waiting_conditions["waiting_rsi_ma50_above_55"] = True
                print_condition("RSI_MA50 >= 55", True)
                print_waiting("RSI_MA50 <= 52")
                return

            if (
                trade_state.waiting_conditions["waiting_rsi_ma50_above_55"]
                and rsi_ma50 <= 52
            ):
                trade_state.waiting_conditions["waiting_rsi_ma50_below_52"] = True
                print_condition("RSI_MA50 <= 52", True)
                execute_exit_order(row, "STRATEGY_A_STEP_3_EXIT")
            elif trade_state.waiting_conditions["waiting_rsi_ma50_above_55"]:
                print_waiting("RSI_MA50 <= 52")
            else:
                print_waiting("RSI_MA50 >= 55")


def capture_path_dual_monitoring(row):
    """Dual monitoring for Capture Path in Strategy A - using precomputed"""
    captured_fibo_1 = trade_state.captured_fibo_1_dip
    if pd.isna(captured_fibo_1):
        return

    ma_100 = row.get("long100")
    ma_200 = row.get("long200")
    fibo_236 = row.get("level_236")

    if pd.isna(ma_100) or pd.isna(ma_200) or pd.isna(fibo_236):
        return

    # Condition 1: MA_100 >= Captured Fibo_1 (met first -> sell)
    cond1_met = ma_100 >= captured_fibo_1

    # Condition 2: MA_200 <= Fibo_236 (met first -> stop loss)
    cond2_met = ma_200 <= fibo_236

    if cond1_met and not cond2_met:
        # Condition 1 met first
        print_signal("CAPTURE PATH EXIT", "MA_100 >= Captured Fibo_1")
        execute_exit_order(row, "CAPTURE_PATH_MA100_EXIT")
    elif cond2_met:
        # Condition 2 met (stop loss)
        print_signal("STOP LOSS TRIGGER", "MA_200 <= Fibo_236 in Capture Path")
        display_stop_loss_activation()
        with state_lock:
            trade_state.captured_fibo_0 = row.get("level_000")
            trade_state.phase = Phase.STOP_LOSS_ACTIVE
        trade_state.stats["stop_loss_triggers"] += 1
    else:
        print_waiting("MA_100 >= Captured Fibo_1 OR MA_200 <= Fibo_236")


# ============================================================================
# EXIT MONITORING - STRATEGY B
# ============================================================================


def monitor_strategy_b_exit(row):
    """Strategy B exit monitoring - Full implementation per flowchart - using precomputed"""
    print_header("STRATEGY B ENHANCED EXIT MONITORING", Colors.ERROR)

    fibo_764 = row.get("level_764")
    ma_100 = row.get("long100")
    ma_500 = row.get("long500")
    ma_350 = row.get("long350")
    fibo_1 = row.get("level_100")  # Fibo_1 level
    ma_200 = row.get("long200")
    fibo_236 = row.get("level_236")

    if any(pd.isna(x) for x in [fibo_764, ma_100, ma_500, ma_350]):
        return

    # Step 1: Profit Target - BOTH MA_100 >= Fibo_764 AND MA_500 >= Fibo_764
    ma100_ge_764 = ma_100 >= fibo_764
    ma500_ge_764 = ma_500 >= fibo_764
    profit_target_met = ma100_ge_764 and ma500_ge_764

    if profit_target_met:
        trade_state.waiting_conditions["waiting_strategy_b_profit_target"] = True
        print_condition("MA_100 >= Fibo_76.4% AND MA_500 >= Fibo_76.4%", True)
    else:
        print_step("1", "Waiting for profit target", Colors.WARNING)
        print_waiting("BOTH MA_100 & MA_500 >= Fibo_76.4%")
        return

    if not trade_state.waiting_conditions["waiting_strategy_b_profit_target"]:
        return

    # Step 2: Reversal Signal - MA_350 <= MA_500
    reversal_met = ma_350 <= ma_500
    if reversal_met:
        trade_state.waiting_conditions["waiting_strategy_b_reversal"] = True
        print_step("2", "Reversal signal: MA_350 <= MA_500", Colors.WARNING)
    else:
        print_step("2", "Profit target reached, waiting for reversal", Colors.INFO)
        print_waiting("MA_350 <= MA_500")
        return

    if not trade_state.waiting_conditions["waiting_strategy_b_reversal"]:
        return

    # Step 3: Capture Fibo_1 level (level_100)
    trade_state.captured_fibo_1 = fibo_1
    print_step("3", f"Captured Fibo_1: {trade_state.captured_fibo_1:.4f}", Colors.INFO)

    # Step 4: Dual Monitoring
    if (
        pd.isna(trade_state.captured_fibo_1)
        or pd.isna(ma_200)
        or pd.isna(fibo_236)
        or pd.isna(ma_350)
    ):
        return

    # Condition 1: MA_200 <= Fibo_236 (stop loss)
    cond1_met = ma_200 <= fibo_236

    # Condition 2: MA_350 >= Captured Fibo_1 (sell)
    cond2_met = ma_350 >= trade_state.captured_fibo_1

    if cond1_met and not cond2_met:
        # Condition 1 met first - Trigger Stop Loss
        print_signal("STRATEGY B STOP LOSS", "MA_200 <= Fibo_236")
        display_stop_loss_activation()
        with state_lock:
            trade_state.captured_fibo_0 = row.get("level_000")
            trade_state.phase = Phase.STOP_LOSS_ACTIVE
        trade_state.stats["stop_loss_triggers"] += 1
    elif cond2_met:
        # Condition 2 met - Execute Sell
        print_signal("STRATEGY B ENHANCED EXIT", "MA_350 >= Captured Fibo_1")
        execute_exit_order(row, "STRATEGY_B_ENHANCED_EXIT")
    else:
        print_step("4", "Dual monitoring active", Colors.WARNING)
        print_waiting("MA_200 <= Fibo_236 OR MA_350 >= Captured Fibo_1")


# ============================================================================
# STOP LOSS PROCESSING
# ============================================================================


def check_stop_loss_conditions(row):
    """Check stop loss conditions - Shared across strategies - using precomputed"""
    global trade_state

    print_header("STOP LOSS FLOW", Colors.ERROR)

    fibo_0 = trade_state.captured_fibo_0
    if pd.isna(fibo_0):
        print_colored("No captured Fibo_0 - Cannot proceed", Colors.ERROR)
        return

    ma_14 = row.get(
        "short21", row.get("short007", 0)
    )  # Assuming short21 as proxy for MA_14 if not present
    ma_100 = row.get("long100")
    ma_200 = row.get("long200")
    ma_500 = row.get("long500")
    rsi_ma50 = row.get("rsi_ma50")

    if any(pd.isna(x) for x in [ma_14, ma_100, ma_200, ma_500]):
        return

    # Check two conditions
    cond1 = ma_14 <= fibo_0  # Condition 1
    cond2 = ma_100 >= ma_500  # Condition 2

    if cond1:
        # Path 1: Wait for MA_100 >= MA_200 -> Exit
        ma100_ge_200 = ma_100 >= ma_200
        if ma100_ge_200:
            trade_state.waiting_conditions["waiting_ma_100_above_ma_200"] = True
        if trade_state.waiting_conditions["waiting_ma_100_above_ma_200"]:
            print_condition("MA_14 <= Fibo_0 -> MA_100 >= MA_200", True)
            execute_exit_order(row, "STOP_LOSS_PATH_1")
        else:
            print_waiting("MA_100 >= MA_200 (Path 1)")
            return
    elif cond2:
        # Path 2: Wait for MA_100 <= MA_500 AND MA_200 <= MA_500
        ma100_le_500 = ma_100 <= ma_500
        ma200_le_500 = ma_200 <= ma_500
        both_le_500 = ma100_le_500 and ma200_le_500

        if both_le_500:
            trade_state.waiting_conditions["waiting_ma_100_below_ma_500"] = True
            trade_state.waiting_conditions["waiting_ma_200_below_ma_500"] = True
            print_condition("MA_100 <= MA_500 AND MA_200 <= MA_500", True)
        else:
            print_waiting("MA_100 <= MA_500 AND MA_200 <= MA_500 (Path 2)")
            return

        if (
            trade_state.waiting_conditions["waiting_ma_100_below_ma_500"]
            and trade_state.waiting_conditions["waiting_ma_200_below_ma_500"]
        ):
            # Wait for RSI sequence: >=53 then <=51 using precomputed
            rsi_ge_53 = rsi_ma50 >= 53
            if (
                rsi_ge_53
                and not trade_state.waiting_conditions["waiting_rsi_ma50_above_53"]
            ):
                trade_state.waiting_conditions["waiting_rsi_ma50_above_53"] = True
                print_condition("RSI_MA50 >= 53", True)
                print_waiting("RSI_MA50 <= 51")
                return

            if (
                trade_state.waiting_conditions["waiting_rsi_ma50_above_53"]
                and rsi_ma50 <= 51
            ):
                trade_state.waiting_conditions["waiting_rsi_ma50_below_51"] = True
                print_condition("RSI_MA50 <= 51", True)
                execute_exit_order(row, "STOP_LOSS_PATH_2")
            elif trade_state.waiting_conditions["waiting_rsi_ma50_above_53"]:
                print_waiting("RSI_MA50 <= 51")
    else:
        print_waiting("Condition 1 (MA_14 <= Fibo_0) OR Condition 2 (MA_100 >= MA_500)")


# ============================================================================
# ORDER EXECUTOR
# ============================================================================


class OrderExecutor:
    """Executes orders safely"""

    def __init__(self, client, safety_mgr, wallet_mgr):
        self.client = client
        self.safety_mgr = safety_mgr
        self.wallet_mgr = wallet_mgr

    def execute_market_order(self, side, token_amount, is_test=False):
        """Execute market order"""
        if not self.safety_mgr.can_place_order():
            return {"status": "COOLDOWN", "error": "Order cooldown active"}

        print_colored(f"\n{'='*60}", Colors.INFO)
        print_colored(
            f"🎯 {'TEST' if is_test else 'LIVE'} {side} ORDER",
            Colors.SUCCESS if side == "BUY" else Colors.ERROR,
            "bold",
        )
        print_colored(f"📊 Amount: {token_amount:.6f} {TOKEN}", Colors.INFO)
        print_colored(f"{'='*60}", Colors.INFO)

        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=SYMBOL)
            current_price = float(ticker["price"])

            if is_test:
                return self._execute_test_order(side, token_amount, current_price)
            else:
                return self._execute_real_order(side, token_amount)

        except Exception as e:
            error_msg = f"Order error: {e}"
            print_colored(error_msg, Colors.ERROR, "bold")
            return {"status": "ERROR", "error": str(e)}
        finally:
            self.safety_mgr.update_last_order_time()

    def _execute_real_order(self, side, token_amount):
        """Execute real Binance order"""
        try:
            # Format quantity
            quantity_str = format(token_amount, "f").rstrip("0").rstrip(".")

            if side == "BUY":
                order = self.client.order_market_buy(
                    symbol=SYMBOL,
                    quoteOrderQty=str(
                        TRADE_AMOUNT_USD
                    ),  # Use quote for BUY if needed, but quantity for consistency
                )
            else:
                order = self.client.order_market_sell(
                    symbol=SYMBOL, quantity=quantity_str
                )

            print_colored("✅ Order executed successfully", Colors.SUCCESS, "bold")
            self._display_order_summary(order)

            return {"status": "SUCCESS", "order": order}

        except BinanceAPIException as e:
            error_msg = f"Binance API error: {e.code} - {e.message}"
            print_colored(error_msg, Colors.ERROR, "bold")
            return {"status": "API_ERROR", "error": error_msg}

        except Exception as e:
            error_msg = f"Order execution failed: {e}"
            print_colored(error_msg, Colors.ERROR, "bold")
            return {"status": "ERROR", "error": error_msg}

    def _execute_test_order(self, side, token_amount, current_price):
        """Execute test order"""
        print_colored("📋 TEST MODE - Simulating order", Colors.WARNING, "bold")

        simulated_order = {
            "orderId": int(time.time() * 1000),
            "symbol": SYMBOL,
            "side": side.upper(),
            "type": "MARKET",
            "origQty": str(token_amount),
            "executedQty": str(token_amount),
            "cummulativeQuoteQty": str(token_amount * current_price),
            "status": "FILLED",
            "fills": [
                {
                    "price": str(current_price),
                    "qty": str(token_amount),
                    "commission": "0.00000000",
                    "commissionAsset": PAIR if side == "BUY" else TOKEN,
                }
            ],
            "transactTime": int(time.time() * 1000),
        }

        self._display_order_summary(simulated_order)

        return {"status": "TEST_SUCCESS", "order": simulated_order}

    def _display_order_summary(self, order):
        """Display order summary"""
        table = Table(show_header=False, box=None)
        table.add_column(style=Colors.INFO, width=20)
        table.add_column(style=Colors.WHITE)

        table.add_row("Order ID", str(order.get("orderId", "TEST")))
        table.add_row("Symbol", order.get("symbol", SYMBOL))
        table.add_row("Side", order.get("side", "N/A"))
        table.add_row("Quantity", f"{float(order.get('origQty', 0)):.6f}")
        table.add_row("Status", order.get("status", "FILLED"))

        fills = order.get("fills", [])
        if fills:
            avg_price = sum(float(f["price"]) for f in fills) / len(fills)
            table.add_row("Avg Price", f"${avg_price:.8f}")

        console.print(Panel(table, title="📊 ORDER SUMMARY", border_style=Colors.INFO))


# ============================================================================
# TRANSACTION LOGGING
# ============================================================================


def log_binance_transaction(order, action, strategy, **kwargs):
    """Log transaction to CSV"""
    try:
        # Extract order data
        transaction = {
            "timestamp": datetime.now().isoformat(),
            "order_id": order.get("orderId", ""),
            "client_order_id": order.get("clientOrderId", ""),
            "symbol": order.get("symbol", SYMBOL),
            "side": order.get("side", ""),
            "type": order.get("type", ""),
            "quantity": float(order.get("origQty", 0)),
            "executed_quantity": float(order.get("executedQty", 0)),
            "cumulative_quote_qty": float(order.get("cummulativeQuoteQty", 0)),
            "status": order.get("status", ""),
            "time_in_force": order.get("timeInForce", ""),
            "transact_time": order.get("transactTime", ""),
            "action": action,
            "strategy": strategy,
        }

        # Add fills
        fills = order.get("fills", [])
        if fills:
            total_commission = 0
            weighted_price = 0
            total_qty = 0

            for fill in fills:
                price = float(fill.get("price", 0))
                qty = float(fill.get("qty", 0))
                commission = float(fill.get("commission", 0))

                total_commission += commission
                weighted_price += price * qty
                total_qty += qty

            transaction["average_price"] = (
                weighted_price / total_qty if total_qty > 0 else 0
            )
            transaction["total_commission"] = total_commission
            transaction["commission_asset"] = fills[0].get("commissionAsset", "")

        # Add P&L if available
        if "pnl_percent" in kwargs:
            transaction["pnl_percent"] = kwargs["pnl_percent"]
            transaction["pnl_usd"] = kwargs["pnl_usd"]

        if "exit_reason" in kwargs:
            transaction["exit_reason"] = kwargs["exit_reason"]

        # Write to CSV
        file_exists = os.path.exists(TRANSACTIONS_CSV)

        with open(TRANSACTIONS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=transaction.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(transaction)

        # Display summary
        display_transaction_summary(transaction)

        logger.info(f"Transaction logged: {action} {strategy}")

    except Exception as e:
        logger.error(f"Transaction logging error: {e}")


def display_transaction_summary(transaction):
    """Display transaction summary"""
    action = transaction.get("action", "N/A")
    strategy = transaction.get("strategy", "N/A")

    table = Table(show_header=True, header_style="bold magenta", box=SIMPLE)
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Timestamp", transaction.get("timestamp", "N/A"))
    table.add_row("Order ID", str(transaction.get("order_id", "N/A")))
    table.add_row("Action", f"[bold]{action}[/bold]")
    table.add_row("Strategy", strategy)
    table.add_row("Side", transaction.get("side", "N/A"))
    table.add_row("Quantity", f"{transaction.get('quantity', 0):.6f}")

    if "average_price" in transaction:
        table.add_row("Avg Price", f"${transaction['average_price']:.8f}")

    if "pnl_percent" in transaction:
        pnl_color = "green" if transaction["pnl_percent"] > 0 else "red"
        table.add_row(
            "P&L %", f"[{pnl_color}]{transaction['pnl_percent']:+.2f}%[/{pnl_color}]"
        )
        table.add_row(
            "P&L $", f"[{pnl_color}]${transaction['pnl_usd']:+.2f}[/{pnl_color}]"
        )

    if "exit_reason" in transaction:
        table.add_row("Exit Reason", transaction["exit_reason"])

    panel_title = "💾 TRANSACTION LOGGED"
    if TEST_MODE:
        panel_title += " [TEST]"

    console.print(Panel(table, title=panel_title, border_style="cyan"))


# ============================================================================
# STATUS DISPLAY
# ============================================================================


def display_status(csv_monitor):
    """Display comprehensive status"""
    # Get CSV stats
    csv_stats = csv_monitor.get_stats()

    # Get current price
    try:
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        current_price = float(ticker["price"])
    except:
        current_price = 0.0

    # Create main table
    main_table = Table(show_header=False, box=None, show_lines=False)
    main_table.add_column(style="cyan", width=25)
    main_table.add_column(style="white")

    main_table.add_row("Time", datetime.now().strftime("%H:%M:%S"))
    main_table.add_row("Symbol", SYMBOL)
    main_table.add_row("Price", f"${current_price:.4f}")

    # CSV status
    csv_color = {
        "ACTIVE": "green",
        "SLOW": "yellow",
        "STALE": "red",
        "STALLED": "red",
        "ERROR": "red",
    }.get(csv_stats["status"], "white")

    main_table.add_row(
        "CSV Status",
        f"[{csv_color}]{csv_stats['status']}[/{csv_color}] "
        f"({csv_stats['age']:.1f}s)",
    )

    if csv_stats["rows"] > 0:
        main_table.add_row("CSV Rows", f"{csv_stats['rows']}")

    # Trading mode
    mode_color = Colors.get_mode_color(trade_state.mode)
    main_table.add_row(
        "Trading Mode",
        f"[bold {mode_color}]{trade_state.mode.value}[/bold {mode_color}]",
    )

    # Strategy
    if trade_state.strategy_variant:
        strat_color = Colors.get_strategy_color(trade_state.strategy_variant)
        main_table.add_row(
            "Strategy",
            f"[{strat_color}]{trade_state.strategy_variant.value}[/{strat_color}]",
        )

    # Phase
    phase_color = {
        "ENTRY_MONITORING": "cyan",
        "ENTRY_SIGNAL_CONFIRMED": "yellow",
        "POSITION_OPEN": "green",
        "EXIT_MONITORING": "magenta",
        "STOP_LOSS_ACTIVE": "red",
    }.get(trade_state.phase.value, "white")

    main_table.add_row(
        "Phase", f"[{phase_color}]{trade_state.phase.value}[/{phase_color}]"
    )

    # Daily diff
    daily_color = "green" if trade_state.current_daily_diff >= 0 else "red"
    main_table.add_row(
        "Daily Diff",
        f"[{daily_color}]{trade_state.current_daily_diff:+.2f}%[/{daily_color}]",
    )

    # Position info
    if trade_state.position_open:
        main_table.add_row("Position", "[bold green]OPEN[/bold green]")

        entry_price = (
            trade_state.entry_price
            if trade_state.position_entered_by_signal
            else trade_state.virtual_entry_price
        )
        main_table.add_row("Entry Price", f"${entry_price:.4f}")
        main_table.add_row("Position Size", f"{trade_state.position_size:.6f} {TOKEN}")

        # Calculate P&L
        position_value = current_price * trade_state.position_size
        entry_value = entry_price * trade_state.position_size
        pnl_usd = position_value - entry_value
        pnl_percent = (pnl_usd / entry_value) * 100 if entry_value > 0 else 0

        pnl_color = "green" if pnl_usd >= 0 else "red"
        main_table.add_row(
            "Current P&L",
            f"[{pnl_color}]${pnl_usd:+.2f} ({pnl_percent:+.2f}%)[/{pnl_color}]",
        )

        if trade_state.active_strategy:
            main_table.add_row("Active Strategy", trade_state.active_strategy.value)

    else:
        main_table.add_row("Position", "[yellow]CLOSED[/yellow]")

    # Stats
    main_table.add_row(
        "Entry Signals", f"{trade_state.stats['entry_signals_detected']}"
    )
    main_table.add_row("Exit Signals", f"{trade_state.stats['exit_signals_detected']}")

    # Active waiting conditions
    active_waits = [
        k.replace("waiting_", "").replace("_", " ").title()
        for k, v in trade_state.waiting_conditions.items()
        if v
    ]
    if active_waits:
        main_table.add_row(
            "Active Waits", f"[yellow]{', '.join(active_waits[:2])}...[/yellow]"
        )

    # Last processed row time
    if trade_state.last_processed_time:
        main_table.add_row("Last Row Time", str(trade_state.last_processed_time))

    # Display panel
    title = "📊 TRADING STATUS"
    if TEST_MODE:
        title += " [TEST MODE]"

    console.print(Panel(main_table, title=title, border_style="blue"))


def update_trading_mode():
    """Update trading mode based on wallet"""
    if trade_state.wallet_mgr:
        trade_state.mode = trade_state.wallet_mgr.determine_mode()

        wallet_info = {}
        if trade_state.mode == TradingMode.BUY:
            for stable in ["FDUSD", "USDT", "USDC"]:
                bal = trade_state.wallet_mgr.get_balance(stable)
                if bal > 0:
                    wallet_info[stable] = f"${bal:.2f}"
        elif trade_state.mode == TradingMode.SELL:
            sol_value = trade_state.wallet_mgr.get_token_value_usd("SOL")
            wallet_info["SOL"] = f"${sol_value:.2f}"

        display_mode_banner(trade_state.mode, wallet_info)


# ============================================================================
# MAIN PROCESSING
# ============================================================================


def process_single_row(row):
    """Process a single CSV row - no calculations, use precomputed; track specific row"""
    global trade_state

    # Update daily diff - parse precomputed string value
    daily_diff_str = row.get("daily_diff", "0")
    daily_diff = calculate_daily_diff(daily_diff_str)
    trade_state.current_daily_diff = daily_diff

    # Determine strategy variant
    new_variant = determine_strategy_variant(daily_diff)
    if trade_state.strategy_variant != new_variant:
        trade_state.strategy_variant = new_variant
        display_strategy_activation(new_variant, daily_diff)

    # Check for entry setup
    entry_setup = check_entry_setup(row)
    if entry_setup:
        print_signal("ENTRY SETUP", f"{entry_setup.value} detected", Colors.MAGENTA)

        # Check if entry condition met
        if is_entry_condition_met(row, entry_setup):
            print_signal("ENTRY CONFIRMED", "All conditions satisfied", Colors.SUCCESS)
            process_entry_signal(row, entry_setup)
        else:
            # Set waiting condition
            waiting_key = f"waiting_{entry_setup.value.lower().replace('_wave', '')}_above_fibo_236"
            if waiting_key in trade_state.waiting_conditions:
                trade_state.waiting_conditions[waiting_key] = True
                print_waiting(f"{entry_setup.value.replace('_WAVE', '')} >= Fibo_23.6%")

    # Check waiting conditions for entry
    check_waiting_ma_conditions(row)

    # If position open and signal confirmed, monitor exit
    if trade_state.position_open and trade_state.entry_signal_confirmed:
        if trade_state.phase == Phase.STOP_LOSS_ACTIVE:
            check_stop_loss_conditions(row)
        elif trade_state.strategy_variant == StrategyVariant.A:
            monitor_strategy_a_exit(row)
        else:
            monitor_strategy_b_exit(row)


def check_waiting_ma_conditions(row):
    """Check waiting MA conditions for entry - using precomputed"""
    fibo_236 = row.get("level_236")
    if pd.isna(fibo_236):
        return

    # Check each waiting condition
    if trade_state.waiting_conditions["waiting_ma_200_above_fibo_236"]:
        ma_200 = row.get("long200")
        if not pd.isna(ma_200) and ma_200 >= fibo_236:
            trade_state.waiting_conditions["waiting_ma_200_above_fibo_236"] = False
            print_signal("WAITING MET", "MA_200 ≥ Fibo_23.6%", Colors.SUCCESS)

            if is_entry_condition_met(row, StrategyType.MA_200_WAVE):
                process_entry_signal(row, StrategyType.MA_200_WAVE)

    if trade_state.waiting_conditions["waiting_ma_350_above_fibo_236"]:
        ma_350 = row.get("long350")
        if not pd.isna(ma_350) and ma_350 >= fibo_236:
            trade_state.waiting_conditions["waiting_ma_350_above_fibo_236"] = False
            print_signal("WAITING MET", "MA_350 ≥ Fibo_23.6%", Colors.SUCCESS)

            if is_entry_condition_met(row, StrategyType.MA_350_WAVE):
                process_entry_signal(row, StrategyType.MA_350_WAVE)

    if trade_state.waiting_conditions["waiting_ma_500_above_fibo_236"]:
        ma_500 = row.get("long500")
        if not pd.isna(ma_500) and ma_500 >= fibo_236:
            trade_state.waiting_conditions["waiting_ma_500_above_fibo_236"] = False
            print_signal("WAITING MET", "MA_500 ≥ Fibo_23.6%", Colors.SUCCESS)

            if is_entry_condition_met(row, StrategyType.MA_500_WAVE):
                process_entry_signal(row, StrategyType.MA_500_WAVE)


# ============================================================================
# MAIN LOOP
# ============================================================================


def signal_handler(sig, frame):
    """Handle shutdown signals - using os for graceful exit"""
    global running
    running = False
    print_colored("\n🛑 Shutdown signal received", Colors.WARNING, "bold")
    print_colored("💾 Saving state...", Colors.INFO)
    save_state()
    print_colored("👋 Bot stopped gracefully", Colors.SUCCESS, "bold")
    os._exit(0)  # Use os for immediate graceful exit


signal.signal(signal.SIGINT, signal_handler)


def save_state():
    """Save trading state to file"""
    try:
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": trade_state.mode.value,
            "strategy_variant": trade_state.strategy_variant.value
            if trade_state.strategy_variant
            else None,
            "phase": trade_state.phase.value,
            "position_open": trade_state.position_open,
            "position_size": trade_state.position_size,
            "entry_price": trade_state.entry_price,
            "virtual_entry_price": trade_state.virtual_entry_price,
            "active_strategy": trade_state.active_strategy.value
            if trade_state.active_strategy
            else None,
            "entry_signal_confirmed": trade_state.entry_signal_confirmed,
            "stats": trade_state.stats,
            "captured_levels": {
                "fibo_0": trade_state.captured_fibo_0,
                "fibo_1": trade_state.captured_fibo_1,
                "fibo_1_dip": trade_state.captured_fibo_1_dip,
            },
            "waiting_conditions": trade_state.waiting_conditions,
            "last_processed_time": str(trade_state.last_processed_time)
            if trade_state.last_processed_time
            else None,
        }

        with open(STATE_FILE, "w") as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"State saved to {STATE_FILE}")

    except Exception as e:
        logger.error(f"Error saving state: {e}")


def initialize_client():
    """Initialize Binance client"""
    try:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            print_colored(
                "Binance API keys not found in environment", Colors.ERROR, "bold"
            )
            print_colored("Set them with:", Colors.WARNING)
            print_colored("  export BINANCE_API_KEY='your_key'", Colors.INFO)
            print_colored("  export BINANCE_API_SECRET='your_secret'", Colors.INFO)
            return None

        client = Client(api_key, api_secret)
        client.ping()  # Test connection

        print_colored("✅ Binance connection successful", Colors.SUCCESS, "bold")
        return client

    except Exception as e:
        print_colored(f"❌ Binance connection failed: {e}", Colors.ERROR, "bold")
        return None


def display_startup():
    """Display startup banner"""
    banner = f"""
    🚀 ADVANCED TRADING BOT v4.0
    {'═'*50}
    📈 Symbol: {SYMBOL}
    💰 Trade Amount: ${TRADE_AMOUNT_USD}
    📊 Strategy A: {DAILY_DIFF_LOWER_LIMIT}% to {DAILY_DIFF_UPPER_LIMIT}%
    📊 Strategy B: ≤ {DAILY_DIFF_LOWER_LIMIT}%
    🔒 Test Mode: {'✅ YES' if TEST_MODE else '❌ NO (REAL TRADING)'}
    📁 Data: {INDICATORS_CSV}
    💾 Logs: {TRANSACTIONS_CSV}
    {'═'*50}
    """

    console.print(
        Panel.fit(
            banner,
            title="[bold cyan]TRADING BOT - FAITHFUL STRATEGY IMPLEMENTATION[/bold cyan]",
            border_style="red" if not TEST_MODE else "cyan",
        )
    )


def main():
    """Main trading loop"""
    global trade_state, client, running

    # Display startup
    display_startup()

    # Check files
    if not os.path.exists(INDICATORS_CSV):
        display_dino(f"File not found: {INDICATORS_CSV}", "sad", "FILE ERROR")
        print_colored(
            f"Please ensure {INDICATORS_CSV} exists with indicator data",
            Colors.ERROR,
            "bold",
        )
        return

    # Initialize Binance
    client = initialize_client()
    if not client and not TEST_MODE:
        return

    # Initialize CSV monitor
    csv_monitor = CSVMonitor(INDICATORS_CSV)
    trade_state.csv_monitor = csv_monitor

    # Initialize managers
    wallet_mgr = WalletManager(client) if client else None
    trade_state.wallet_mgr = wallet_mgr

    safety_mgr = SafetyManager(client) if client else SafetyManager(None)
    safety_mgr.get_symbol_info()
    trade_state.safety_mgr = safety_mgr

    order_executor = OrderExecutor(client, safety_mgr, wallet_mgr) if client else None
    trade_state.order_executor = order_executor

    # Determine initial mode
    if wallet_mgr:
        trade_state.mode = wallet_mgr.determine_mode()
    else:
        trade_state.mode = TradingMode.NEUTRAL

    if trade_state.mode == TradingMode.NEUTRAL:
        display_dino(
            "Insufficient funds for trading (or no client in test mode)",
            "sad",
            "WALLET STATUS",
        )
        print_colored(
            f"Need ≥${TRADE_AMOUNT_USD} in stablecoins or ≥${MIN_TOKEN_VALUE_FOR_SELL} in SOL",
            Colors.WARNING,
        )
        if TEST_MODE:
            print_colored(
                "Proceeding in TEST MODE with simulated balances", Colors.INFO
            )
            trade_state.mode = TradingMode.BUY  # Default for test

    # Display mode
    wallet_info = {}
    if trade_state.mode == TradingMode.BUY:
        wallet_info["Simulated USD"] = f"${TRADE_AMOUNT_USD}"
    else:
        wallet_info["Simulated SOL"] = f"${MIN_TOKEN_VALUE_FOR_SELL}"

    display_mode_banner(trade_state.mode, wallet_info)

    print_colored("\n🔄 Starting main trading loop...", Colors.INFO, "bold")
    print_colored("   • CSV monitoring every 2 seconds", Colors.INFO)
    print_colored("   • Status updates every 10 seconds", Colors.INFO)
    print_colored(
        "   • Processing rows sequentially for specific triggers", Colors.INFO
    )

    # Main loop
    last_status = 0
    consecutive_no_updates = 0

    try:
        while running:
            current_time = time.time()

            # Check CSV for updates
            updated, status = csv_monitor.check_update()

            if updated:
                consecutive_no_updates = 0

                # Load and process new rows sequentially
                new_rows = csv_monitor.load_new_rows()

                if not new_rows.empty:
                    print_colored(
                        f"\n📥 Processing {len(new_rows)} new row(s)...",
                        Colors.INFO,
                        "bold",
                    )

                    for idx, row_series in new_rows.iterrows():
                        try:
                            row = row_series.to_dict()

                            # Skip if already processed (track by time/index)
                            row_id = row.get("Open Time", idx)
                            row_str = (
                                str(row_id) if hasattr(row_id, "__str__") else str(idx)
                            )
                            if row_str in csv_monitor.processed_rows:
                                if VERBOSE_LOGGING:
                                    print_colored(
                                        f"   Skipping processed row {row_str}",
                                        Colors.GRAY,
                                        indent=2,
                                    )
                                continue

                            # Mark as processed
                            csv_monitor.processed_rows.add(row_str)

                            if VERBOSE_LOGGING:
                                price = row.get("Close", 0)
                                daily_diff_str = row.get("daily_diff", "0")
                                daily_diff = calculate_daily_diff(daily_diff_str)
                                print_colored(
                                    f"   Row {row_str}: ${price:.4f}, Daily: {daily_diff:+.2f}%",
                                    Colors.GRAY,
                                    indent=2,
                                )

                            process_single_row(row)

                        except Exception as e:
                            logger.error(f"Row processing error: {e}")
                            print_colored(f"Row error: {e}", Colors.ERROR, indent=2)

                    if VERBOSE_LOGGING:
                        print_colored("   Batch complete", Colors.SUCCESS, indent=2)

            else:
                consecutive_no_updates += 1

                # Warn if CSV is stale
                if consecutive_no_updates >= 15:  # 30 seconds
                    stats = csv_monitor.get_stats()
                    if stats["status"] in ["STALE", "STALLED"]:
                        print_colored(
                            f"⚠️ CSV {stats['status'].lower()} ({stats['age']:.0f}s)",
                            Colors.WARNING,
                        )

            # Update trading mode periodically
            if current_time % 60 < 1:  # Every minute
                update_trading_mode()

            # Display status periodically
            if current_time - last_status >= STATUS_UPDATE_INTERVAL:
                print_colored("\n" + "=" * 80, Colors.BLUE)
                display_status(csv_monitor)
                print_colored("=" * 80 + "\n", Colors.BLUE)
                last_status = current_time

            # Sleep
            time.sleep(CSV_UPDATE_CHECK_INTERVAL)

    except KeyboardInterrupt:
        print_colored("\n🛑 Manual interruption", Colors.WARNING, "bold")
    except Exception as e:
        print_colored(f"\n❌ Critical error: {e}", Colors.ERROR, "bold")
        logger.exception("Main loop error")
    finally:
        save_state()
        display_dino("Trading bot stopped", "sleep", "SHUTDOWN COMPLETE")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check API keys for real trading
    if not TEST_MODE:
        if not os.environ.get("BINANCE_API_KEY") or not os.environ.get(
            "BINANCE_API_SECRET"
        ):
            display_dino(
                "API KEYS REQUIRED FOR REAL TRADING", "trex_roar", "CONFIGURATION ERROR"
            )
            print_colored(
                "Set environment variables or enable TEST_MODE", Colors.ERROR, "bold"
            )
            os._exit(1)

    try:
        main()
    except Exception as e:
        display_dino(f"Fatal error: {str(e)[:100]}", "dead", "FATAL ERROR")
        logger.exception("Fatal error")
        os._exit(1)
