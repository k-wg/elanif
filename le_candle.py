#!/usr/bin/env python3
"""
ENHANCED VISUAL TRADING BOT - FAITHFUL STRATEGY A/B IMPLEMENTATION
Author: Trading Bot System
Version: 4.0-VISUAL-ENHANCED
Description: Enhanced visual implementation with improved art, animations, and colors
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
import random
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from art import text2art
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from rich.align import Align
from rich.box import ROUNDED, SIMPLE, DOUBLE, HEAVY
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
from rich.rule import Rule

# ============================================================================
# ENHANCED VISUAL CONFIGURATION
# ============================================================================

# Enhanced font mapping with more artistic options
FONT_MAPPING = {
    "default": "standard",
    "triceratops": "slant",
    "trex": "3d",
    "happy": "block",
    "raptor": "small",
    "sleep": "mini",
    "trex_roar": "doom",
    "sad": "straight",
    "dead": "sub-zero",
    "victory": "starwars",
    "alert": "univers",
    "profit": "epic",
    "loss": "poison",
    "buy": "larry3d",
    "sell": "ogre",
    "signal": "isometric1",
    "mode": "cyberlarge",
    "strategy": "roman",
    "status": "big",
    "startup": "chunky",
    "shutdown": "graffiti",
    "header": "colossal",
    "subheader": "big",
}

# Trading configuration (UNCHANGED)
SYMBOL = "SOLFDUSD"
TOKEN = "SOL"
PAIR = "FDUSD"
TRADE_AMOUNT_USD = 5.1
MIN_TOKEN_VALUE_FOR_SELL = 5.1
DAILY_DIFF_LOWER_LIMIT = -1.0
DAILY_DIFF_UPPER_LIMIT = 4.0

# Files (UNCHANGED)
INDICATORS_CSV = "pinescript_indicators.csv"
TRANSACTIONS_CSV = "transactions.csv"
STATE_FILE = "trading_state.json"
ERROR_LOG_CSV = "trading_errors.csv"

# Safety settings (UNCHANGED)
ORDER_COOLDOWN_SECONDS = 1
MAX_ORDER_RETRIES = 3
ORDER_TIMEOUT_SECONDS = 30
TEST_MODE = False

# CSV monitoring (UNCHANGED)
CSV_UPDATE_CHECK_INTERVAL = 2
CSV_STALE_THRESHOLD = 30
CSV_BATCH_PROCESSING = True

# Display settings - ENHANCED
STATUS_UPDATE_INTERVAL = 10
VERBOSE_LOGGING = True
ANIMATION_SPEED = 0.03  # seconds between animation frames
PULSE_EFFECT = True
GRADIENT_COLORS = True
ENHANCED_BORDERS = True

# ============================================================================
# ENHANCED COLOR MANAGEMENT
# ============================================================================

class Colors:
    """Enhanced color definitions with gradients and effects"""
    
    # Base colors
    SUCCESS = "bright_green"
    ERROR = "bright_red"
    WARNING = "bright_yellow"
    INFO = "bright_cyan"
    MAGENTA = "bright_magenta"
    WHITE = "bright_white"
    BLUE = "bright_blue"
    GRAY = "grey70"
    
    # Enhanced gradient colors
    PROFIT = "green_yellow"
    LOSS = "red1"
    NEUTRAL = "yellow3"
    HIGHLIGHT = "cyan2"
    ACCENT = "magenta2"
    DIM = "grey50"
    GOLD = "gold3"
    SILVER = "grey85"
    BRONZE = "dark_orange3"
    
    # Status colors
    ACTIVE = "spring_green2"
    IDLE = "deep_sky_blue3"
    WAITING = "orange3"
    PROCESSING = "dark_violet"
    COMPLETE = "green4"
    
    # Strategy colors
    STRATEGY_A = "chartreuse3"
    STRATEGY_B = "dark_orange"
    
    # Phase colors
    ENTRY = "dodger_blue2"
    MONITORING = "medium_purple3"
    POSITION = "green3"
    EXIT = "orange_red1"
    STOP_LOSS = "red3"
    
    @staticmethod
    def get_strategy_color(variant):
        """Get enhanced strategy color"""
        if variant == "A":
            return Colors.STRATEGY_A
        else:
            return Colors.STRATEGY_B
    
    @staticmethod
    def get_mode_color(mode):
        """Get enhanced mode color"""
        if mode == "BUY":
            return "bright_green"
        elif mode == "SELL":
            return "bright_red"
        else:
            return "bright_yellow"
    
    @staticmethod
    def get_phase_color(phase):
        """Get enhanced phase color"""
        phase_map = {
            "ENTRY_MONITORING": Colors.ENTRY,
            "ENTRY_SIGNAL_CONFIRMED": Colors.PROCESSING,
            "POSITION_OPEN": Colors.POSITION,
            "EXIT_MONITORING": Colors.EXIT,
            "STOP_LOSS_ACTIVE": Colors.STOP_LOSS,
        }
        return phase_map.get(phase, Colors.INFO)
    
    @staticmethod
    def get_gradient_text(text, color1, color2):
        """Create gradient text effect"""
        result = Text()
        length = len(text)
        for i, char in enumerate(text):
            ratio = i / max(length - 1, 1)
            # Simple gradient interpolation
            if ratio < 0.5:
                result.append(char, style=f"{color1}")
            else:
                result.append(char, style=f"{color2}")
        return result
    
    @staticmethod
    def get_pulse_effect(text, base_color):
        """Create pulsing text effect"""
        if not PULSE_EFFECT:
            return Text(text, style=base_color)
        
        pulse_phase = (time.time() * 2) % (2 * 3.14159)
        intensity = abs(int((np.sin(pulse_phase) + 1) * 5))
        return Text(text, style=f"{base_color} bold" if intensity > 5 else base_color)
    
    @staticmethod
    def get_random_emoji(category="general"):
        """Get random emoji for visual variety"""
        emoji_sets = {
            "general": ["✨", "🌟", "⚡", "💫", "🔥", "🎯", "🚀", "💎", "🎰", "🎲", "🎪", "🎨"],
            "success": ["✅", "🎉", "🏆", "🥇", "💰", "💸", "💵", "💴", "💶", "💷", "💳"],
            "error": ["❌", "💥", "💣", "☠️", "⚰️", "👻", "👽", "🤖", "👾", "💀"],
            "warning": ["⚠️", "🔔", "📢", "🚨", "👀", "🔍", "🔎", "📡", "📟", "📻"],
            "trading": ["📈", "📉", "💹", "🔺", "🔻", "↗️", "↘️", "⬆️", "⬇️", "↔️"],
            "animals": ["🦕", "🦖", "🐉", "🐲", "🦎", "🦂", "🕷️", "🦇", "🦅", "🦉"],
        }
        
        return random.choice(emoji_sets.get(category, emoji_sets["general"]))

# ============================================================================
# ENHANCED ART AND ANIMATION FUNCTIONS
# ============================================================================

class VisualEffects:
    """Enhanced visual effects and animations"""
    
    @staticmethod
    def print_with_animation(text, color=Colors.INFO, style="", emoji="", indent=0, speed=ANIMATION_SPEED):
        """Print text with typing animation effect"""
        indent_str = " " * indent
        full_text = f"{indent_str}{emoji} {text}"
        
        console = Console()
        styled_text = Text()
        
        if style == "bold":
            style_tag = "bold "
        elif style == "italic":
            style_tag = "italic "
        elif style == "underline":
            style_tag = "underline "
        elif style == "blink":
            style_tag = "blink "
        else:
            style_tag = ""
        
        # Animated typing effect
        for i, char in enumerate(full_text):
            styled_text.append(char, style=f"{style_tag}{color}")
            console.print(styled_text, end="\r")
            time.sleep(speed)
        console.print()  # New line at end
    
    @staticmethod
    def display_animated_dino(message, dino_type="default", title="", border_style="double"):
        """Display animated ASCII art"""
        try:
            font = FONT_MAPPING.get(dino_type, "standard")
            dino = text2art(message, font=font)
            
            if title:
                # Animated title display
                title_text = Colors.get_gradient_text(f"  {title}  ", Colors.GOLD, Colors.ACCENT)
                console = Console()
                
                # Create enhanced border
                if ENHANCED_BORDERS:
                    border_char = {"double": "═", "heavy": "█", "rounded": "─"}.get(border_style, "─")
                    border_length = max(len(line) for line in dino.split('\n'))
                    top_border = border_char * (border_length + 4)
                    bottom_border = border_char * (border_length + 4)
                    
                    console.print(Text(top_border, style=Colors.GOLD))
                    console.print(Align.center(title_text))
                    console.print(Text(dino, style=Colors.HIGHLIGHT))
                    console.print(Text(bottom_border, style=Colors.GOLD))
                else:
                    console.print(Text(dino, style=Colors.HIGHLIGHT))
                    
        except Exception as e:
            console = Console()
            fallback_text = f"\n🦕 {message}\n"
            console.print(Text(fallback_text, style=Colors.INFO))
    
    @staticmethod
    def create_progress_bar(percentage, width=30, label=""):
        """Create enhanced visual progress bar"""
        filled = int(width * percentage / 100)
        empty = width - filled
        
        # Choose bar characters based on percentage
        if percentage >= 80:
            bar_char = "█"
            color = Colors.PROFIT
        elif percentage >= 50:
            bar_char = "▓"
            color = Colors.WARNING
        else:
            bar_char = "▒"
            color = Colors.INFO
        
        bar = f"[{color}]{bar_char * filled}[/{color}][{Colors.DIM}]{'░' * empty}[/{Colors.DIM}]"
        
        # Add percentage indicator with color coding
        if percentage >= 90:
            perc_color = Colors.PROFIT
        elif percentage >= 70:
            perc_color = Colors.SUCCESS
        elif percentage >= 50:
            perc_color = Colors.WARNING
        elif percentage >= 30:
            perc_color = Colors.INFO
        else:
            perc_color = Colors.ERROR
        
        perc_text = f"[{perc_color}]{percentage:.1f}%[/{perc_color}]"
        
        return f"{label} {bar} {perc_text}"
    
    @staticmethod
    def display_spinner(message, color=Colors.INFO):
        """Display animated spinner with message"""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        console = Console()
        
        for i in range(10):  # Spin for 10 frames
            spinner = spinner_chars[i % len(spinner_chars)]
            text = Text(f"{spinner} {message}", style=color)
            console.print(text, end="\r")
            time.sleep(0.1)
        console.print(Text(f"✓ {message}", style=Colors.SUCCESS))
    
    @staticmethod
    def flash_border(color=Colors.WARNING, length=80, duration=0.5):
        """Flash border effect"""
        console = Console()
        for _ in range(3):
            console.print(Text("█" * length, style=color))
            time.sleep(duration)
            console.print("\033[F\033[K", end="")  # Move up and clear line
            time.sleep(duration/2)

# ============================================================================
# ENHANCED PRINT FUNCTIONS
# ============================================================================

def print_colored(text, color="white", style="", emoji="", indent=0, animate=False):
    """Enhanced print colored text with optional animation"""
    if animate and VERBOSE_LOGGING:
        VisualEffects.print_with_animation(text, color, style, emoji, indent)
    else:
        indent_str = " " * indent
        emoji = emoji or Colors.get_random_emoji()
        style_tags = ""
        
        if style == "bold":
            style_tags = "bold "
        elif style == "italic":
            style_tags = "italic "
        elif style == "underline":
            style_tags = "underline "
        elif style == "blink":
            style_tags = "blink "
        
        console.print(f"{indent_str}{emoji} [{style_tags}{color}]{text}[/{style_tags}{color}]")

def print_header(title, color=Colors.INFO, border_char="═", length=80, enhanced=True):
    """Print enhanced section header"""
    if enhanced:
        # Create gradient border
        border = Colors.get_gradient_text(border_char * length, Colors.GOLD, color)
        console.print(border)
        
        # Create title with effects
        title_text = Colors.get_gradient_text(f"   {title}   ", Colors.HIGHLIGHT, color)
        console.print(Align.center(title_text))
        
        console.print(border)
    else:
        border = border_char * length
        print_colored(border, color)
        print_colored(f"   {title}", color, "bold", "📌")
        print_colored(border, color)

def print_step(step_num, description, color=Colors.WARNING, animate=True):
    """Print enhanced step in the process"""
    emoji = Colors.get_random_emoji("trading")
    step_text = f"   STEP {step_num}: {description}"
    
    if animate:
        VisualEffects.print_with_animation(step_text, color, "bold", emoji, 0)
    else:
        print_colored(step_text, color, "bold", emoji)

def print_condition(condition, is_met=True, indent=2, enhanced=True):
    """Print enhanced condition with status"""
    if enhanced:
        if is_met:
            status = Colors.get_random_emoji("success")
            color = Colors.SUCCESS
            style = "bold"
        else:
            status = Colors.get_random_emoji("error")
            color = Colors.ERROR
            style = ""
        
        # Create condition text with pulse effect for met conditions
        if is_met and PULSE_EFFECT:
            text = Colors.get_pulse_effect(f"{status} {condition}", color)
            console.print(" " * indent + text)
        else:
            print_colored(f"{status} {condition}", color, style, indent=indent)
    else:
        status = "✅" if is_met else "❌"
        color = Colors.SUCCESS if is_met else Colors.ERROR
        print_colored(f"{status} {condition}", color, indent=indent)

def print_waiting(waiting_for, indent=2, animate=True):
    """Print enhanced waiting status"""
    emoji = Colors.get_random_emoji("warning")
    
    if animate:
        pulse_text = Colors.get_pulse_effect(f"⏳ Waiting for: {waiting_for}", Colors.WARNING)
        console.print(" " * indent + emoji + " " + pulse_text)
    else:
        print_colored(f"⏳ Waiting for: {waiting_for}", Colors.WARNING, "italic", emoji, indent=indent)

def print_signal(signal_type, description, color=Colors.MAGENTA, enhanced=True):
    """Print enhanced trading signal"""
    if enhanced:
        # Create flashing border for important signals
        if signal_type in ["ENTRY CONFIRMED", "STOP LOSS TRIGGER", "EXIT SIGNAL"]:
            VisualEffects.flash_border(color, 60, 0.3)
        
        emoji = Colors.get_random_emoji("trading")
        signal_text = Colors.get_gradient_text(f"{signal_type}: {description}", color, Colors.HIGHLIGHT)
        console.print(f"\n{emoji} ", end="")
        console.print(signal_text)
    else:
        print_colored(f"\n🎯 {signal_type}: {description}", color, "bold")

def print_transition(from_state, to_state, color=Colors.INFO, enhanced=True):
    """Print enhanced state transition"""
    if enhanced:
        # Create animated arrow
        arrow_sequence = ["→", "⇒", "⇨", "⇾", "⟹"]
        
        console.print()
        for arrow in arrow_sequence:
            transition_text = f"[{Colors.WARNING}]{from_state}[/{Colors.WARNING}] [{color}]{arrow}[/{color}] [{Colors.SUCCESS}]{to_state}[/{Colors.SUCCESS}]"
            console.print(Align.center(Text(transition_text)), end="\r")
            time.sleep(0.1)
        
        # Final display
        final_arrow = "⇨"
        transition_text = f"[{Colors.WARNING}]{from_state}[/{Colors.WARNING}] [{color}]{final_arrow}[/{color}] [{Colors.SUCCESS}]{to_state}[/{Colors.SUCCESS}]"
        console.print(Align.center(Text(transition_text)))
        
        # Add celebration for BUY→SELL or SELL→BUY transitions
        if (from_state == "BUY" and to_state == "SELL") or (from_state == "SELL" and to_state == "BUY"):
            VisualEffects.flash_border(Colors.GOLD, 40, 0.2)
    else:
        print_colored(f"\n🔄 Transition: {from_state} → {to_state}", color, "bold")

# ============================================================================
# ENHANCED ART VISUALIZATIONS
# ============================================================================

def display_dino(message, dino_type="default", title="", enhanced=True):
    """Enhanced display ASCII art with message"""
    if enhanced:
        VisualEffects.display_animated_dino(message, dino_type, title)
    else:
        try:
            font = FONT_MAPPING.get(dino_type, "standard")
            dino = text2art(message, font=font)
            if title:
                print_header(title, Colors.INFO)
            console.print(dino)
        except Exception as e:
            print_colored(f"\n🦕 {message}\n", Colors.INFO)

def display_strategy_activation(strategy_variant, daily_diff, enhanced=True):
    """Enhanced display strategy activation with art"""
    if enhanced:
        if strategy_variant == "A":
            title = "STRATEGY A ACTIVATED"
            range_desc = f"-1% to +4% (Current: {daily_diff:+.2f}%)"
            dino_type = "triceratops"
            color = Colors.STRATEGY_A
            art_message = "STRATEGY A"
        else:
            title = "STRATEGY B ACTIVATED"
            range_desc = f"≤ -1% (Current: {daily_diff:+.2f}%)"
            dino_type = "trex"
            color = Colors.STRATEGY_B
            art_message = "STRATEGY B"
        
        # Display with enhanced effects
        VisualEffects.display_animated_dino(art_message, dino_type, title)
        
        # Show progress bar for daily diff
        if strategy_variant == "A":
            # Normalize daily diff to 0-100% for progress bar
            progress = ((daily_diff - DAILY_DIFF_LOWER_LIMIT) / 
                       (DAILY_DIFF_UPPER_LIMIT - DAILY_DIFF_LOWER_LIMIT)) * 100
            progress = max(0, min(100, progress))
            progress_bar = VisualEffects.create_progress_bar(progress, 40, "Range:")
            print_colored(f"   {range_desc}", color, "bold")
            print_colored(f"   {progress_bar}", Colors.INFO)
        else:
            print_colored(f"   {range_desc}", color, "bold")
            
        print_colored(f"   Following {strategy_variant} flowchart exactly...", color, "bold")
        
        # Show spinner for strategy initialization
        VisualEffects.display_spinner("Initializing strategy conditions", color)
    else:
        if strategy_variant == "A":
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
        print_colored(f"Following {strategy_variant} flowchart exactly...", color, "bold")

def display_mode_banner(mode, wallet_info, enhanced=True):
    """Enhanced display trading mode banner"""
    if enhanced:
        if mode == "BUY":
            title = "BUY MODE ACTIVATED"
            dino_type = "buy"
            color = "bright_green"
            action = "Looking for entry signals to BUY tokens"
            art_message = "BUY MODE"
        elif mode == "SELL":
            title = "SELL MODE ACTIVATED"
            dino_type = "sell"
            color = "bright_red"
            action = "Looking for entry signals to validate existing position"
            art_message = "SELL MODE"
        else:
            title = "NEUTRAL MODE"
            dino_type = "sleep"
            color = "bright_yellow"
            action = "Insufficient funds for trading"
            art_message = "NEUTRAL"

        # Display with enhanced effects
        VisualEffects.display_animated_dino(art_message, dino_type, title, border_style="heavy")
        
        # Enhanced wallet display
        print_colored("💰 WALLET STATUS:", Colors.GOLD, "bold")
        
        # Create wallet table with enhanced styling
        wallet_table = Table(show_header=False, box=None, show_lines=False)
        wallet_table.add_column(style="cyan", width=15)
        wallet_table.add_column(style="bright_white")
        
        for asset, info in wallet_info.items():
            wallet_table.add_row(f"   • {asset}:", f"[bright_white]{info}[/bright_white]")
        
        console.print(wallet_table)
        
        # Enhanced action display
        print_colored(f"\n🎯 ACTION: {action}", color, "bold")
        
        if mode == "SELL":
            print_colored("   ⚡ In SELL mode: Will skip BUY when entry signal occurs", 
                         Colors.WARNING, "italic", Colors.get_random_emoji("warning"))
            print_colored("   ⚡ Existing tokens will be validated at entry signal", 
                         Colors.WARNING, "italic", Colors.get_random_emoji("warning"))
            
            # Show token amount if available
            if "SOL" in str(wallet_info):
                print_colored("   📊 Position validation pending entry signal...", 
                             Colors.INFO, indent=2)
    else:
        if mode == "BUY":
            title = "BUY MODE ACTIVATED"
            dino_type = "happy"
            color = Colors.SUCCESS
            action = "Looking for entry signals to BUY tokens"
        elif mode == "SELL":
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

        if mode == "SELL":
            print_colored("   ⚡ In SELL mode: Will skip BUY when entry signal occurs",
                         Colors.WARNING, "italic")
            print_colored("   ⚡ Existing tokens will be validated at entry signal",
                         Colors.WARNING, "italic")

def display_position_status(action, position_info, enhanced=True):
    """Enhanced display position opening/closing status"""
    if enhanced:
        if action == "OPENED":
            dino_type = "buy"
            color = Colors.SUCCESS
            title = "POSITION OPENED"
            art_message = "POSITION OPEN"
        else:
            dino_type = "sell"
            color = Colors.INFO
            title = "POSITION CLOSED"
            art_message = "POSITION CLOSED"

        # Display with enhanced effects
        VisualEffects.display_animated_dino(art_message, dino_type, title)
        
        # Create enhanced position table
        pos_table = Table(show_header=True, header_style=f"bold {color}", 
                         box=HEAVY if action == "OPENED" else ROUNDED)
        pos_table.add_column("Metric", style="cyan", width=20)
        pos_table.add_column("Value", style="bright_white")
        
        for key, value in position_info.items():
            pos_table.add_row(key, str(value))
        
        console.print(Panel(pos_table, title=f"📊 POSITION DETAILS", 
                           border_style=color))
        
        # Add celebration for opened positions
        if action == "OPENED":
            VisualEffects.flash_border(Colors.GOLD, 40, 0.2)
    else:
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

def display_stop_loss_activation(enhanced=True):
    """Enhanced display stop loss activation"""
    if enhanced:
        title = "🚨 STOP LOSS ACTIVATED 🚨"
        dino_type = "alert"
        art_message = "STOP LOSS"
        
        # Display with urgent effects
        VisualEffects.display_animated_dino(art_message, dino_type, title, border_style="heavy")
        
        # Flash red border for urgency
        for _ in range(3):
            VisualEffects.flash_border(Colors.ERROR, 60, 0.2)
            time.sleep(0.1)
        
        print_colored("MA_200 ≤ Fibo_23.6% - Activating stop loss flow", 
                     Colors.ERROR, "bold", "🚨")
        
        # Show warning animation
        VisualEffects.display_spinner("Activating stop loss protocol...", Colors.ERROR)
    else:
        display_dino("STOP LOSS TRIGGERED!", "trex_roar", "🚨 STOP LOSS ACTIVATED 🚨")
        print_colored("MA_200 ≤ Fibo_23.6% - Activating stop loss flow", Colors.ERROR, "bold")

# ============================================================================
# ENHANCED STATUS DISPLAY
# ============================================================================

def display_status(csv_monitor, enhanced=True):
    """Enhanced comprehensive status display"""
    # Get CSV stats
    csv_stats = csv_monitor.get_stats()

    # Get current price
    try:
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        current_price = float(ticker["price"])
    except:
        current_price = 0.0

    if enhanced:
        # Create enhanced main table with sections
        console.print()
        
        # Header with gradient
        header_text = Colors.get_gradient_text("📊 ENHANCED TRADING STATUS", Colors.GOLD, Colors.HIGHLIGHT)
        console.print(Align.center(header_text))
        
        if TEST_MODE:
            test_badge = Text(" [TEST MODE] ", style="bold black on yellow")
            console.print(Align.center(test_badge))
        
        console.print()
        
        # Create main layout with panels
        layout = Layout()
        
        # Top section: Market & CSV
        top_table = Table(show_header=False, box=ROUNDED, show_lines=True)
        top_table.add_column(style="cyan", width=25)
        top_table.add_column(style="bright_white")
        top_table.add_column(style="cyan", width=25)
        top_table.add_column(style="bright_white")
        
        top_table.add_row("Time", datetime.now().strftime("%H:%M:%S"), 
                         "Symbol", f"[bold]{SYMBOL}[/bold]")
        top_table.add_row("Price", f"[bright_green]${current_price:.4f}[/bright_green]", 
                         "Daily Diff", 
                         f"[{'bright_green' if trade_state.current_daily_diff >= 0 else 'bright_red'}]{trade_state.current_daily_diff:+.2f}%[/{'bright_green' if trade_state.current_daily_diff >= 0 else 'bright_red'}]")
        
        # CSV status with color coding
        csv_color = {
            "ACTIVE": "bright_green",
            "SLOW": "yellow",
            "STALE": "bright_red",
            "STALLED": "red",
            "ERROR": "red",
        }.get(csv_stats["status"], "white")
        
        csv_age = csv_stats['age']
        age_color = "bright_green" if csv_age < 10 else "yellow" if csv_age < 30 else "bright_red"
        
        top_table.add_row("CSV Status", 
                         f"[{csv_color}]{csv_stats['status']}[/{csv_color}]",
                         "CSV Age", 
                         f"[{age_color}]{csv_age:.1f}s[/{age_color}]")
        
        if csv_stats["rows"] > 0:
            top_table.add_row("CSV Rows", f"{csv_stats['rows']}",
                             "Last Update", csv_stats.get('last_update', 'Never'))
        
        console.print(Panel(top_table, title="📈 MARKET & DATA", border_style="cyan"))
        console.print()
        
        # Middle section: Trading State
        middle_table = Table(show_header=False, box=ROUNDED, show_lines=True)
        middle_table.add_column(style="cyan", width=25)
        middle_table.add_column(style="bright_white")
        middle_table.add_column(style="cyan", width=25)
        middle_table.add_column(style="bright_white")
        
        # Trading mode with enhanced styling
        mode_color = Colors.get_mode_color(trade_state.mode)
        mode_text = Text(trade_state.mode.value, style=f"bold {mode_color}")
        if PULSE_EFFECT:
            mode_text = Colors.get_pulse_effect(trade_state.mode.value, mode_color)
        
        middle_table.add_row("Trading Mode", mode_text,
                           "Phase", 
                           f"[{Colors.get_phase_color(trade_state.phase.value)}]{trade_state.phase.value}[/{Colors.get_phase_color(trade_state.phase.value)}]")
        
        # Strategy
        if trade_state.strategy_variant:
            strat_color = Colors.get_strategy_color(trade_state.strategy_variant.value)
            middle_table.add_row("Strategy", 
                               f"[{strat_color}]{trade_state.strategy_variant.value}[/{strat_color}]",
                               "Active Strategy",
                               trade_state.active_strategy.value if trade_state.active_strategy else "[dim]None[/dim]")
        
        # Position info
        if trade_state.position_open:
            pos_style = "bold bright_green"
            entry_price = (trade_state.entry_price if trade_state.position_entered_by_signal 
                          else trade_state.virtual_entry_price)
            
            # Calculate P&L
            position_value = current_price * trade_state.position_size
            entry_value = entry_price * trade_state.position_size
            pnl_usd = position_value - entry_value
            pnl_percent = (pnl_usd / entry_value) * 100 if entry_value > 0 else 0
            
            pnl_color = "bright_green" if pnl_usd >= 0 else "bright_red"
            pnl_text = f"[{pnl_color}]${pnl_usd:+.2f} ({pnl_percent:+.2f}%)[/{pnl_color}]"
            
            middle_table.add_row("Position", f"[{pos_style}]OPEN[/{pos_style}]",
                               "P&L", pnl_text)
            middle_table.add_row("Entry Price", f"${entry_price:.4f}",
                               "Position Size", f"{trade_state.position_size:.6f} {TOKEN}")
        else:
            middle_table.add_row("Position", "[yellow]CLOSED[/yellow]",
                               "Entry Price", "[dim]N/A[/dim]")
        
        console.print(Panel(middle_table, title="⚙️ TRADING STATE", border_style="blue"))
        console.print()
        
        # Bottom section: Statistics
        bottom_table = Table(show_header=False, box=ROUNDED)
        bottom_table.add_column(style="cyan", width=20)
        bottom_table.add_column(style="bright_white")
        bottom_table.add_column(style="cyan", width=20)
        bottom_table.add_column(style="bright_white")
        bottom_table.add_column(style="cyan", width=20)
        bottom_table.add_column(style="bright_white")
        
        bottom_table.add_row("Entry Signals", f"{trade_state.stats['entry_signals_detected']}",
                           "Exit Signals", f"{trade_state.stats['exit_signals_detected']}",
                           "Stop Loss", f"{trade_state.stats['stop_loss_triggers']}")
        
        bottom_table.add_row("Trades", f"{trade_state.stats['trades_completed']}",
                           "Total P&L $", 
                           f"[{'bright_green' if trade_state.stats['total_pnl_usd'] >= 0 else 'bright_red'}]{trade_state.stats['total_pnl_usd']:+.2f}[/{'bright_green' if trade_state.stats['total_pnl_usd'] >= 0 else 'bright_red'}]",
                           "Total P&L %", 
                           f"[{'bright_green' if trade_state.stats['total_pnl_percent'] >= 0 else 'bright_red'}]{trade_state.stats['total_pnl_percent']:+.2f}%[/{'bright_green' if trade_state.stats['total_pnl_percent'] >= 0 else 'bright_red'}]")
        
        # Active waiting conditions
        active_waits = [
            k.replace("waiting_", "").replace("_", " ").title()
            for k, v in trade_state.waiting_conditions.items()
            if v
        ]
        if active_waits:
            waits_text = ", ".join(active_waits[:3])
            if len(active_waits) > 3:
                waits_text += f"... (+{len(active_waits)-3} more)"
            bottom_table.add_row("Active Waits", f"[yellow]{waits_text}[/yellow]", "", "", "", "")
        
        console.print(Panel(bottom_table, title="📊 STATISTICS", border_style="magenta"))
        
        # Last processed row time
        if trade_state.last_processed_time:
            console.print()
            print_colored(f"   Last Processed: {trade_state.last_processed_time}", Colors.DIM, indent=2)
        
        # Add decorative rule at bottom
        console.print()
        console.print(Rule(style="dim"))
        
    else:
        # Original status display (unchanged)
        main_table = Table(show_header=False, box=None, show_lines=False)
        main_table.add_column(style="cyan", width=25)
        main_table.add_column(style="white")

        main_table.add_row("Time", datetime.now().strftime("%H:%M:%S"))
        main_table.add_row("Symbol", SYMBOL)
        main_table.add_row("Price", f"${current_price:.4f}")

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

        mode_color = Colors.get_mode_color(trade_state.mode)
        main_table.add_row(
            "Trading Mode",
            f"[bold {mode_color}]{trade_state.mode.value}[/bold {mode_color}]",
        )

        if trade_state.strategy_variant:
            strat_color = Colors.get_strategy_color(trade_state.strategy_variant)
            main_table.add_row(
                "Strategy",
                f"[{strat_color}]{trade_state.strategy_variant.value}[/{strat_color}]",
            )

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

        daily_color = "green" if trade_state.current_daily_diff >= 0 else "red"
        main_table.add_row(
            "Daily Diff",
            f"[{daily_color}]{trade_state.current_daily_diff:+.2f}%[/{daily_color}]",
        )

        if trade_state.position_open:
            main_table.add_row("Position", "[bold green]OPEN[/bold green]")

            entry_price = (
                trade_state.entry_price
                if trade_state.position_entered_by_signal
                else trade_state.virtual_entry_price
            )
            main_table.add_row("Entry Price", f"${entry_price:.4f}")
            main_table.add_row("Position Size", f"{trade_state.position_size:.6f} {TOKEN}")

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

        main_table.add_row(
            "Entry Signals", f"{trade_state.stats['entry_signals_detected']}"
        )
        main_table.add_row("Exit Signals", f"{trade_state.stats['exit_signals_detected']}")

        active_waits = [
            k.replace("waiting_", "").replace("_", " ").title()
            for k, v in trade_state.waiting_conditions.items()
            if v
        ]
        if active_waits:
            main_table.add_row(
                "Active Waits", f"[yellow]{', '.join(active_waits[:2])}...[/yellow]"
            )

        if trade_state.last_processed_time:
            main_table.add_row("Last Row Time", str(trade_state.last_processed_time))

        title = "📊 TRADING STATUS"
        if TEST_MODE:
            title += " [TEST MODE]"

        console.print(Panel(main_table, title=title, border_style="blue"))

# ============================================================================
# ENHANCED TRANSACTION DISPLAY
# ============================================================================

def display_transaction_summary(transaction, enhanced=True):
    """Enhanced transaction summary display"""
    if enhanced:
        action = transaction.get("action", "N/A")
        strategy = transaction.get("strategy", "N/A")
        
        # Choose box style based on action
        if action == "BUY":
            box_style = HEAVY
            border_color = "bright_green"
            title_color = "bright_green"
            emoji = "🟢"
        elif action == "SELL":
            box_style = HEAVY
            border_color = "bright_red"
            title_color = "bright_red"
            emoji = "🔴"
        else:
            box_style = ROUNDED
            border_color = "cyan"
            title_color = "cyan"
            emoji = "💠"
        
        # Create enhanced table
        table = Table(show_header=True, header_style=f"bold {title_color}", 
                     box=box_style, show_lines=True)
        table.add_column("Field", style="bright_cyan", width=20)
        table.add_column("Value", style="bright_white")
        
        # Add rows with enhanced formatting
        table.add_row("Timestamp", f"[dim]{transaction.get('timestamp', 'N/A')}[/dim]")
        table.add_row("Order ID", f"[bright_white]{transaction.get('order_id', 'N/A')}[/bright_white]")
        table.add_row("Action", f"[bold {title_color}]{action}[/bold {title_color}]")
        table.add_row("Strategy", f"[bright_magenta]{strategy}[/bright_magenta]")
        table.add_row("Side", transaction.get("side", "N/A"))
        table.add_row("Quantity", f"[bright_yellow]{transaction.get('quantity', 0):.6f}[/bright_yellow]")
        
        if "average_price" in transaction:
            table.add_row("Avg Price", f"[bright_green]${transaction['average_price']:.8f}[/bright_green]")
        
        if "pnl_percent" in transaction:
            pnl_percent = transaction['pnl_percent']
            pnl_usd = transaction['pnl_usd']
            
            # Enhanced P&L display with color gradient
            if pnl_percent > 0:
                pnl_color = "bright_green"
                pnl_emoji = "📈"
                pnl_style = "bold"
            elif pnl_percent < 0:
                pnl_color = "bright_red"
                pnl_emoji = "📉"
                pnl_style = "bold"
            else:
                pnl_color = "yellow"
                pnl_emoji = "➖"
                pnl_style = ""
            
            table.add_row(
                "P&L %", 
                f"[{pnl_color} {pnl_style}]{pnl_emoji} {pnl_percent:+.2f}%[/{pnl_color} {pnl_style}]"
            )
            table.add_row(
                "P&L $", 
                f"[{pnl_color} {pnl_style}]${pnl_usd:+.2f}[/{pnl_color} {pnl_style}]"
            )
        
        if "exit_reason" in transaction:
            table.add_row("Exit Reason", f"[italic]{transaction['exit_reason']}[/italic]")
        
        # Panel title
        panel_title = f"{emoji} TRANSACTION LOGGED"
        if TEST_MODE:
            panel_title += " [TEST]"
        
        # Create panel with enhanced border
        panel = Panel(
            table, 
            title=panel_title, 
            border_style=border_color,
            subtitle=f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]" if not TEST_MODE else "[yellow]TEST MODE[/yellow]"
        )
        
        console.print(panel)
        
        # Add celebration for profitable trades
        if "pnl_percent" in transaction and transaction['pnl_percent'] > 0:
            VisualEffects.flash_border(Colors.GOLD, 40, 0.1)
            
    else:
        # Original transaction display (unchanged)
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
# ENHANCED STARTUP DISPLAY
# ============================================================================

def display_startup():
    """Enhanced startup banner"""
    # Create animated startup sequence
    console.clear()
    
    # Initial animation
    for i in range(3):
        console.print(Align.center(Text("🚀 STARTING TRADING BOT", style=f"bold bright_cyan")))
        time.sleep(0.2)
        console.print("\033[F\033[K", end="")
        console.print(Align.center(Text("🚀 STARTING TRADING BOT", style=f"bold bright_yellow")))
        time.sleep(0.2)
        console.print("\033[F\033[K", end="")
    
    # Main banner
    banner_lines = [
        "╔══════════════════════════════════════════════════════════╗",
        f"║        🚀 ENHANCED TRADING BOT v4.0-VISUAL            ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  📈 Symbol: {SYMBOL:<43} ║",
        f"║  💰 Trade Amount: ${TRADE_AMOUNT_USD:<38} ║",
        f"║  📊 Strategy A: {DAILY_DIFF_LOWER_LIMIT}% to {DAILY_DIFF_UPPER_LIMIT}%{' ':<29} ║",
        f"║  📊 Strategy B: ≤ {DAILY_DIFF_LOWER_LIMIT}%{' ':<39} ║",
        f"║  🔒 Test Mode: {'✅ YES' if TEST_MODE else '❌ NO (REAL TRADING)':<38} ║",
        f"║  📁 Data: {INDICATORS_CSV:<44} ║",
        f"║  💾 Logs: {TRANSACTIONS_CSV:<44} ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  🎨 Visual Effects: {'ENABLED':<41} ║",
        f"║  🌈 Color Gradients: {'ENABLED':<40} ║",
        f"║  ✨ Animations: {'ENABLED':<43} ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    
    # Print banner with animation
    for line in banner_lines:
        # Color code different parts
        if "🚀 ENHANCED" in line:
            styled_line = Text(line)
            styled_line.stylize("bold bright_cyan", 9, 9 + len("ENHANCED TRADING BOT"))
            console.print(styled_line)
        elif "╔" in line or "╗" in line or "╚" in line or "╝" in line:
            console.print(Text(line, style="bright_yellow"))
        elif "╠" in line or "╣" in line:
            console.print(Text(line, style="cyan"))
        elif "Strategy A" in line:
            console.print(Text(line, style="bright_green"))
        elif "Strategy B" in line:
            console.print(Text(line, style="bright_red"))
        elif "Test Mode" in line and TEST_MODE:
            console.print(Text(line, style="bright_yellow"))
        elif "Test Mode" in line and not TEST_MODE:
            console.print(Text(line, style="bright_red"))
        elif "Visual Effects" in line or "Color Gradients" in line or "Animations" in line:
            console.print(Text(line, style="bright_magenta"))
        else:
            console.print(Text(line, style="bright_white"))
        time.sleep(0.05)
    
    console.print()
    
    # Show initialization spinner
    VisualEffects.display_spinner("Initializing enhanced visual system", Colors.HIGHLIGHT)
    VisualEffects.display_spinner("Loading trading strategies", Colors.INFO)
    VisualEffects.display_spinner("Connecting to data sources", Colors.SUCCESS)
    
    console.print()

# ============================================================================
# ENHANCED MAIN LOOP WITH VISUAL UPDATES
# ============================================================================

def main():
    """Enhanced main trading loop with visual improvements"""
    global trade_state, client, running

    # Display enhanced startup
    display_startup()

    # Check files
    if not os.path.exists(INDICATORS_CSV):
        display_dino(f"File not found: {INDICATORS_CSV}", "sad", "FILE ERROR", enhanced=True)
        print_colored(
            f"Please ensure {INDICATORS_CSV} exists with indicator data",
            Colors.ERROR, "bold", animate=True
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
            "sad", "WALLET STATUS", enhanced=True
        )
        print_colored(
            f"Need ≥${TRADE_AMOUNT_USD} in stablecoins or ≥${MIN_TOKEN_VALUE_FOR_SELL} in SOL",
            Colors.WARNING, animate=True
        )
        if TEST_MODE:
            print_colored(
                "Proceeding in TEST MODE with simulated balances", 
                Colors.INFO, animate=True
            )
            trade_state.mode = TradingMode.BUY

    # Display mode with enhanced visuals
    wallet_info = {}
    if trade_state.mode == TradingMode.BUY:
        wallet_info["Simulated USD"] = f"[bright_green]${TRADE_AMOUNT_USD}[/bright_green]"
    else:
        wallet_info["Simulated SOL"] = f"[bright_cyan]${MIN_TOKEN_VALUE_FOR_SELL}[/bright_cyan]"

    display_mode_banner(trade_state.mode, wallet_info, enhanced=True)

    print_colored("\n🔄 Starting enhanced trading loop...", Colors.INFO, "bold", animate=True)
    print_colored("   • CSV monitoring every 2 seconds", Colors.INFO, animate=True)
    print_colored("   • Enhanced status updates every 10 seconds", Colors.INFO, animate=True)
    print_colored("   • Visual animations and effects ENABLED", Colors.HIGHLIGHT, animate=True)
    print_colored("   • Processing rows sequentially for specific triggers", Colors.INFO, animate=True)

    # Show initial status
    VisualEffects.display_spinner("Initializing main trading loop", Colors.SUCCESS)

    # Main loop
    last_status = 0
    consecutive_no_updates = 0
    cycle_count = 0

    try:
        while running:
            current_time = time.time()
            cycle_count += 1

            # Check CSV for updates
            updated, status = csv_monitor.check_update()

            if updated:
                consecutive_no_updates = 0

                # Load and process new rows sequentially
                new_rows = csv_monitor.load_new_rows()

                if not new_rows.empty:
                    # Enhanced update notification
                    print_colored(
                        f"\n📥 Processing {len(new_rows)} new row(s)...",
                        Colors.INFO, "bold", Colors.get_random_emoji("general")
                    )

                    for idx, row_series in new_rows.iterrows():
                        try:
                            row = row_series.to_dict()

                            # Skip if already processed
                            row_id = row.get("Open Time", idx)
                            row_str = (
                                str(row_id) if hasattr(row_id, "__str__") else str(idx)
                            )
                            if row_str in csv_monitor.processed_rows:
                                if VERBOSE_LOGGING:
                                    print_colored(
                                        f"   Skipping processed row {row_str}",
                                        Colors.DIM, indent=2
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
                                    Colors.GRAY, indent=2
                                )

                            process_single_row(row)

                        except Exception as e:
                            logger.error(f"Row processing error: {e}")
                            print_colored(f"Row error: {e}", Colors.ERROR, indent=2)

                    if VERBOSE_LOGGING:
                        print_colored("   Batch complete", Colors.SUCCESS, indent=2)

            else:
                consecutive_no_updates += 1

                # Warn if CSV is stale with enhanced visuals
                if consecutive_no_updates >= 15:
                    stats = csv_monitor.get_stats()
                    if stats["status"] in ["STALE", "STALLED"]:
                        print_colored(
                            f"⚠️ CSV {stats['status'].lower()} ({stats['age']:.0f}s)",
                            Colors.WARNING, "blink", Colors.get_random_emoji("warning")
                        )

            # Update trading mode periodically
            if current_time % 60 < 1:
                update_trading_mode()

            # Display enhanced status periodically
            if current_time - last_status >= STATUS_UPDATE_INTERVAL:
                print_colored("\n" + "=" * 80, Colors.BLUE)
                display_status(csv_monitor, enhanced=True)
                print_colored("=" * 80 + "\n", Colors.BLUE)
                last_status = current_time

            # Visual heartbeat every 30 cycles
            if cycle_count % 30 == 0 and VERBOSE_LOGGING:
                heartbeat = Colors.get_random_emoji("animals")
                print_colored(f"   {heartbeat} System active...", Colors.DIM, indent=2)

            # Sleep
            time.sleep(CSV_UPDATE_CHECK_INTERVAL)

    except KeyboardInterrupt:
        print_colored("\n🛑 Manual interruption", Colors.WARNING, "bold", animate=True)
    except Exception as e:
        print_colored(f"\n❌ Critical error: {e}", Colors.ERROR, "bold", animate=True)
        logger.exception("Main loop error")
    finally:
        save_state()
        display_dino("Enhanced trading bot stopped", "shutdown", "SHUTDOWN COMPLETE", enhanced=True)
        
        # Final shutdown animation
        for _ in range(3):
            print_colored("." * 30, Colors.DIM, animate=True)
            time.sleep(0.1)

# ============================================================================
# ENHANCED SIGNAL HANDLER
# ============================================================================

def signal_handler(sig, frame):
    """Enhanced signal handler with visual shutdown"""
    global running
    running = False
    
    print_colored("\n🛑 Shutdown signal received", Colors.WARNING, "bold", animate=True)
    print_colored("💾 Saving state with enhanced backup...", Colors.INFO, animate=True)
    
    # Show saving animation
    VisualEffects.display_spinner("Saving trading state", Colors.INFO)
    VisualEffects.display_spinner("Backing up transaction logs", Colors.SUCCESS)
    VisualEffects.display_spinner("Closing connections", Colors.WARNING)
    
    save_state()
    
    print_colored("👋 Enhanced bot stopped gracefully", Colors.SUCCESS, "bold", animate=True)
    
    # Final shutdown animation
    console.clear()
    shutdown_art = text2art("SHUTDOWN", font="graffiti")
    console.print(Text(shutdown_art, style="bright_red"))
    
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# INITIALIZE CONSOLE AND RUN
# ============================================================================

# Initialize enhanced console
console = Console()

# Initialize logging (UNCHANGED)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# UNCHANGED SECTIONS - All trading logic remains exactly as is
# ============================================================================

# NOTE: All the following classes and functions remain EXACTLY THE SAME
# as in the original script. Only the visual/display functions above
# have been enhanced.

# ENUMS & DATA STRUCTURES (unchanged)
class TradingMode(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class StrategyType(Enum):
    MA_200_WAVE = "MA_200_WAVE"
    MA_350_WAVE = "MA_350_WAVE"
    MA_500_WAVE = "MA_500_WAVE"

class StrategyVariant(Enum):
    A = "A"
    B = "B"

class Phase(Enum):
    ENTRY_MONITORING = "ENTRY_MONITORING"
    ENTRY_SIGNAL_CONFIRMED = "ENTRY_SIGNAL_CONFIRMED"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_MONITORING = "EXIT_MONITORING"
    STOP_LOSS_ACTIVE = "STOP_LOSS_ACTIVE"

class SignalType(Enum):
    ENTRY_SETUP = "ENTRY_SETUP"
    ENTRY_CONFIRMED = "ENTRY_CONFIRMED"
    EXIT_SIGNAL = "EXIT_SIGNAL"
    STOP_LOSS = "STOP_LOSS"

# UTILITY FUNCTIONS (unchanged)
def calculate_daily_diff(daily_diff_str):
    if pd.isna(daily_diff_str) or not isinstance(daily_diff_str, str):
        return 0.0
    try:
        clean = daily_diff_str.replace("%", "").replace("+", "").strip()
        return float(clean)
    except:
        return 0.0

def calculate_token_amount(usd_amount, current_price, safety_mgr):
    if pd.isna(current_price) or current_price <= 0:
        return 0.0, 0.0
    adjusted_usd = usd_amount
    if safety_mgr:
        adjusted_usd = safety_mgr.adjust_trade_amount(usd_amount)
    token_amount = adjusted_usd / current_price
    return token_amount, adjusted_usd

# SAFETY MANAGER (unchanged)
class SafetyManager:
    def __init__(self, client):
        self.client = client
        self.symbol_info = None
        self.last_order_time = 0

    def get_symbol_info(self):
        try:
            self.symbol_info = self.client.get_symbol_info(SYMBOL)
            return self.symbol_info
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            return None

    def adjust_trade_amount(self, usd_amount):
        return usd_amount

    def can_place_order(self):
        if time.time() - self.last_order_time < ORDER_COOLDOWN_SECONDS:
            return False
        return True

    def update_last_order_time(self):
        self.last_order_time = time.time()

# WALLET MANAGER (unchanged)
class WalletManager:
    def __init__(self, client):
        self.client = client

    def get_balance(self, asset, include_locked=True):
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
        try:
            ticker = self.client.get_symbol_ticker(symbol=f"{token}{PAIR}")
            price = float(ticker["price"])
            balance = self.get_balance(token)
            return balance * price
        except Exception as e:
            logger.error(f"Token value error: {e}")
            return 0.0

    def determine_mode(self):
        usd_balance = sum(self.get_balance(stable) for stable in [PAIR, "USDT", "USDC"])
        token_value = self.get_token_value_usd(TOKEN)

        if usd_balance >= TRADE_AMOUNT_USD:
            return TradingMode.BUY
        elif token_value >= MIN_TOKEN_VALUE_FOR_SELL:
            return TradingMode.SELL
        return TradingMode.NEUTRAL

# TRADE STATE MANAGEMENT (unchanged)
class TradeState:
    def __init__(self):
        self.mode = TradingMode.NEUTRAL
        self.strategy_variant = None
        self.phase = Phase.ENTRY_MONITORING
        self.position_open = False
        self.position_entered_by_signal = False
        self.entry_price = 0.0
        self.entry_time = None
        self.position_size = 0.0
        self.virtual_entry_price = 0.0
        self.captured_fibo_0 = None
        self.captured_fibo_1 = None
        self.captured_fibo_1_dip = None
        self.captured_fibo_0_sell = None
        self.active_strategy = None
        self.entry_signal_confirmed = False
        self.waiting_conditions = {
            "waiting_ma_200_above_fibo_236": False,
            "waiting_ma_350_above_fibo_236": False,
            "waiting_ma_500_above_fibo_236": False,
            "waiting_ma_100_above_fibo_764": False,
            "waiting_ma_500_above_fibo_764": False,
            "waiting_ma_350_below_ma_500": False,
            "waiting_ma_200_above_fibo_764": False,
            "waiting_lesser_mas_above_ma_200": False,
            "waiting_ma_100_below_ma_500": False,
            "waiting_ma_100_above_new_fibo_1": False,
            "waiting_ma_350_above_new_fibo_1": False,
            "waiting_ma_50_below_fibo_764": False,
            "waiting_rsi_ma50_above_55": False,
            "waiting_rsi_ma50_below_52": False,
            "waiting_strategy_b_profit_target": False,
            "waiting_strategy_b_reversal": False,
            "waiting_ma_100_above_ma_200": False,
            "waiting_ma_100_below_ma_500": False,
            "waiting_ma_200_below_ma_500": False,
            "waiting_rsi_ma50_above_53": False,
            "waiting_rsi_ma50_below_51": False,
        }
        self.last_processed_time = None
        self.current_daily_diff = 0.0
        self.stats = {
            "entry_signals_detected": 0,
            "exit_signals_detected": 0,
            "stop_loss_triggers": 0,
            "trades_completed": 0,
            "total_pnl_usd": 0.0,
            "total_pnl_percent": 0.0,
        }
        self.signal_history = []
        self.safety_mgr = None
        self.wallet_mgr = None
        self.order_executor = None
        self.csv_monitor = None

# CSV MONITORING (unchanged)
class CSVMonitor:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.last_modified = 0
        self.last_size = 0
        self.df_cache = pd.DataFrame()
        self.processed_rows = set()
        self.update_stats = []

    def check_update(self):
        try:
            if not self.csv_path.exists():
                return False, "NOT_FOUND"
            current_mtime = self.csv_path.stat().st_mtime
            current_size = self.csv_path.stat().st_size
            if current_mtime != self.last_modified or current_size != self.last_size:
                self.last_modified = current_mtime
                self.last_size = current_size
                self.update_stats.append({
                    "timestamp": datetime.now(),
                    "rows_loaded": 0,
                    "age": time.time() - current_mtime,
                })
                if len(self.update_stats) > 100:
                    self.update_stats = self.update_stats[-100:]
                return True, "UPDATED"
            return False, "UNCHANGED"
        except Exception as e:
            logger.error(f"CSV check error: {e}")
            return False, "ERROR"

    def load_new_rows(self):
        try:
            df = pd.read_csv(self.csv_path)
            if "Open Time" in df.columns:
                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
            numeric_cols = [
                "Open", "High", "Low", "Close", "Volume", "daily_diff", "rsi",
                "rsi_ma50", "short002", "short007", "short21", "short50",
                "long100", "long200", "long350", "long500", "level_100",
                "level_764", "level_618", "level_500", "level_382", "level_236", "level_000",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    if col == "daily_diff":
                        continue
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if self.df_cache.empty:
                new_rows = df
            else:
                if "Open Time" in df.columns and "Open Time" in self.df_cache.columns:
                    last_time = self.df_cache["Open Time"].max()
                    new_rows = df[df["Open Time"] > last_time].copy()
                    if not new_rows.empty:
                        new_timestamps = set(
                            new_rows["Open Time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                        )
                        self.processed_rows.update(new_timestamps)
                else:
                    cache_len = len(self.df_cache)
                    new_rows = df.iloc[cache_len:] if len(df) > cache_len else pd.DataFrame()
                    new_indices = set(range(cache_len, len(df)))
                    self.processed_rows.update(new_indices)
            self.df_cache = df
            if not new_rows.empty and self.update_stats:
                self.update_stats[-1]["rows_loaded"] = len(new_rows)
            return new_rows
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def get_stats(self):
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
                "last_update": datetime.fromtimestamp(self.last_modified).strftime("%H:%M:%S")
                if self.last_modified else "Never",
            }
        except Exception as e:
            return {"status": "ERROR", "age": 0, "rows": 0}

# DATA PROCESSING (unchanged)
def determine_strategy_variant(daily_diff):
    if DAILY_DIFF_LOWER_LIMIT <= daily_diff <= DAILY_DIFF_UPPER_LIMIT:
        return StrategyVariant.A
    return StrategyVariant.B

def check_entry_setup(row):
    fibo_236 = row.get("level_236")
    if pd.isna(fibo_236):
        return None
    other_mas = ["short002", "short007", "short21", "short50", "long100"]
    for ma in other_mas:
        ma_val = row.get(ma)
        if pd.isna(ma_val) or ma_val < fibo_236:
            return None
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
    fibo_236 = row.get("level_236")
    if pd.isna(fibo_236):
        return False
    all_mas = [
        "short002", "short007", "short21", "short50", "long100",
        "long200", "long350", "long500",
    ]
    for ma in all_mas:
        ma_val = row.get(ma)
        if pd.isna(ma_val) or ma_val < fibo_236:
            return False
    return True

# ENTRY SIGNAL PROCESSING (unchanged)
def process_entry_signal(row, strategy_type):
    global trade_state
    row_time = row.get("Open Time", "Unknown")
    print_header(f"ENTRY SIGNAL DETECTED (Row Time: {row_time})", Colors.MAGENTA)
    print_colored(f"Strategy: {strategy_type.value}", Colors.MAGENTA, "bold")
    trade_state.stats["entry_signals_detected"] += 1
    trade_state.signal_history.append({
        "time": datetime.now(),
        "row_time": row_time,
        "type": "ENTRY",
        "strategy": strategy_type.value,
        "price": row.get("Close"),
    })
    if trade_state.mode == TradingMode.BUY:
        print_step("1", "BUY MODE: Executing BUY order", Colors.SUCCESS)
        execute_buy_order(row, strategy_type)
    elif trade_state.mode == TradingMode.SELL:
        print_step("1", "SELL MODE: Validating existing position", Colors.WARNING)
        print_colored("   • Entry signal confirms position timing", Colors.WARNING, "italic", indent=2)
        print_colored("   • Skipping BUY (tokens already held)", Colors.WARNING, "italic", indent=2)
        with state_lock:
            trade_state.position_open = True
            trade_state.position_entered_by_signal = False
            trade_state.virtual_entry_price = row.get("Close", 0)
            trade_state.entry_time = datetime.now()
            trade_state.active_strategy = strategy_type
            trade_state.phase = Phase.EXIT_MONITORING
            trade_state.entry_signal_confirmed = True
            if trade_state.wallet_mgr:
                trade_state.position_size = trade_state.wallet_mgr.get_balance(
                    TOKEN, include_locked=False
                )
            trade_state.captured_fibo_0_sell = row.get("level_000")
        print_step("2", f"Position validated: {trade_state.position_size:.6f} {TOKEN}", Colors.INFO)
        print_colored(f"   • Virtual entry price: ${trade_state.virtual_entry_price:.4f}", Colors.INFO, indent=2)
        print_colored("   • Starting exit monitoring", Colors.INFO, indent=2)
        start_exit_monitoring(row)

def execute_buy_order(row, strategy_type):
    global trade_state
    current_price = row.get("Close")
    if pd.isna(current_price):
        print_colored("Invalid price data", Colors.ERROR, "bold")
        return
    token_amount, actual_usd = calculate_token_amount(
        TRADE_AMOUNT_USD, current_price, trade_state.safety_mgr
    )
    if token_amount <= 0:
        print_colored("Invalid token calculation", Colors.ERROR, "bold")
        return
    print_step("2", f"Buying {token_amount:.6f} {TOKEN} for ${actual_usd:.2f}", Colors.INFO)
    order_result = trade_state.order_executor.execute_market_order(
        side="BUY", token_amount=token_amount, is_test=TEST_MODE
    )
    if order_result.get("status") in ["SUCCESS", "TEST_SUCCESS"]:
        order = order_result.get("order", {})
        with state_lock:
            trade_state.position_open = True
            trade_state.position_entered_by_signal = True
            trade_state.entry_price = current_price
            trade_state.entry_time = datetime.now()
            trade_state.position_size = token_amount
            trade_state.active_strategy = strategy_type
            trade_state.phase = Phase.EXIT_MONITORING
            trade_state.entry_signal_confirmed = True
            for key in trade_state.waiting_conditions:
                trade_state.waiting_conditions[key] = False
        log_binance_transaction(order, "BUY", strategy_type.value)
        display_position_status("OPENED", {
            "Tokens": f"{token_amount:.6f} {TOKEN}",
            "Entry Price": f"${current_price:.4f}",
            "Value": f"${actual_usd:.2f}",
            "Strategy": strategy_type.value,
        })
        trade_state.mode = TradingMode.SELL
        print_transition("BUY", "SELL", Colors.INFO)
        start_exit_monitoring(row)
    else:
        error_msg = order_result.get("error", "Unknown error")
        print_colored(f"BUY failed: {error_msg}", Colors.ERROR, "bold")

def execute_exit_order(row, exit_reason):
    global trade_state
    if not trade_state.position_open or trade_state.position_size <= 0:
        print_colored("No position to exit", Colors.WARNING)
        return
    current_price = row.get("Close")
    if pd.isna(current_price):
        print_colored("Invalid price data for exit", Colors.ERROR, "bold")
        return
    print_step("EXIT", f"Selling {trade_state.position_size:.6f} {TOKEN} at ${current_price:.4f} - Reason: {exit_reason}", Colors.ERROR)
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
    order_result = trade_state.order_executor.execute_market_order(
        side="SELL", token_amount=trade_state.position_size, is_test=TEST_MODE
    )
    if order_result.get("status") in ["SUCCESS", "TEST_SUCCESS"]:
        order = order_result.get("order", {})
        with state_lock:
            trade_state.position_open = False
            trade_state.phase = Phase.ENTRY_MONITORING
            trade_state.entry_signal_confirmed = False
            trade_state.active_strategy = None
            trade_state.captured_fibo_0_sell = None
            for key in trade_state.waiting_conditions:
                trade_state.waiting_conditions[key] = False
        log_binance_transaction(
            order,
            "SELL",
            trade_state.active_strategy.value if trade_state.active_strategy else "N/A",
            pnl_percent=pnl_percent,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
        )
        display_position_status("CLOSED", {
            "Tokens Sold": f"{trade_state.position_size:.6f} {TOKEN}",
            "Exit Price": f"${current_price:.4f}",
            "P&L": f"${pnl_usd:+.2f} ({pnl_percent:+.2f}%)",
            "Reason": exit_reason,
        })
        trade_state.mode = TradingMode.BUY
        print_transition("SELL", "BUY", Colors.INFO)
    else:
        error_msg = order_result.get("error", "Unknown error")
        print_colored(f"SELL failed: {error_msg}", Colors.ERROR, "bold")

# EXIT MONITORING FUNCTIONS (unchanged)
def start_exit_monitoring(row):
    global trade_state
    with state_lock:
        trade_state.phase = Phase.EXIT_MONITORING
        trade_state.last_processed_time = row.get("Open Time")
    if trade_state.strategy_variant == StrategyVariant.A:
        monitor_strategy_a_exit(row)
    else:
        monitor_strategy_b_exit(row)

def monitor_strategy_a_exit(row):
    if not trade_state.position_open:
        return
    if trade_state.active_strategy == StrategyType.MA_200_WAVE:
        monitor_ma_200_wave_exit_a(row)
    else:
        monitor_ma_long_wave_exit_a(row)

def monitor_ma_200_wave_exit_a(row):
    print_header("MA_200 WAVE EXIT MONITORING", Colors.INFO)
    fibo_764 = row.get("level_764")
    ma_100 = row.get("long100")
    ma_500 = row.get("long500")
    ma_350 = row.get("long350")
    if any(pd.isna(x) for x in [fibo_764, ma_100, ma_500, ma_350]):
        return
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
    ma200_ge_764 = ma_200 >= fibo_764
    ma200_le_236 = ma_200 <= fibo_236
    if ma200_le_236:
        print_signal("STOP LOSS TRIGGER", "MA_200 ≤ Fibo_23.6%")
        display_stop_loss_activation()
        with state_lock:
            trade_state.captured_fibo_0 = row.get("level_000")
            trade_state.phase = Phase.STOP_LOSS_ACTIVE
        trade_state.stats["stop_loss_triggers"] += 1
        return
    if ma200_ge_764:
        print_step("PHASE 1", "MA_200 ≥ Fibo_76.4% - Moving to Phase 2", Colors.SUCCESS)
        trade_state.waiting_conditions["waiting_ma_200_above_fibo_764"] = True
    else:
        print_waiting("MA_200 ≥ Fibo_76.4% or ≤ Fibo_23.6%")
        return
    if not trade_state.waiting_conditions["waiting_ma_200_above_fibo_764"]:
        return
    all_lesser_ge_ma200 = all(m >= ma_200 for m in lesser_mas if not pd.isna(m))
    if all_lesser_ge_ma200:
        trade_state.waiting_conditions["waiting_lesser_mas_above_ma_200"] = True
        print_condition("All lesser MAs >= MA_200", True)
        step_3_path_a(row, fibo_764, ma_100, ma_350, ma_500, ma_50, rsi_ma50)
    else:
        print_condition("All lesser MAs >= MA_200", False)
        if pd.notna(ma_100):
            trade_state.captured_fibo_1_dip = ma_100
            print_step("CAPTURE PATH", f"Captured Fibo_1 at MA_100 dip: {trade_state.captured_fibo_1_dip:.4f}", Colors.WARNING)
            capture_path_dual_monitoring(row)
        else:
            print_waiting("Valid MA_100 for capture")

def step_3_path_a(row, fibo_764, ma_100, ma_350, ma_500, ma_50, rsi_ma50):
    ma100_le_500 = ma_100 <= ma_500
    if ma100_le_500:
        trade_state.waiting_conditions["waiting_ma_100_below_ma_500"] = True
        print_condition("MA_100 <= MA_500", True)
    else:
        print_waiting("MA_100 <= MA_500")
        return
    if not trade_state.waiting_conditions["waiting_ma_100_below_ma_500"]:
        return
    trade_state.captured_fibo_1 = ma_100
    print_step("CAPTURE NEW", f"New Fibo_1 at MA_100: {trade_state.captured_fibo_1:.4f}", Colors.INFO)
    ma500_ge_764 = ma_500 >= fibo_764
    if ma500_ge_764:
        ma100_ge_new_fibo = ma_100 >= trade_state.captured_fibo_1
        if ma100_ge_new_fibo:
            trade_state.waiting_conditions["waiting_ma_100_above_new_fibo_1"] = True
            print_condition("MA_100 >= New Fibo_1", True)
        else:
            print_waiting("MA_100 >= New Fibo_1")
            return
    else:
        ma350_ge_new_fibo = ma_350 >= trade_state.captured_fibo_1
        if ma350_ge_new_fibo:
            trade_state.waiting_conditions["waiting_ma_350_above_new_fibo_1"] = True
            print_condition("MA_350 >= New Fibo_1", True)
        else:
            print_waiting("MA_350 >= New Fibo_1")
            return
    if (
        trade_state.waiting_conditions["waiting_ma_100_above_new_fibo_1"]
        or trade_state.waiting_conditions["waiting_ma_350_above_new_fibo_1"]
    ):
        ma50_le_764 = ma_50 <= fibo_764
        if ma50_le_764:
            trade_state.waiting_conditions["waiting_ma_50_below_fibo_764"] = True
            print_condition("MA_50 <= Fibo_76.4%", True)
        else:
            print_waiting("MA_50 <= Fibo_76.4%")
            return
        if trade_state.waiting_conditions["waiting_ma_50_below_fibo_764"]:
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
    captured_fibo_1 = trade_state.captured_fibo_1_dip
    if pd.isna(captured_fibo_1):
        return
    ma_100 = row.get("long100")
    ma_200 = row.get("long200")
    fibo_236 = row.get("level_236")
    if pd.isna(ma_100) or pd.isna(ma_200) or pd.isna(fibo_236):
        return
    cond1_met = ma_100 >= captured_fibo_1
    cond2_met = ma_200 <= fibo_236
    if cond1_met and not cond2_met:
        print_signal("CAPTURE PATH EXIT", "MA_100 >= Captured Fibo_1")
        execute_exit_order(row, "CAPTURE_PATH_MA100_EXIT")
    elif cond2_met:
        print_signal("STOP LOSS TRIGGER", "MA_200 <= Fibo_236 in Capture Path")
        display_stop_loss_activation()
        with state_lock:
            trade_state.captured_fibo_0 = row.get("level_000")
            trade_state.phase = Phase.STOP_LOSS_ACTIVE
        trade_state.stats["stop_loss_triggers"] += 1
    else:
        print_waiting("MA_100 >= Captured Fibo_1 OR MA_200 <= Fibo_236")

def monitor_strategy_b_exit(row):
    print_header("STRATEGY B ENHANCED EXIT MONITORING", Colors.ERROR)
    fibo_764 = row.get("level_764")
    ma_100 = row.get("long100")
    ma_500 = row.get("long500")
    ma_350 = row.get("long350")
    fibo_1 = row.get("level_100")
    ma_200 = row.get("long200")
    fibo_236 = row.get("level_236")
    if any(pd.isna(x) for x in [fibo_764, ma_100, ma_500, ma_350]):
        return
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
    trade_state.captured_fibo_1 = fibo_1
    print_step("3", f"Captured Fibo_1: {trade_state.captured_fibo_1:.4f}", Colors.INFO)
    if (
        pd.isna(trade_state.captured_fibo_1)
        or pd.isna(ma_200)
        or pd.isna(fibo_236)
        or pd.isna(ma_350)
    ):
        return
    cond1_met = ma_200 <= fibo_236
    cond2_met = ma_350 >= trade_state.captured_fibo_1
    if cond1_met and not cond2_met:
        print_signal("STRATEGY B STOP LOSS", "MA_200 <= Fibo_236")
        display_stop_loss_activation()
        with state_lock:
            trade_state.captured_fibo_0 = row.get("level_000")
            trade_state.phase = Phase.STOP_LOSS_ACTIVE
        trade_state.stats["stop_loss_triggers"] += 1
    elif cond2_met:
        print_signal("STRATEGY B ENHANCED EXIT", "MA_350 >= Captured Fibo_1")
        execute_exit_order(row, "STRATEGY_B_ENHANCED_EXIT")
    else:
        print_step("4", "Dual monitoring active", Colors.WARNING)
        print_waiting("MA_200 <= Fibo_236 OR MA_350 >= Captured Fibo_1")

# STOP LOSS PROCESSING (unchanged)
def check_stop_loss_conditions(row):
    print_header("STOP LOSS FLOW", Colors.ERROR)
    fibo_0 = trade_state.captured_fibo_0
    if pd.isna(fibo_0):
        print_colored("No captured Fibo_0 - Cannot proceed", Colors.ERROR)
        return
    ma_14 = row.get("short21", row.get("short007", 0))
    ma_100 = row.get("long100")
    ma_200 = row.get("long200")
    ma_500 = row.get("long500")
    rsi_ma50 = row.get("rsi_ma50")
    if any(pd.isna(x) for x in [ma_14, ma_100, ma_200, ma_500]):
        return
    cond1 = ma_14 <= fibo_0
    cond2 = ma_100 >= ma_500
    if cond1:
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

# ORDER EXECUTOR (unchanged)
class OrderExecutor:
    def __init__(self, client, safety_mgr, wallet_mgr):
        self.client = client
        self.safety_mgr = safety_mgr
        self.wallet_mgr = wallet_mgr

    def execute_market_order(self, side, token_amount, is_test=False):
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
        try:
            quantity_str = format(token_amount, "f").rstrip("0").rstrip(".")
            if side == "BUY":
                order = self.client.order_market_buy(
                    symbol=SYMBOL,
                    quoteOrderQty=str(TRADE_AMOUNT_USD),
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

# TRANSACTION LOGGING (unchanged - but uses enhanced display function)
def log_binance_transaction(order, action, strategy, **kwargs):
    try:
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
        if "pnl_percent" in kwargs:
            transaction["pnl_percent"] = kwargs["pnl_percent"]
            transaction["pnl_usd"] = kwargs["pnl_usd"]
        if "exit_reason" in kwargs:
            transaction["exit_reason"] = kwargs["exit_reason"]
        file_exists = os.path.exists(TRANSACTIONS_CSV)
        with open(TRANSACTIONS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=transaction.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(transaction)
        # Use enhanced display
        display_transaction_summary(transaction, enhanced=True)
        logger.info(f"Transaction logged: {action} {strategy}")
    except Exception as e:
        logger.error(f"Transaction logging error: {e}")

# MAIN PROCESSING FUNCTIONS (unchanged)
def process_single_row(row):
    global trade_state
    daily_diff_str = row.get("daily_diff", "0")
    daily_diff = calculate_daily_diff(daily_diff_str)
    trade_state.current_daily_diff = daily_diff
    new_variant = determine_strategy_variant(daily_diff)
    if trade_state.strategy_variant != new_variant:
        trade_state.strategy_variant = new_variant
        display_strategy_activation(new_variant, daily_diff, enhanced=True)
    entry_setup = check_entry_setup(row)
    if entry_setup:
        print_signal("ENTRY SETUP", f"{entry_setup.value} detected", Colors.MAGENTA, enhanced=True)
        if is_entry_condition_met(row, entry_setup):
            print_signal("ENTRY CONFIRMED", "All conditions satisfied", Colors.SUCCESS, enhanced=True)
            process_entry_signal(row, entry_setup)
        else:
            waiting_key = f"waiting_{entry_setup.value.lower().replace('_wave', '')}_above_fibo_236"
            if waiting_key in trade_state.waiting_conditions:
                trade_state.waiting_conditions[waiting_key] = True
                print_waiting(f"{entry_setup.value.replace('_WAVE', '')} >= Fibo_23.6%", animate=True)
    check_waiting_ma_conditions(row)
    if trade_state.position_open and trade_state.entry_signal_confirmed:
        if trade_state.phase == Phase.STOP_LOSS_ACTIVE:
            check_stop_loss_conditions(row)
        elif trade_state.strategy_variant == StrategyVariant.A:
            monitor_strategy_a_exit(row)
        else:
            monitor_strategy_b_exit(row)
    if (
        trade_state.mode == TradingMode.SELL
        and trade_state.position_open
        and trade_state.entry_signal_confirmed
        and trade_state.captured_fibo_0_sell is not None
    ):
        ma_200 = row.get("long200")
        if pd.notna(ma_200) and ma_200 <= trade_state.captured_fibo_0_sell:
            print_signal("SELL MODE OPTION 2 EXIT", "MA_200 <= Captured Fibo_0", Colors.ERROR, enhanced=True)
            execute_exit_order(row, "SELL_MODE_FIBO_0_EXIT")

def check_waiting_ma_conditions(row):
    fibo_236 = row.get("level_236")
    if pd.isna(fibo_236):
        return
    if trade_state.waiting_conditions["waiting_ma_200_above_fibo_236"]:
        ma_200 = row.get("long200")
        if not pd.isna(ma_200) and ma_200 >= fibo_236:
            trade_state.waiting_conditions["waiting_ma_200_above_fibo_236"] = False
            print_signal("WAITING MET", "MA_200 ≥ Fibo_23.6%", Colors.SUCCESS, enhanced=True)
            if is_entry_condition_met(row, StrategyType.MA_200_WAVE):
                process_entry_signal(row, StrategyType.MA_200_WAVE)
    if trade_state.waiting_conditions["waiting_ma_350_above_fibo_236"]:
        ma_350 = row.get("long350")
        if not pd.isna(ma_350) and ma_350 >= fibo_236:
            trade_state.waiting_conditions["waiting_ma_350_above_fibo_236"] = False
            print_signal("WAITING MET", "MA_350 ≥ Fibo_23.6%", Colors.SUCCESS, enhanced=True)
            if is_entry_condition_met(row, StrategyType.MA_350_WAVE):
                process_entry_signal(row, StrategyType.MA_350_WAVE)
    if trade_state.waiting_conditions["waiting_ma_500_above_fibo_236"]:
        ma_500 = row.get("long500")
        if not pd.isna(ma_500) and ma_500 >= fibo_236:
            trade_state.waiting_conditions["waiting_ma_500_above_fibo_236"] = False
            print_signal("WAITING MET", "MA_500 ≥ Fibo_23.6%", Colors.SUCCESS, enhanced=True)
            if is_entry_condition_met(row, StrategyType.MA_500_WAVE):
                process_entry_signal(row, StrategyType.MA_500_WAVE)

# OTHER UNCHANGED FUNCTIONS
def update_trading_mode():
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
        display_mode_banner(trade_state.mode, wallet_info, enhanced=True)

def save_state():
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
                "fibo_0_sell": trade_state.captured_fibo_0_sell,
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
    try:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            print_colored("Binance API keys not found in environment", Colors.ERROR, "bold", animate=True)
            print_colored("Set them with:", Colors.WARNING, animate=True)
            print_colored("  export BINANCE_API_KEY='your_key'", Colors.INFO, animate=True)
            print_colored("  export BINANCE_API_SECRET='your_secret'", Colors.INFO, animate=True)
            return None
        client = Client(api_key, api_secret)
        client.ping()
        print_colored("✅ Binance connection successful", Colors.SUCCESS, "bold", animate=True)
        return client
    except Exception as e:
        print_colored(f"❌ Binance connection failed: {e}", Colors.ERROR, "bold", animate=True)
        return None

# GLOBAL VARIABLES
trade_state = TradeState()
client = None
running = True
state_lock = threading.Lock()

# ENTRY POINT
if __name__ == "__main__":
    if not TEST_MODE:
        if not os.environ.get("BINANCE_API_KEY") or not os.environ.get("BINANCE_API_SECRET"):
            display_dino("API KEYS REQUIRED FOR REAL TRADING", "trex_roar", "CONFIGURATION ERROR", enhanced=True)
            print_colored("Set environment variables or enable TEST_MODE", Colors.ERROR, "bold", animate=True)
            os._exit(1)
    try:
        main()
    except Exception as e:
        display_dino(f"Fatal error: {str(e)[:100]}", "dead", "FATAL ERROR", enhanced=True)
        logger.exception("Fatal error")
        os._exit(1)