"""
Combined Analysis Tools for Finance Analyst AI Agent

This module provides tools for combining technical and fundamental analysis
to create comprehensive stock evaluations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import base64
from io import BytesIO
import yfinance as yf
import warnings
import json

# Import local modules
from tools.fundamental_analysis import FundamentalAnalysisTools
from tools.enhanced_visualization import EnhancedVisualizationTools

class CombinedAnalysisTools:
    """Tools for combining technical and fundamental analysis"""
    
    @staticmethod
    def create_combined_analysis(symbol: str, period: str = "1y") -> Dict:
        """
        Create a comprehensive analysis combining technical and fundamental data
        
        Args:
            symbol: Stock ticker symbol
            period: Time period for technical analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with combined analysis results
        """
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return {"error": f"Could not retrieve data for {symbol}"}
            
            # Get company name
            company_name = info.get('longName', symbol)
            
            # Get technical indicators
            data = ticker.history(period=period)
            if data.empty:
                return {"error": f"Could not retrieve historical data for {symbol}"}
            
            # Calculate technical indicators
            # 1. Moving Averages
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            
            # 2. RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['Signal']
            
            # 4. Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            data['BB_Std'] = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
            data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
            
            # 5. Volatility
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252) * 100  # Annualized
            
            # Get fundamental data
            financial_ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
            industry_comparison = FundamentalAnalysisTools.get_industry_comparison(symbol)
            
            # Create combined analysis visualization
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(4, 2, figure=fig)
            
            # 1. Price chart with moving averages and Bollinger Bands
            ax_price = fig.add_subplot(gs[0, :])
            ax_price.plot(data.index, data['Close'], label='Close Price', color='blue')
            ax_price.plot(data.index, data['SMA50'], label='50-day MA', color='orange', alpha=0.7)
            ax_price.plot(data.index, data['SMA200'], label='200-day MA', color='red', alpha=0.7)
            ax_price.plot(data.index, data['BB_Upper'], label='Upper BB', color='green', linestyle='--', alpha=0.7)
            ax_price.plot(data.index, data['BB_Middle'], label='20-day MA', color='purple', alpha=0.7)
            ax_price.plot(data.index, data['BB_Lower'], label='Lower BB', color='green', linestyle='--', alpha=0.7)
            ax_price.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], color='green', alpha=0.05)
            ax_price.set_title(f"{company_name} ({symbol}) - Price Chart")
            ax_price.set_ylabel('Price')
            ax_price.grid(True, alpha=0.3)
            ax_price.legend(loc='upper left')
            
            # 2. Volume
            ax_volume = fig.add_subplot(gs[1, 0])
            ax_volume.bar(data.index, data['Volume'], label='Volume', color='blue', alpha=0.5)
            ax_volume.set_title('Volume')
            ax_volume.set_ylabel('Volume')
            ax_volume.grid(True, alpha=0.3)
            
            # 3. RSI
            ax_rsi = fig.add_subplot(gs[1, 1])
            ax_rsi.plot(data.index, data['RSI'], label='RSI (14-day)', color='purple')
            ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax_rsi.set_title('RSI')
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)
            
            # 4. MACD
            ax_macd = fig.add_subplot(gs[2, 0])
            ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue')
            ax_macd.plot(data.index, data['Signal'], label='Signal', color='red')
            ax_macd.bar(data.index, data['MACD_Histogram'], label='Histogram', color='green', alpha=0.5)
            ax_macd.set_title('MACD')
            ax_macd.grid(True, alpha=0.3)
            ax_macd.legend(loc='upper left')
            
            # 5. Volatility
            ax_vol = fig.add_subplot(gs[2, 1])
            ax_vol.plot(data.index, data['Volatility'], label='Volatility (20-day)', color='red')
            ax_vol.set_title('Volatility')
            ax_vol.set_ylabel('Volatility (%)')
            ax_vol.grid(True, alpha=0.3)
            
            # 6. Key Financial Ratios
            ax_ratios = fig.add_subplot(gs[3, :])
            
            # Extract key ratios for display
            ratio_names = []
            ratio_values = []
            
            if "error" not in financial_ratios:
                for ratio_name in ["pe_ratio", "peg_ratio", "ps_ratio", "pb_ratio", "de_ratio", "roe"]:
                    if ratio_name in financial_ratios["ratios"] and financial_ratios["ratios"][ratio_name]["value"] is not None:
                        ratio_names.append(ratio_name)
                        ratio_values.append(financial_ratios["ratios"][ratio_name]["value"])
            
            # Create horizontal bar chart for ratios
            y_pos = np.arange(len(ratio_names))
            ax_ratios.barh(y_pos, ratio_values, align='center')
            ax_ratios.set_yticks(y_pos)
            
            # Format ratio names for display
            display_names = []
            for name in ratio_names:
                if name == "pe_ratio":
                    display_names.append("P/E Ratio")
                elif name == "peg_ratio":
                    display_names.append("PEG Ratio")
                elif name == "ps_ratio":
                    display_names.append("P/S Ratio")
                elif name == "pb_ratio":
                    display_names.append("P/B Ratio")
                elif name == "de_ratio":
                    display_names.append("D/E Ratio")
                elif name == "roe":
                    display_names.append("ROE")
                else:
                    display_names.append(name.replace("_", " ").title())
            
            ax_ratios.set_yticklabels(display_names)
            ax_ratios.invert_yaxis()  # Labels read top-to-bottom
            ax_ratios.set_title('Key Financial Ratios')
            ax_ratios.set_xlabel('Value')
            ax_ratios.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            os.makedirs('outputs/combined_analysis', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = f"outputs/combined_analysis/combined_{symbol}_{timestamp}.png"
            plt.savefig(png_path, dpi=100)
            plt.close()
            
            # Prepare analysis summary
            summary = {
                "symbol": symbol,
                "company_name": company_name,
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "period": period,
                "technical_analysis": {},
                "fundamental_analysis": {},
                "combined_rating": {},
                "visualization_path": png_path
            }
            
            # Technical Analysis Summary
            current_price = data['Close'].iloc[-1]
            sma50 = data['SMA50'].iloc[-1]
            sma200 = data['SMA200'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            signal = data['Signal'].iloc[-1]
            
            # Determine price trend
            if sma50 > sma200:
                price_trend = "Bullish (Golden Cross)"
            elif sma50 < sma200:
                price_trend = "Bearish (Death Cross)"
            else:
                price_trend = "Neutral"
                
            # Determine RSI condition
            if rsi > 70:
                rsi_condition = "Overbought"
            elif rsi < 30:
                rsi_condition = "Oversold"
            else:
                rsi_condition = "Neutral"
                
            # Determine MACD condition
            if macd > signal:
                macd_condition = "Bullish"
            else:
                macd_condition = "Bearish"
                
            # Determine overall technical rating
            technical_signals = 0
            if sma50 > sma200:
                technical_signals += 1
            if current_price > sma50:
                technical_signals += 1
            if rsi > 50 and rsi < 70:
                technical_signals += 1
            if macd > signal:
                technical_signals += 1
                
            if technical_signals >= 3:
                technical_rating = "Bullish"
            elif technical_signals <= 1:
                technical_rating = "Bearish"
            else:
                technical_rating = "Neutral"
                
            summary["technical_analysis"] = {
                "current_price": current_price,
                "price_trend": price_trend,
                "rsi": {
                    "value": rsi,
                    "condition": rsi_condition
                },
                "macd": {
                    "value": macd,
                    "signal": signal,
                    "condition": macd_condition
                },
                "moving_averages": {
                    "sma50": sma50,
                    "sma200": sma200,
                    "relationship": "Above" if sma50 > sma200 else "Below"
                },
                "volatility": data['Volatility'].iloc[-1],
                "overall_rating": technical_rating
            }
            
            # Fundamental Analysis Summary
            if "error" not in financial_ratios:
                # Extract key ratios
                pe_ratio = financial_ratios["ratios"].get("pe_ratio", {}).get("value")
                peg_ratio = financial_ratios["ratios"].get("peg_ratio", {}).get("value")
                ps_ratio = financial_ratios["ratios"].get("ps_ratio", {}).get("value")
                pb_ratio = financial_ratios["ratios"].get("pb_ratio", {}).get("value")
                de_ratio = financial_ratios["ratios"].get("de_ratio", {}).get("value")
                roe = financial_ratios["ratios"].get("roe", {}).get("value")
                
                # Determine valuation
                fundamental_signals = 0
                
                # P/E ratio analysis
                if pe_ratio is not None:
                    if pe_ratio < 15:
                        pe_evaluation = "Potentially Undervalued"
                        fundamental_signals += 1
                    elif pe_ratio > 30:
                        pe_evaluation = "Potentially Overvalued"
                    else:
                        pe_evaluation = "Fair Valued"
                        fundamental_signals += 0.5
                else:
                    pe_evaluation = "Not Available"
                
                # PEG ratio analysis
                if peg_ratio is not None:
                    if peg_ratio < 1:
                        peg_evaluation = "Potentially Undervalued"
                        fundamental_signals += 1
                    elif peg_ratio > 2:
                        peg_evaluation = "Potentially Overvalued"
                    else:
                        peg_evaluation = "Fair Valued"
                        fundamental_signals += 0.5
                else:
                    peg_evaluation = "Not Available"
                
                # P/B ratio analysis
                if pb_ratio is not None:
                    if pb_ratio < 1:
                        pb_evaluation = "Potentially Undervalued"
                        fundamental_signals += 1
                    elif pb_ratio > 5:
                        pb_evaluation = "Potentially Overvalued"
                    else:
                        pb_evaluation = "Fair Valued"
                        fundamental_signals += 0.5
                else:
                    pb_evaluation = "Not Available"
                
                # ROE analysis
                if roe is not None:
                    if roe > 0.15:
                        roe_evaluation = "Strong"
                        fundamental_signals += 1
                    elif roe < 0.05:
                        roe_evaluation = "Weak"
                    else:
                        roe_evaluation = "Average"
                        fundamental_signals += 0.5
                else:
                    roe_evaluation = "Not Available"
                
                # Determine overall fundamental rating
                if fundamental_signals >= 3:
                    fundamental_rating = "Strong"
                elif fundamental_signals >= 1.5:
                    fundamental_rating = "Moderate"
                else:
                    fundamental_rating = "Weak"
                
                summary["fundamental_analysis"] = {
                    "valuation_ratios": {
                        "pe_ratio": {
                            "value": pe_ratio,
                            "evaluation": pe_evaluation
                        },
                        "peg_ratio": {
                            "value": peg_ratio,
                            "evaluation": peg_evaluation
                        },
                        "ps_ratio": {
                            "value": ps_ratio
                        },
                        "pb_ratio": {
                            "value": pb_ratio,
                            "evaluation": pb_evaluation
                        }
                    },
                    "financial_health": {
                        "de_ratio": {
                            "value": de_ratio,
                            "evaluation": "Low Risk" if de_ratio and de_ratio < 1 else 
                                         "Moderate Risk" if de_ratio and de_ratio < 2 else 
                                         "High Risk" if de_ratio else "Not Available"
                        }
                    },
                    "profitability": {
                        "roe": {
                            "value": roe,
                            "evaluation": roe_evaluation
                        }
                    },
                    "overall_rating": fundamental_rating
                }
                
                # Add industry comparison if available
                if "error" not in industry_comparison:
                    summary["fundamental_analysis"]["industry_comparison"] = {
                        "industry": industry_comparison.get("industry"),
                        "sector": industry_comparison.get("sector"),
                        "peer_count": industry_comparison.get("peer_count"),
                        "key_comparisons": {}
                    }
                    
                    # Add key ratio comparisons
                    for ratio in ["pe_ratio", "ps_ratio", "pb_ratio", "de_ratio", "roe"]:
                        if ratio in industry_comparison.get("comparisons", {}):
                            comparison_data = industry_comparison["comparisons"][ratio]
                            summary["fundamental_analysis"]["industry_comparison"]["key_comparisons"][ratio] = {
                                "stock_value": comparison_data.get("stock_value"),
                                "industry_average": comparison_data.get("industry_average"),
                                "percentage_difference": comparison_data.get("percentage_difference"),
                                "evaluation": comparison_data.get("evaluation")
                            }
            
            # Combined Rating
            # Determine overall investment rating based on both technical and fundamental analysis
            technical_score = 0
            if summary["technical_analysis"]["overall_rating"] == "Bullish":
                technical_score = 3
            elif summary["technical_analysis"]["overall_rating"] == "Neutral":
                technical_score = 2
            else:
                technical_score = 1
                
            fundamental_score = 0
            if "fundamental_analysis" in summary and "overall_rating" in summary["fundamental_analysis"]:
                if summary["fundamental_analysis"]["overall_rating"] == "Strong":
                    fundamental_score = 3
                elif summary["fundamental_analysis"]["overall_rating"] == "Moderate":
                    fundamental_score = 2
                else:
                    fundamental_score = 1
            
            combined_score = (technical_score + fundamental_score) / 2
            
            if combined_score >= 2.5:
                combined_rating = "Strong Buy"
            elif combined_score >= 2:
                combined_rating = "Buy"
            elif combined_score >= 1.5:
                combined_rating = "Hold"
            else:
                combined_rating = "Sell"
                
            summary["combined_rating"] = {
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "combined_score": combined_score,
                "rating": combined_rating
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Error creating combined analysis for {symbol}: {str(e)}"}
    
    @staticmethod
    def format_combined_analysis(analysis: Dict) -> str:
        """
        Format combined analysis results for display
        
        Args:
            analysis: Dictionary with combined analysis results
            
        Returns:
            Formatted string for display
        """
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        result = f"🔍 COMBINED ANALYSIS: {analysis['company_name']} ({analysis['symbol']})\n"
        result += f"Date: {analysis['analysis_date']} | Period: {analysis['period']}\n\n"
        
        # Overall Rating
        result += "🏆 OVERALL RATING\n"
        result += f"Combined Rating: {analysis['combined_rating']['rating']}\n"
        result += f"Technical Score: {analysis['combined_rating']['technical_score']}/3 | "
        result += f"Fundamental Score: {analysis['combined_rating']['fundamental_score']}/3\n\n"
        
        # Technical Analysis
        result += "📈 TECHNICAL ANALYSIS\n"
        result += f"Current Price: ${analysis['technical_analysis']['current_price']:.2f}\n"
        result += f"Price Trend: {analysis['technical_analysis']['price_trend']}\n"
        result += f"RSI (14): {analysis['technical_analysis']['rsi']['value']:.2f} - {analysis['technical_analysis']['rsi']['condition']}\n"
        result += f"MACD: {analysis['technical_analysis']['macd']['condition']}\n"
        result += f"Moving Averages: SMA50 {analysis['technical_analysis']['moving_averages']['relationship']} SMA200\n"
        result += f"Volatility: {analysis['technical_analysis']['volatility']:.2f}%\n"
        result += f"Overall Technical Rating: {analysis['technical_analysis']['overall_rating']}\n\n"
        
        # Fundamental Analysis
        if "fundamental_analysis" in analysis:
            result += "📊 FUNDAMENTAL ANALYSIS\n"
            
            # Valuation Ratios
            if "valuation_ratios" in analysis["fundamental_analysis"]:
                result += "Valuation Ratios:\n"
                
                if "pe_ratio" in analysis["fundamental_analysis"]["valuation_ratios"]:
                    pe = analysis["fundamental_analysis"]["valuation_ratios"]["pe_ratio"]
                    if pe["value"] is not None:
                        result += f"  • P/E Ratio: {pe['value']:.2f} - {pe['evaluation']}\n"
                
                if "peg_ratio" in analysis["fundamental_analysis"]["valuation_ratios"]:
                    peg = analysis["fundamental_analysis"]["valuation_ratios"]["peg_ratio"]
                    if peg["value"] is not None:
                        result += f"  • PEG Ratio: {peg['value']:.2f} - {peg['evaluation']}\n"
                
                if "pb_ratio" in analysis["fundamental_analysis"]["valuation_ratios"]:
                    pb = analysis["fundamental_analysis"]["valuation_ratios"]["pb_ratio"]
                    if pb["value"] is not None:
                        result += f"  • P/B Ratio: {pb['value']:.2f} - {pb.get('evaluation', 'N/A')}\n"
                
                if "ps_ratio" in analysis["fundamental_analysis"]["valuation_ratios"]:
                    ps = analysis["fundamental_analysis"]["valuation_ratios"]["ps_ratio"]
                    if ps["value"] is not None:
                        result += f"  • P/S Ratio: {ps['value']:.2f}\n"
            
            # Financial Health
            if "financial_health" in analysis["fundamental_analysis"]:
                result += "Financial Health:\n"
                
                if "de_ratio" in analysis["fundamental_analysis"]["financial_health"]:
                    de = analysis["fundamental_analysis"]["financial_health"]["de_ratio"]
                    if de["value"] is not None:
                        result += f"  • D/E Ratio: {de['value']:.2f} - {de['evaluation']}\n"
            
            # Profitability
            if "profitability" in analysis["fundamental_analysis"]:
                result += "Profitability:\n"
                
                if "roe" in analysis["fundamental_analysis"]["profitability"]:
                    roe = analysis["fundamental_analysis"]["profitability"]["roe"]
                    if roe["value"] is not None:
                        result += f"  • Return on Equity: {roe['value']*100:.2f}% - {roe['evaluation']}\n"
            
            # Industry Comparison
            if "industry_comparison" in analysis["fundamental_analysis"]:
                ic = analysis["fundamental_analysis"]["industry_comparison"]
                result += f"\nIndustry Comparison ({ic['industry']}, {ic['peer_count']} peers):\n"
                
                if "key_comparisons" in ic:
                    for ratio, data in ic["key_comparisons"].items():
                        if ratio == "pe_ratio":
                            ratio_name = "P/E Ratio"
                        elif ratio == "ps_ratio":
                            ratio_name = "P/S Ratio"
                        elif ratio == "pb_ratio":
                            ratio_name = "P/B Ratio"
                        elif ratio == "de_ratio":
                            ratio_name = "D/E Ratio"
                        elif ratio == "roe":
                            ratio_name = "ROE"
                        else:
                            ratio_name = ratio.replace("_", " ").title()
                            
                        # Format the values based on ratio type
                        if ratio == "roe":
                            stock_value = f"{data['stock_value']*100:.2f}%"
                            industry_avg = f"{data['industry_average']*100:.2f}%"
                        else:
                            stock_value = f"{data['stock_value']:.2f}"
                            industry_avg = f"{data['industry_average']:.2f}"
                            
                        result += f"  • {ratio_name}: {stock_value} vs Industry {industry_avg} "
                        result += f"({data['percentage_difference']:+.1f}%) - {data['evaluation'].title()}\n"
            
            result += f"\nOverall Fundamental Rating: {analysis['fundamental_analysis']['overall_rating']}\n"
        
        # Investment Recommendation
        result += "\n💡 INVESTMENT RECOMMENDATION\n"
        rating = analysis['combined_rating']['rating']
        
        if rating == "Strong Buy":
            result += "Strong Buy: Technical and fundamental indicators are highly favorable. Consider adding to portfolio with appropriate position sizing.\n"
        elif rating == "Buy":
            result += "Buy: Positive indicators suggest potential upside. Consider buying with risk management in place.\n"
        elif rating == "Hold":
            result += "Hold: Mixed signals suggest maintaining current positions but not adding more.\n"
        else:
            result += "Sell: Negative indicators suggest considering reducing exposure or exiting position.\n"
        
        # Risk Assessment
        result += "\n⚠️ RISK ASSESSMENT\n"
        volatility = analysis['technical_analysis']['volatility']
        
        if volatility < 15:
            result += f"Low Volatility ({volatility:.2f}%): Stock shows relatively stable price movements.\n"
        elif volatility < 35:
            result += f"Moderate Volatility ({volatility:.2f}%): Stock shows average price fluctuations.\n"
        else:
            result += f"High Volatility ({volatility:.2f}%): Stock shows significant price swings. Higher risk.\n"
        
        # Add note about visualization
        result += f"\nA comprehensive visualization has been saved to: {analysis['visualization_path']}\n"
        
        return result
