# Cryptocurrency Analysis Multi-Agent System

## Project Overview

This project implements a sophisticated multi-agent system for cryptocurrency analysis and prediction. The system collects and processes data from various sources including cryptocurrency market data, news articles, social media, and forum discussions. Using sentiment analysis and machine learning techniques, the system generates insights and predictions about cryptocurrency price movements.

## System Architecture

The system follows a hierarchical multi-agent architecture where specialized agents handle different aspects of data collection, processing, and analysis. The agents communicate with each other through message passing using the SPADE framework.

![Agents Workflow](Agents-Workflow.png)

### Agent Hierarchy

1. **Global Orchestrator**: The top-level agent that coordinates all other orchestrators
2. **Specialized Orchestrators**: 
   - Crypto Orchestrator - Manages all crypto data collection agents
   - News Orchestrator - Manages all news and social media data collection agents
3. **Data Collection Agents**:
   - Crypto Price Agent - Real-time cryptocurrency prices
   - Detailed Crypto Data Agent - Historical crypto price and volume data
   - Fear & Greed Index Agent - Market sentiment indicators
   - Reddit Agent - Posts from cryptocurrency subreddits
   - Articles Agent - News articles from crypto news sources
   - Forum Agent - Discussions from crypto forums
4. **Analysis Agents**:
   - Sentiment Analysis Agent - Analyzes text content for sentiment
   - (Future) Prediction Models Agent - Generates price predictions

## Agent Responsibilities

### Orchestrator Agents

- **Global Orchestrator**: Initiates and coordinates all system activities, receives final processed data
- **Crypto Orchestrator**: Manages all cryptocurrency data collection agents and forwards the results
- **News Orchestrator**: Manages all news and social media agents and forwards data to sentiment analysis

### Data Collection Agents

- **Crypto Price Agent**: Collects real-time price data for top cryptocurrencies every 10 minutes
- **Detailed Crypto Data Agent**: Gathers daily OHLCV (Open, High, Low, Close, Volume) historical data
- **Fear & Greed Index Agent**: Retrieves the Bitcoin Fear & Greed Index daily
- **Reddit Agent**: Scrapes posts from cryptocurrency subreddits hourly
- **Articles Agent**: Retrieves articles from crypto news sources hourly
- **Forum Agent**: Collects posts from crypto forums daily

### Analysis Agents

- **Sentiment Analysis Agent**: Performs sentiment analysis on text data from news, Reddit, and forums
- **Prediction Model**: Uses machine learning to forecast cryptocurrency prices (Random Forest, XGBoost, LSTM)

## Agent Communication

Agents communicate using a well-defined set of message performatives (message types) that represent different intents:

### Communication Performatives

- **start_agent**: Initiates an agent's data collection behaviors. Sent from orchestrators to data collectors.
- **job_finished**: Signals that a data collection job has completed. Includes information about which database collection was updated.
- **new_data_to_analyze**: Notifies the Sentiment Analysis agent that new text data is available for processing. Includes metadata about the source.
- **new_data_available**: Informs the Global Orchestrator that new analyzed data is available for prediction models.

### Communication Patterns

1. **Initialization Sequence**:
   - Global Orchestrator sends `start_agent` to specialized orchestrators
   - Specialized orchestrators send `start_agent` to their data collection agents
   - Data collection agents begin their scheduled operations

2. **Data Collection Cycle**:
   - Data collection agents periodically fetch and process data
   - When complete, they send `job_finished` to their orchestrator
   - News Orchestrator forwards text data with `new_data_to_analyze` to Sentiment Analysis
   - After analysis, Sentiment Analysis sends `new_data_available` to Global Orchestrator

3. **Data Payload**:
   - Agents include metadata about data sources in their messages
   - Payload includes database collection names and provider agent identifiers
   - This metadata ensures proper data tracking and processing flow

## Data Collection and Processing

### Cryptocurrency Data

The system collects several types of cryptocurrency market data:
- Real-time prices for top coins (BTC, ETH, XRP, BNB, SOL, DOGE, TRX, ADA)
- Historical daily price data with OHLCV values
- Fear & Greed Index which measures market sentiment

### News and Social Media Data

The system gathers text data from multiple sources:
- Reddit posts from cryptocurrency subreddits
- News articles from over 30 crypto news sources via RSS feeds
- Forum posts from CryptoPanic API

### Data Storage

All collected data is stored in a MongoDB database with these collections:
- crypto-price: Real-time price snapshots
- detailed-crypto-data: Historical OHLCV data
- crypto-fear-greed: Daily Fear & Greed Index values
- reddit: Posts from cryptocurrency subreddits
- articles: News articles from crypto sources
- forum: Posts from crypto forums

## Analysis and Prediction

### Sentiment Analysis

Text data undergoes sentiment analysis using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner):
- Each text entry receives a compound score from -1 (very negative) to +1 (very positive)
- Sentiment labels (Positive, Neutral, Negative) are assigned based on the scores
- Results are stored alongside original data in MongoDB

### Price Prediction

The system includes a sophisticated prediction module that:
- Combines price data, technical indicators, and sentiment scores
- Trains multiple models (Random Forest, XGBoost, LSTM neural networks)
- Selects the best performing model for each cryptocurrency
- Generates price predictions for the next 7 days
- Creates market overview reports with sentiment analysis

## Installation and Setup

1. Clone this repository
2. Install required packages
3. Copy `.env.example` to `.env` and fill in your API keys and credentials
4. Start a MongoDB server
5. Run the system:
   ```
   python main.py
   ```
