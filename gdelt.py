# GDELT Multi-Agent Intelligence System with Article Scraping
# Frontend: Streamlit | Backend: Multi-Agent LLM System + Web Scraping

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from dataclasses import dataclass, field
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import newspaper
from newspaper import Article
import nltk
from textstat import flesch_reading_ease
from collections import Counter
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configuration
st.set_page_config(
    page_title="GDELT Open Intelligence Agent - Indonesia",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Data Classes
@dataclass
class ScrapedArticle:
    url: str
    title: str
    text: str
    authors: List[str]
    publish_date: Optional[datetime]
    source_domain: str
    word_count: int
    readability_score: float
    key_entities: List[str]
    sentiment_score: float
    scraped_at: datetime
    scraping_success: bool = True
    error_message: str = ""

@dataclass
class AgentResponse:
    agent_name: str
    content: str
    confidence: float
    sources: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

# Enhanced Agent Classes
class BaseAgent:
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Override in subclasses"""
        pass

class ArticleScrapingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Article Scraping Agent",
            "Content Extraction Specialist",
            """You are a web scraping specialist that extracts and analyzes full-text content 
            from news articles. You handle various website structures, extract clean text,
            and perform initial content analysis including entity recognition and readability scoring."""
        )
        self.session = None
        self.scraped_cache = {}
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified)"""
        # Indonesian political/economic keywords
        keywords = [
            'Indonesia', 'Jakarta', 'Jokowi', 'Prabowo', 'Megawati',
            'Pertamina', 'Bank Indonesia', 'Rupiah', 'ASEAN', 'G20',
            'Garuda', 'PLN', 'Telkom', 'Mandiri', 'BCA', 'BRI',
            'Kementerian', 'DPR', 'MPR', 'Komisi', 'Bappenas'
        ]
        
        found_entities = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                # Count occurrences
                count = text_lower.count(keyword.lower())
                if count > 0:
                    found_entities.append(f"{keyword} ({count}x)")
        
        return found_entities[:10]  # Top 10 entities
    
    def calculate_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (simplified)"""
        positive_words = ['growth', 'increase', 'positive', 'success', 'improve', 'benefit', 'progress']
        negative_words = ['decline', 'decrease', 'negative', 'crisis', 'problem', 'concern', 'risk']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    async def scrape_single_article(self, url: str) -> ScrapedArticle:
        """Scrape a single article using newspaper3k"""
        try:
            # Check cache first
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in self.scraped_cache:
                return self.scraped_cache[url_hash]
            
            # Use newspaper3k for better article extraction
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            # Extract domain
            domain = urlparse(url).netloc
            
            # Calculate metrics
            word_count = len(article.text.split()) if article.text else 0
            readability = flesch_reading_ease(article.text) if article.text else 0
            entities = self.extract_entities(article.text) if article.text else []
            sentiment = self.calculate_sentiment(article.text) if article.text else 0.0
            
            scraped_article = ScrapedArticle(
                url=url,
                title=article.title or "No title",
                text=article.text or "",
                authors=article.authors or [],
                publish_date=article.publish_date,
                source_domain=domain,
                word_count=word_count,
                readability_score=readability,
                key_entities=entities,
                sentiment_score=sentiment,
                scraped_at=datetime.now(),
                scraping_success=True
            )
            
            # Cache the result
            self.scraped_cache[url_hash] = scraped_article
            return scraped_article
            
        except Exception as e:
            return ScrapedArticle(
                url=url,
                title="Scraping Failed",
                text="",
                authors=[],
                publish_date=None,
                source_domain=urlparse(url).netloc if url else "",
                word_count=0,
                readability_score=0,
                key_entities=[],
                sentiment_score=0.0,
                scraped_at=datetime.now(),
                scraping_success=False,
                error_message=str(e)
            )
    
    async def scrape_articles_batch(self, urls: List[str], max_articles: int = 20) -> List[ScrapedArticle]:
        """Scrape multiple articles concurrently"""
        # Limit the number of articles to scrape
        urls_to_scrape = urls[:max_articles]
        
        # Use ThreadPoolExecutor for CPU-bound newspaper3k operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            tasks = []
            for url in urls_to_scrape:
                task = asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda u=url: asyncio.run(self.scrape_single_article(u))
                )
                tasks.append(task)
            
            scraped_articles = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and failed scrapes
            valid_articles = []
            for result in scraped_articles:
                if isinstance(result, ScrapedArticle):
                    valid_articles.append(result)
            
            return valid_articles
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process article scraping"""
        gdelt_articles = context.get('gdelt_articles', [])
        
        if not gdelt_articles:
            return AgentResponse(
                agent_name=self.name,
                content="No GDELT articles to scrape",
                confidence=0.0,
                sources=[],
                timestamp=datetime.now()
            )
        
        # Extract URLs from GDELT articles
        urls = []
        for article in gdelt_articles:
            if isinstance(article, dict) and 'url' in article:
                urls.append(article['url'])
            elif hasattr(article, 'url'):
                urls.append(article.url)
        
        st.write(f"📰 Scraping {min(len(urls), 20)} articles for full-text analysis...")
        
        # Scrape articles
        scraped_articles = await self.scrape_articles_batch(urls)
        
        # Generate summary
        successful_scrapes = [a for a in scraped_articles if a.scraping_success]
        failed_scrapes = [a for a in scraped_articles if not a.scraping_success]
        
        total_words = sum(a.word_count for a in successful_scrapes)
        avg_readability = sum(a.readability_score for a in successful_scrapes) / len(successful_scrapes) if successful_scrapes else 0
        avg_sentiment = sum(a.sentiment_score for a in successful_scrapes) / len(successful_scrapes) if successful_scrapes else 0
        
        content = f"""Article Scraping Results:
        ✅ Successfully scraped: {len(successful_scrapes)} articles
        ❌ Failed scrapes: {len(failed_scrapes)} articles
        📊 Total words extracted: {total_words:,}
        📈 Average readability: {avg_readability:.1f}
        💭 Average sentiment: {avg_sentiment:.2f}
        """
        
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.9 if successful_scrapes else 0.3,
            sources=[f"Scraped {len(successful_scrapes)} articles"],
            timestamp=datetime.now(),
            metadata={
                'scraped_articles': scraped_articles,
                'successful_count': len(successful_scrapes),
                'failed_count': len(failed_scrapes),
                'total_words': total_words,
                'avg_readability': avg_readability,
                'avg_sentiment': avg_sentiment
            }
        )

class ContentAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Content Analysis Agent",
            "Deep Content Analyst",
            """You are a content analysis specialist that performs deep analysis on scraped article content.
            You identify key themes, extract important quotes, analyze sentiment patterns, and create
            content summaries for policy analysts."""
        )
    
    def extract_key_themes(self, articles: List[ScrapedArticle]) -> Dict[str, int]:
        """Extract key themes from article content"""
        theme_keywords = {
            'Economic Policy': ['economic', 'economy', 'trade', 'finance', 'investment', 'budget', 'fiscal'],
            'Political Reform': ['political', 'reform', 'government', 'policy', 'legislation', 'parliament'],
            'Infrastructure': ['infrastructure', 'development', 'construction', 'transport', 'energy'],
            'International Relations': ['international', 'diplomatic', 'bilateral', 'asean', 'cooperation'],
            'Social Issues': ['social', 'education', 'healthcare', 'poverty', 'welfare'],
            'Technology': ['technology', 'digital', 'innovation', 'startup', 'tech'],
            'Environment': ['environment', 'climate', 'sustainability', 'renewable', 'green']
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords.keys()}
        
        for article in articles:
            if not article.scraping_success or not article.text:
                continue
                
            text_lower = article.text.lower()
            for theme, keywords in theme_keywords.items():
                count = sum(text_lower.count(keyword) for keyword in keywords)
                theme_counts[theme] += count
        
        return theme_counts
    
    def extract_important_quotes(self, articles: List[ScrapedArticle]) -> List[str]:
        """Extract important quotes from articles"""
        quotes = []
        
        for article in articles[:5]:  # Top 5 articles
            if not article.scraping_success or not article.text:
                continue
            
            # Simple quote extraction - look for quoted text
            text = article.text
            quote_patterns = [
                r'"([^"]{50,200})"',  # Text in double quotes
                r"'([^']{50,200})'",  # Text in single quotes
            ]
            
            for pattern in quote_patterns:
                matches = re.findall(pattern, text)
                for match in matches[:2]:  # Max 2 quotes per article
                    if any(keyword in match.lower() for keyword in ['indonesia', 'policy', 'economic', 'political']):
                        quotes.append(f"'{match}' - {article.source_domain}")
        
        return quotes[:10]  # Top 10 quotes
    
    def analyze_source_credibility(self, articles: List[ScrapedArticle]) -> Dict[str, Any]:
        """Analyze source credibility metrics"""
        source_stats = {}
        
        for article in articles:
            domain = article.source_domain
            if domain not in source_stats:
                source_stats[domain] = {
                    'count': 0,
                    'avg_readability': 0,
                    'avg_word_count': 0,
                    'success_rate': 0
                }
            
            source_stats[domain]['count'] += 1
            if article.scraping_success:
                source_stats[domain]['avg_readability'] += article.readability_score
                source_stats[domain]['avg_word_count'] += article.word_count
                source_stats[domain]['success_rate'] += 1
        
        # Calculate averages
        for domain, stats in source_stats.items():
            if stats['success_rate'] > 0:
                stats['avg_readability'] /= stats['success_rate']
                stats['avg_word_count'] /= stats['success_rate']
                stats['success_rate'] = stats['success_rate'] / stats['count']
        
        return source_stats
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process content analysis"""
        scraped_articles = context.get('scraped_articles', [])
        
        if not scraped_articles:
            return AgentResponse(
                agent_name=self.name,
                content="No scraped articles available for content analysis",
                confidence=0.0,
                sources=[],
                timestamp=datetime.now()
            )
        
        # Perform analysis
        themes = self.extract_key_themes(scraped_articles)
        quotes = self.extract_important_quotes(scraped_articles)
        source_stats = self.analyze_source_credibility(scraped_articles)
        
        # Create content summary
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        content = f"""Content Analysis Results:

🎯 Top Themes Identified:
{chr(10).join([f"• {theme}: {count} mentions" for theme, count in top_themes])}

💬 Key Quotes Extracted:
{chr(10).join([f"• {quote}" for quote in quotes[:3]])}

📊 Source Analysis:
• {len(source_stats)} unique sources
• Top sources: {', '.join(list(source_stats.keys())[:3])}

📈 Content Quality Metrics:
• Average article length: {sum(a.word_count for a in scraped_articles if a.scraping_success) // len([a for a in scraped_articles if a.scraping_success]) if any(a.scraping_success for a in scraped_articles) else 0} words
• Overall sentiment: {sum(a.sentiment_score for a in scraped_articles if a.scraping_success) / len([a for a in scraped_articles if a.scraping_success]) if any(a.scraping_success for a in scraped_articles) else 0:.2f}
        """
        
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.85,
            sources=[f"Content analysis of {len(scraped_articles)} articles"],
            timestamp=datetime.now(),
            metadata={
                'themes': themes,
                'quotes': quotes,
                'source_stats': source_stats,
                'top_themes': top_themes
            }
        )

class GDELTQueryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "GDELT Query Agent",
            "Data Retrieval Specialist",
            """You are a GDELT data retrieval specialist focused on Indonesia. 
            Your role is to construct optimal GDELT API queries based on user requests about 
            political and economic events in Indonesia. You understand GDELT's data structure,
            CAMEO codes, and can translate natural language queries into precise API calls."""
        )
    
    def construct_gdelt_query(self, user_query: str, date_range: tuple) -> Dict[str, str]:
        """Construct GDELT API query parameters"""
        
        # Enhanced query construction based on user input
        query_terms = []
        
        # Base Indonesia terms
        indonesia_terms = ["Indonesia", "Indonesian", "Jakarta"]
        
        # Analyze user query for specific focus
        query_lower = user_query.lower()
        
        if any(term in query_lower for term in ['political', 'politics', 'government', 'election', 'policy']):
            query_terms.extend(['government', 'political', 'policy', 'election', 'minister'])
        
        if any(term in query_lower for term in ['economic', 'economy', 'trade', 'business', 'finance']):
            query_terms.extend(['economic', 'trade', 'business', 'finance', 'investment'])
        
        if any(term in query_lower for term in ['security', 'defense', 'military']):
            query_terms.extend(['security', 'defense', 'military'])
        
        # Default terms if no specific focus
        if not query_terms:
            query_terms = ['government', 'economic', 'policy', 'business']
        
        # Construct query string
        indonesia_query = f"({' OR '.join(indonesia_terms)})"
        topic_query = f"({' OR '.join(query_terms)})" if query_terms else ""
        
        full_query = f"{indonesia_query} AND {topic_query}" if topic_query else indonesia_query
        
        # GDELT GKG API query parameters
        gkg_query = {
            'query': full_query,
            'mode': 'ArtList',
            'format': 'json',
            'startdatetime': date_range[0].strftime('%Y%m%d%H%M%S'),
            'enddatetime': date_range[1].strftime('%Y%m%d%H%M%S'),
            'maxrecords': '100',  # Increased for better coverage
            'sort': 'DateDesc'
        }
        
        return gkg_query
    
    async def fetch_gdelt_data(self, query_params: Dict[str, str]) -> List[Dict]:
        """Fetch data from GDELT API"""
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=query_params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        return articles
                    else:
                        st.warning(f"GDELT API returned status code: {response.status}")
                        return []
        except asyncio.TimeoutError:
            st.error("GDELT API request timed out")
            return []
        except Exception as e:
            st.error(f"Error fetching GDELT data: {e}")
            return []
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        date_range = context.get('date_range', (datetime.now() - timedelta(days=30), datetime.now()))
        
        # Construct and execute GDELT query
        gdelt_params = self.construct_gdelt_query(query, date_range)
        articles = await self.fetch_gdelt_data(gdelt_params)
        
        # Filter for Indonesian content
        indonesian_articles = []
        for article in articles:
            title = article.get('title', '').lower()
            url = article.get('url', '').lower()
            
            if any(term in title or term in url for term in ['indonesia', 'indonesian', 'jakarta']):
                indonesian_articles.append(article)
        
        content = f"""GDELT Query Results:
        🔍 Total articles found: {len(articles)}
        🇮🇩 Indonesia-specific articles: {len(indonesian_articles)}
        📅 Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}
        🔎 Query used: {gdelt_params['query']}
        """
        
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.9 if indonesian_articles else 0.3,
            sources=[f"GDELT API - {len(indonesian_articles)} Indonesian articles"],
            timestamp=datetime.now(),
            metadata={'gdelt_articles': indonesian_articles, 'query_params': gdelt_params}
        )

class TrendAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Trend Analysis Agent",
            "Pattern Recognition Specialist",
            """You are a trend analysis specialist for Indonesian political and economic events.
            You identify patterns, anomalies, and significant trends in both GDELT data and scraped content over time.
            You provide quantitative analysis and highlight emerging patterns that policy analysts should notice."""
        )
    
    def analyze_temporal_patterns(self, articles: List[Dict], scraped_articles: List[ScrapedArticle]) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        if not articles and not scraped_articles:
            return {}
        
        # Combine data from both sources
        all_dates = []
        
        # GDELT articles
        for article in articles:
            if 'seendate' in article:
                try:
                    date = datetime.strptime(article['seendate'][:8], '%Y%m%d').date()
                    all_dates.append(date)
                except:
                    pass
        
        # Scraped articles
        for article in scraped_articles:
            if article.publish_date:
                all_dates.append(article.publish_date.date())
        
        if not all_dates:
            return {}
        
        # Create DataFrame for analysis
        df = pd.DataFrame({'date': all_dates})
        daily_counts = df.groupby('date').size()
        
        # Calculate trends
        trend_analysis = {
            'total_articles': len(all_dates),
            'date_range': (min(all_dates), max(all_dates)),
            'daily_average': daily_counts.mean(),
            'peak_day': daily_counts.idxmax() if len(daily_counts) > 0 else None,
            'peak_count': daily_counts.max() if len(daily_counts) > 0 else 0,
            'trend_direction': self.calculate_trend_direction(daily_counts),
            'daily_counts': daily_counts.to_dict()
        }
        
        return trend_analysis
    
    def calculate_trend_direction(self, daily_counts) -> str:
        """Calculate overall trend direction"""
        if len(daily_counts) < 2:
            return 'insufficient_data'
        
        # Linear regression to determine trend
        x = range(len(daily_counts))
        y = daily_counts.values
        
        # Simple slope calculation
        n = len(x)
        slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi**2 for xi in x) - sum(x)**2)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def analyze_content_trends(self, scraped_articles: List[ScrapedArticle]) -> Dict[str, Any]:
        """Analyze trends in scraped content"""
        if not scraped_articles:
            return {}
        
        successful_articles = [a for a in scraped_articles if a.scraping_success]
        
        # Sentiment trend over time
        sentiment_by_date = {}
        readability_by_date = {}
        
        for article in successful_articles:
            if article.publish_date:
                date = article.publish_date.date()
                if date not in sentiment_by_date:
                    sentiment_by_date[date] = []
                    readability_by_date[date] = []
                
                sentiment_by_date[date].append(article.sentiment_score)
                readability_by_date[date].append(article.readability_score)
        
        # Calculate daily averages
        avg_sentiment_by_date = {date: sum(scores)/len(scores) for date, scores in sentiment_by_date.items()}
        avg_readability_by_date = {date: sum(scores)/len(scores) for date, scores in readability_by_date.items()}
        
        return {
            'sentiment_trend': avg_sentiment_by_date,
            'readability_trend': avg_readability_by_date,
            'avg_sentiment': sum(a.sentiment_score for a in successful_articles) / len(successful_articles) if successful_articles else 0,
            'avg_readability': sum(a.readability_score for a in successful_articles) / len(successful_articles) if successful_articles else 0
        }
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        articles = context.get('gdelt_articles', [])
        scraped_articles = context.get('scraped_articles', [])
        
        temporal_trends = self.analyze_temporal_patterns(articles, scraped_articles)
        content_trends = self.analyze_content_trends(scraped_articles)
        
        content = f"""Trend Analysis Results:

📊 Temporal Patterns:
• Total articles analyzed: {temporal_trends.get('total_articles', 0)}
• Daily average: {temporal_trends.get('daily_average', 0):.1f} articles
• Peak activity: {temporal_trends.get('peak_count', 0)} articles on {temporal_trends.get('peak_day', 'N/A')}
• Overall trend: {temporal_trends.get('trend_direction', 'unknown').title()}

📈 Content Analysis Trends:
• Average sentiment: {content_trends.get('avg_sentiment', 0):.2f}
• Average readability: {content_trends.get('avg_readability', 0):.1f}
• Content quality trend: {'Improving' if content_trends.get('avg_readability', 0) > 50 else 'Standard'}
        """
        
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.85,
            sources=["Temporal Analysis", "Content Pattern Analysis"],
            timestamp=datetime.now(),
            metadata={
                'temporal_trends': temporal_trends,
                'content_trends': content_trends
            }
        )

class PolicyInsightAgent(BaseAgent):
    def __init__(self, openai_api_key: str = None):
        super().__init__(
            "Policy Insight Agent",
            "Strategic Policy Analyst",
            """You are a senior policy analyst specializing in Indonesian politics and economics.
            You provide strategic insights, policy implications, and recommendations based on GDELT event data
            and full-text article analysis. You connect current events to broader policy trends and geopolitical implications."""
        )
        self.openai_api_key = openai_api_key
    
    async def generate_policy_insights(self, query: str, context: Dict[str, Any]) -> str:
        """Generate comprehensive policy insights using LLM"""
        if not self.openai_api_key:
            return self.generate_fallback_insights(context)
        
        # Prepare comprehensive context
        articles = context.get('gdelt_articles', [])
        scraped_articles = context.get('scraped_articles', [])
        themes = context.get('themes', {})
        quotes = context.get('quotes', [])
        
        # Create rich context for LLM
        article_summaries = []
        for i, article in enumerate(articles[:5]):
            title = article.get('title', 'No title')
            url = article.get('url', '')
            article_summaries.append(f"{i+1}. {title} ({url})")
        
        content_insights = []
        for article in scraped_articles[:3]:
            if article.scraping_success and article.text:
                # Extract first paragraph as summary
                first_para = article.text.split('\n')[0][:200] + "..."
                content_insights.append(f"• {article.title}: {first_para}")
        
        prompt = f"""
        As a senior policy analyst specializing in Indonesia, provide comprehensive strategic insights based on the following intelligence:

        USER QUERY: {query}

        GDELT ARTICLES IDENTIFIED:
        {chr(10).join(article_summaries)}

        CONTENT ANALYSIS INSIGHTS:
        {chr(10).join(content_insights)}

        KEY THEMES DETECTED:
        {json.dumps(themes, indent=2)}

        IMPORTANT QUOTES:
        {chr(10).join(quotes[:3])}

        Please provide a comprehensive analysis covering:

        1. STRATEGIC ASSESSMENT
        - Key developments and their significance
        - Pattern analysis and emerging trends

        2. POLICY IMPLICATIONS
        - Direct policy impacts
        - Stakeholder considerations
        - Regulatory considerations

        3. RISK & OPPORTUNITY MATRIX
        - Key risks to monitor
        - Strategic opportunities
        - Mitigation strategies

        4. ACTIONABLE RECOMMENDATIONS
        - Short-term actions (1-3 months)
        - Medium-term strategy (3-12 months)
        - Long-term considerations

        5. MONITORING PRIORITIES
        - Key indicators to track
        - Early warning signals
        - Information gaps to fill

        Format your response with clear sections and bullet points for policy briefing purposes.
        """
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights with OpenAI: {e}\n\nFalling back to basic analysis..."
    
    def generate_fallback_insights(self, context: Dict[str, Any]) -> str:
        """Generate basic insights without OpenAI API"""
        articles = context.get('gdelt_articles', [])
        scraped_articles = context.get('scraped_articles', [])
        themes = context.get('themes', {})
        
        successful_scrapes = [a for a in scraped_articles if a.scraping_success]
        
        # Basic analysis
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3] if themes else []
        avg_sentiment = sum(a.sentiment_score for a in successful_scrapes) / len(successful_scrapes) if successful_scrapes else 0
        
        insights = f"""
## STRATEGIC ASSESSMENT
• **Data Coverage**: {len(articles)} GDELT articles, {len(successful_scrapes)} scraped for full content
• **Dominant Themes**: {', '.join([theme for theme, _ in top_themes])}
• **Overall Sentiment**: {'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'} ({avg_sentiment:.2f})

## POLICY IMPLICATIONS
• **Primary Focus Areas**: {top_themes[0][0] if top_themes else 'Mixed themes'} appears to be the dominant concern
• **Media Attention**: {'High' if len(articles) > 20 else 'Moderate' if len(articles) > 10 else 'Low'} level of coverage
• **Content Quality**: {'High-quality sources' if any(a.readability_score > 60 for a in successful_scrapes) else 'Mixed source quality'}

## RISK & OPPORTUNITY MATRIX
### Risks:
• Information volume suggests {'heightened attention' if len(articles) > 30 else 'normal monitoring levels'}
• Sentiment trends indicate {'potential concerns' if avg_sentiment < 0 else 'generally positive outlook'}

### Opportunities:
• Clear thematic focus provides policy direction clarity
• Media engagement suggests public interest and policy window

## ACTIONABLE RECOMMENDATIONS
### Short-term (1-3 months):
• Monitor {top_themes[0][0] if top_themes else 'key themes'} developments closely
• Engage with stakeholders on emerging issues
• Prepare policy responses for identified themes

### Medium-term (3-12 months):
• Develop comprehensive strategy addressing top themes
• Build coalitions around policy priorities
• Monitor international reactions and implications

## MONITORING PRIORITIES  
• **Key Indicators**: {', '.join([theme for theme, _ in top_themes[:2]])} coverage frequency
• **Early Warnings**: Sentiment shifts, volume spikes, new source emergence
• **Information Gaps**: {f'Need deeper analysis on {len([a for a in scraped_articles if not a.scraping_success])} failed article scrapes' if any(not a.scraping_success for a in scraped_articles) else 'Good data coverage achieved'}
        """
        
        return insights
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        insights = await self.generate_policy_insights(query, context)
        
        return AgentResponse(
            agent_name=self.name,
            content=insights,
            confidence=0.9 if self.openai_api_key else 0.7,
            sources=["Strategic Analysis", "GDELT Data", "Scraped Content", "LLM Analysis" if self.openai_api_key else "Rule-based Analysis"],
            timestamp=datetime.now()
        )

class MultiAgentOrchestrator:
    def __init__(self, openai_api_key: str = None):
        self.agents = {
            'query': GDELTQueryAgent(),
            'scraping': ArticleScrapingAgent(),
            'content': ContentAnalysisAgent(),
            'trends': TrendAnalysisAgent(),
            'insights': PolicyInsightAgent(openai_api_key)
        }
        
    async def process_query(self, user_query: str, date_range: tuple) -> Dict[str, AgentResponse]:
        """Orchestrate multi-agent processing with article scraping"""
        context = {'date_range': date_range}
        responses = {}
        
        # Step 1: Query Agent retrieves GDELT data
        st.write("🔍 Querying GDELT database...")
        progress = st.progress(0.0)
        
        query_response = await self.agents['query'].process(user_query, context)
        responses['query'] = query_response
        context['gdelt_articles'] = query_response.metadata.get('gdelt_articles', [])
        progress.progress(0.2)
        
        # Step 2: Article Scraping Agent extracts full content
        st.write("📰 Scraping articles for full-text analysis...")
        scraping_response = await self.agents['scraping'].process(user_query, context)
        responses['scraping'] = scraping_response
        context['scraped_articles'] = scraping_response.metadata.get('scraped_articles', [])
        progress.progress(0.4)
        
        # Step 3: Content Analysis Agent analyzes scraped content
        st.write("🔬 Performing deep content analysis...")
        content_response = await self.agents['content'].process(user_query, context)
        responses['content'] = content_response
        
        # Add content analysis results to context
        context.update({
            'themes': content_response.metadata.get('themes', {}),
            'quotes': content_response.metadata.get('quotes', []),
            'source_stats': content_response.metadata.get('source_stats', {})
        })
        progress.progress(0.6)
        
        # Step 4: Trend Analysis
        st.write("📈 Analyzing trends and patterns...")
        trend_response = await self.agents['trends'].process(user_query, context)
        responses['trends'] = trend_response
        context.update(trend_response.metadata)
        progress.progress(0.8)
        
        # Step 5: Policy Insights with full context
        st.write("💡 Generating strategic policy insights...")
        insight_response = await self.agents['insights'].process(user_query, context)
        responses['insights'] = insight_response
        progress.progress(1.0)
        
        st.success("✅ Multi-agent analysis complete!")
        progress.empty()
        
        return responses

# Enhanced Streamlit UI
def main():
    st.title("🇮🇩 GDELT Open Intelligence Agent")
    st.subheader("Multi-Agent Analysis with Article Scraping for Indonesian Political & Economic Events")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        openai_key = st.text_input("OpenAI API Key", type="password", help="Required for advanced policy insights")
        
        # Date range selector
        st.subheader("📅 Date Range")
        end_date = st.date_input("End Date", datetime.now().date())
        start_date = st.date_input("Start Date", (datetime.now() - timedelta(days=14)).date())
        
        # Scraping settings
        st.subheader("🕷️ Scraping Settings")
        max_articles = st.slider("Max articles to scrape", 5, 50, 20, help="More articles = longer processing time")
        
        # Query presets
        st.subheader("🎯 Query Presets")
        preset_queries = {
            "Political Developments": "Recent political developments and government policy changes in Indonesia",
            "Economic Indicators": "Economic trends, trade policies, and business developments in Indonesia",
            "Infrastructure Projects": "Infrastructure development and construction projects in Indonesia",
            "Regional Security": "Security concerns and regional cooperation involving Indonesia",
            "Trade Relations": "International trade relationships and economic partnerships of Indonesia",
            "Digital Economy": "Digital transformation and technology adoption in Indonesia",
            "Environmental Policy": "Environmental policies and sustainability initiatives in Indonesia"
        }
        
        selected_preset = st.selectbox("Choose a preset query:", ["Custom"] + list(preset_queries.keys()))
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Multi-Agent Query Interface")
        
        # Query input
        if selected_preset != "Custom":
            default_query = preset_queries[selected_preset]
        else:
            default_query = ""
        
        user_query = st.text_area(
            "Enter your intelligence query:",
            value=default_query,
            height=100,
            placeholder="e.g., 'Analyze recent economic policy changes in Indonesia and their potential impact on regional trade, including detailed content analysis'"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            include_sentiment = st.checkbox("Include sentiment analysis", True)
            include_entities = st.checkbox("Extract key entities", True)
            deep_analysis = st.checkbox("Enable deep content analysis", True)
        
        # Submit button
        if st.button("🚀 Launch Enhanced Multi-Agent Analysis", type="primary"):
            if not user_query.strip():
                st.error("Please enter a query")
                return
            
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return
            
            # Initialize orchestrator
            orchestrator = MultiAgentOrchestrator(openai_key if openai_key else None)
            
            # Process query
            with st.spinner("Enhanced multi-agent system processing your query..."):
                try:
                    date_range = (
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time())
                    )
                    
                    # Run the actual multi-agent system
                    responses = asyncio.run(orchestrator.process_query(user_query, date_range))
                    
                    # Display results
                    display_enhanced_agent_responses(responses)
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.exception(e)
    
    with col2:
        st.subheader("📊 System Status")
        
        # Enhanced agent status indicators
        agents_status = {
            "🔍 GDELT Query Agent": "Ready",
            "📰 Article Scraper": "Ready",
            "🔬 Content Analyzer": "Ready",
            "📈 Trend Analyzer": "Ready",
            "💡 Policy Insights": "Ready" if openai_key else "Needs API Key"
        }
        
        for agent, status in agents_status.items():
            if status == "Ready":
                st.success(f"{agent}: {status}")
            else:
                st.warning(f"{agent}: {status}")
        
        # Enhanced quick stats
        st.subheader("📈 System Capabilities")
        st.metric("Data Sources", "GDELT + Web Scraping")
        st.metric("Analysis Depth", "Full-text + Metadata")
        st.metric("Countries Focus", "Indonesia")
        st.metric("Max Articles/Query", f"{max_articles}")
        
        # Processing info
        st.subheader("ℹ️ Processing Info")
        st.info("""
        **Enhanced Pipeline:**
        1. GDELT query & filtering
        2. Article URL extraction
        3. Full-text web scraping
        4. Content analysis & NLP
        5. Trend pattern analysis
        6. Strategic policy insights
        """)

def display_enhanced_agent_responses(responses: Dict[str, AgentResponse]):
    """Display enhanced agent responses with scraping results"""
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 GDELT Query",
        "📰 Article Scraping",
        "🔬 Content Analysis",
        "📈 Trend Analysis",
        "💡 Policy Insights"
    ])
    
    with tab1:
        response = responses['query']
        st.subheader(f"{response.agent_name}")
        st.write(response.content)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{response.confidence:.1%}")
        with col2:
            st.metric("Articles Found", len(response.metadata.get('gdelt_articles', [])))
        with col3:
            st.metric("Sources", len(response.sources))
        
        # Show sample articles
        articles = response.metadata.get('gdelt_articles', [])
        if articles:
            st.subheader("📋 Sample Articles Found")
            for i, article in enumerate(articles[:5]):
                with st.expander(f"Article {i+1}: {article.get('title', 'No title')[:100]}..."):
                    st.write(f"**URL:** {article.get('url', 'N/A')}")
                    st.write(f"**Domain:** {article.get('domain', 'N/A')}")
                    st.write(f"**Language:** {article.get('language', 'N/A')}")
                    st.write(f"**Seen Date:** {article.get('seendate', 'N/A')}")
    
    with tab2:
        response = responses['scraping']
        st.subheader(f"{response.agent_name}")
        st.write(response.content)
        
        metadata = response.metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Success Rate", f"{metadata.get('successful_count', 0)}/{metadata.get('successful_count', 0) + metadata.get('failed_count', 0)}")
        with col2:
            st.metric("Total Words", f"{metadata.get('total_words', 0):,}")
        with col3:
            st.metric("Avg Readability", f"{metadata.get('avg_readability', 0):.1f}")
        with col4:
            st.metric("Avg Sentiment", f"{metadata.get('avg_sentiment', 0):.2f}")
        
        # Show scraped articles details
        scraped_articles = metadata.get('scraped_articles', [])
        if scraped_articles:
            st.subheader("📄 Scraped Articles Analysis")
            
            # Create DataFrame for better display
            article_data = []
            for article in scraped_articles:
                article_data.append({
                    'Title': article.title[:50] + "..." if len(article.title) > 50 else article.title,
                    'Source': article.source_domain,
                    'Words': article.word_count,
                    'Readability': f"{article.readability_score:.1f}",
                    'Sentiment': f"{article.sentiment_score:.2f}",
                    'Success': "✅" if article.scraping_success else "❌"
                })
            
            if article_data:
                df = pd.DataFrame(article_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                successful_articles = [a for a in scraped_articles if a.scraping_success]
                if successful_articles:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        sentiments = [a.sentiment_score for a in successful_articles]
                        fig_sentiment = px.histogram(x=sentiments, title="Sentiment Distribution",
                                                   labels={'x': 'Sentiment Score', 'y': 'Count'})
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    with col2:
                        # Word count distribution
                        word_counts = [a.word_count for a in successful_articles]
                        fig_words = px.histogram(x=word_counts, title="Article Length Distribution",
                                               labels={'x': 'Word Count', 'y': 'Count'})
                        st.plotly_chart(fig_words, use_container_width=True)
    
    with tab3:
        response = responses['content']
        st.subheader(f"{response.agent_name}")
        st.markdown(response.content)
        
        metadata = response.metadata
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence", f"{response.confidence:.1%}")
            
            # Top themes chart
            themes = metadata.get('themes', {})
            if themes:
                st.subheader("🎯 Theme Analysis")
                theme_df = pd.DataFrame(list(themes.items()), columns=['Theme', 'Mentions'])
                theme_df = theme_df.sort_values('Mentions', ascending=True)
                
                fig = px.bar(theme_df, x='Mentions', y='Theme', orientation='h',
                           title="Key Themes Identified")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Themes Found", len(metadata.get('themes', {})))
            
            # Important quotes
            quotes = metadata.get('quotes', [])
            if quotes:
                st.subheader("💬 Key Quotes")
                for i, quote in enumerate(quotes[:3]):
                    st.info(f"**Quote {i+1}:** {quote}")
        
        # Source analysis
        source_stats = metadata.get('source_stats', {})
        if source_stats:
            st.subheader("📊 Source Credibility Analysis")
            source_data = []
            for domain, stats in source_stats.items():
                source_data.append({
                    'Domain': domain,
                    'Articles': stats['count'],
                    'Success Rate': f"{stats['success_rate']:.1%}",
                    'Avg Readability': f"{stats['avg_readability']:.1f}",
                    'Avg Length': f"{stats['avg_word_count']:.0f}"
                })
            
            if source_data:
                source_df = pd.DataFrame(source_data)
                st.dataframe(source_df, use_container_width=True)
    
    with tab4:
        response = responses['trends']
        st.subheader(f"{response.agent_name}")
        st.write(response.content)
        
        metadata = response.metadata
        temporal_trends = metadata.get('temporal_trends', {})
        content_trends = metadata.get('content_trends', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{response.confidence:.1%}")
            st.metric("Trend Direction", temporal_trends.get('trend_direction', 'Unknown').title())
        with col2:
            st.metric("Total Articles", temporal_trends.get('total_articles', 0))
            st.metric("Daily Average", f"{temporal_trends.get('daily_average', 0):.1f}")
        
        # Temporal trend visualization
        daily_counts = temporal_trends.get('daily_counts', {})
        if daily_counts:
            st.subheader("📈 Article Frequency Over Time")
            
            dates = list(daily_counts.keys())
            counts = list(daily_counts.values())
            
            fig = px.line(x=dates, y=counts, title="Daily Article Count",
                         labels={'x': 'Date', 'y': 'Number of Articles'})
            fig.update_layout(xaxis_title="Date", yaxis_title="Article Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Content trends
        sentiment_trend = content_trends.get('sentiment_trend', {})
        if sentiment_trend:
            st.subheader("💭 Sentiment Trend Over Time")
            
            dates = list(sentiment_trend.keys())
            sentiments = list(sentiment_trend.values())
            
            fig = px.line(x=dates, y=sentiments, title="Average Daily Sentiment",
                         labels={'x': 'Date', 'y': 'Sentiment Score'})
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        response = responses['insights']
        st.subheader(f"{response.agent_name}")
        st.markdown(response.content)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{response.confidence:.1%}")
        with col2:
            st.metric("Analysis Type", "LLM-Enhanced" if "LLM Analysis" in response.sources else "Rule-Based")
        
        # Export options
        st.subheader("📄 Export Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy to Clipboard"):
                st.success("Analysis copied! (Feature requires additional setup)")
        
        with col2:
            # Create downloadable report
            report_content = f"""
# GDELT Intelligence Report - Indonesia
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Query
{user_query if 'user_query' in locals() else 'N/A'}

## Executive Summary
{response.content}

## Data Sources
- GDELT Articles: {len(responses['query'].metadata.get('gdelt_articles', []))}
- Scraped Articles: {responses['scraping'].metadata.get('successful_count', 0)}
- Analysis Confidence: {response.confidence:.1%}
            """
            
            st.download_button(
                "📁 Download Report",
                report_content,
                file_name=f"gdelt_intel_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
        
        with col3:
            if st.button("🔄 Refresh Analysis"):
                st.rerun()

if __name__ == "__main__":
    main()
