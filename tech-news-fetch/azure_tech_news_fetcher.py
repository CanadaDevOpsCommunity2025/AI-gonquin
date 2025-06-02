#!/usr/bin/env python3
"""
Tech News Data Fetcher - Modified for Azure Blob Storage
Collects articles from various tech news sources and saves to Azure Blob Storage
for Azure ML training pipeline
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
import random
from io import StringIO

# Azure imports
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureBlobTechNewsFetcher:
    def __init__(self, 
                 connection_string: str = None,
                 container_name: str = "tech-news-data",
                 data_dir: str = "data"):
        """
        Initialize with Azure Blob Storage connection
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Blob container name
            data_dir: Local data directory (for fallback)
        """
        self.data_dir = data_dir
        self.container_name = container_name
        
        # Initialize Azure Blob Storage client
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = self.blob_service_client.get_container_client(container_name)
            self._ensure_container_exists()
        else:
            # Try to get from environment variables
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                self.container_client = self.blob_service_client.get_container_client(container_name)
                self._ensure_container_exists()
            else:
                logger.warning("No Azure Storage connection string provided. Will save locally only.")
                self.blob_service_client = None
                self.container_client = None
        
        # Create local directories as fallback
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.llama_dir = os.path.join(data_dir, "llama_training")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.llama_dir, exist_ok=True)
        
        # Tech news RSS feeds
        self.rss_feeds = {
            "techcrunch": "https://techcrunch.com/feed/",
            "ars_technica": "https://feeds.arstechnica.com/arstechnica/index",
            "the_verge": "https://www.theverge.com/rss/index.xml",
            "wired": "https://www.wired.com/feed/rss",
            "zdnet": "https://www.zdnet.com/news/rss.xml",
            "engadget": "https://www.engadget.com/rss.xml",
            "venturebeat": "https://venturebeat.com/feed/",
            "tech_radar": "https://www.techradar.com/rss",
            "hacker_news": "https://hnrss.org/frontpage"
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # LLaMA-specific system prompts
        self.system_prompts = [
            "You are a helpful AI assistant specializing in technology news and discussions. Provide informative, accurate, and engaging responses about the latest tech developments.",
            "You are an expert technology journalist AI. Share the latest tech news and insights in a conversational and accessible way.",
            "You are a knowledgeable tech enthusiast AI assistant. Help users stay updated with technology trends and developments.",
            "You are a professional tech news curator AI. Provide concise, informative updates about technology and innovation."
        ]

    def _ensure_container_exists(self):
        """Ensure the blob container exists"""
        if not self.container_client:
            return
            
        try:
            self.container_client.create_container()
            logger.info(f"Created container: {self.container_name}")
        except ResourceExistsError:
            logger.info(f"Container {self.container_name} already exists")
        except Exception as e:
            logger.error(f"Error creating container: {str(e)}")

    def upload_to_blob(self, data: any, blob_name: str, as_jsonl: bool = False) -> bool:
        """Upload data to Azure Blob Storage"""
        if not self.container_client:
            logger.warning("No Azure Blob Storage client available")
            return False
            
        try:
            if as_jsonl:
                # For JSONL data (list of strings)
                content = '\n'.join(data)
            else:
                # For JSON data
                content = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Upload to blob
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(content, overwrite=True, encoding='utf-8')
            
            logger.info(f"Successfully uploaded {blob_name} to Azure Blob Storage")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading {blob_name} to blob storage: {str(e)}")
            return False

    def save_data(self, data: List, filename: str, directory: str = None, as_jsonl: bool = False, blob_folder: str = "processed"):
        """Save data locally and to Azure Blob Storage"""
        if directory is None:
            directory = self.processed_dir
            
        # Save locally (fallback)
        local_filepath = os.path.join(directory, filename)
        
        try:
            if as_jsonl:
                with open(local_filepath, 'w', encoding='utf-8') as f:
                    for line in data:
                        f.write(line + '\n')
            else:
                with open(local_filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(data)} items to {local_filepath}")
        except Exception as e:
            logger.error(f"Error saving locally: {str(e)}")
        
        # Save to Azure Blob Storage
        blob_name = f"{blob_folder}/{filename}"
        upload_success = self.upload_to_blob(data, blob_name, as_jsonl)
        
        if not upload_success:
            logger.warning(f"Failed to upload {filename} to blob storage, but saved locally")

    def fetch_rss_articles(self, source: str, feed_url: str, max_articles: int = 50) -> List[Dict]:
        """Fetch articles from RSS feed"""
        try:
            logger.info(f"Fetching articles from {source}")
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    "source": source,
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", "").strip(),
                    "published": entry.get("published", ""),
                    "published_parsed": entry.get("published_parsed", None),
                    "fetched_at": datetime.now().isoformat(),
                    "tags": entry.get("tags", [])
                }
                
                # Try to get full content
                full_content = self.extract_article_content(article["link"])
                if full_content:
                    article["content"] = full_content
                else:
                    article["content"] = article["summary"]
                
                # Extract categories/topics
                article["categories"] = self.extract_categories(article)
                
                if article["title"] and article["content"]:
                    articles.append(article)
                
                # Be respectful with requests
                time.sleep(0.5)
            
            logger.info(f"Fetched {len(articles)} articles from {source}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {source}: {str(e)}")
            return []

    def extract_article_content(self, url: str) -> Optional[str]:
        """Extract full article content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
                tag.decompose()
            
            # Try common content selectors
            content_selectors = [
                "article",
                ".article-content",
                ".entry-content", 
                ".post-content",
                ".content",
                ".story-body",
                "main"
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(strip=True)
                    break
            
            if not content:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            # Return content if it's substantial
            return content if len(content) > 150 else None
            
        except Exception as e:
            logger.warning(f"Could not extract content from {url}: {str(e)}")
            return None

    def extract_categories(self, article: Dict) -> List[str]:
        """Extract categories/topics from article"""
        categories = []
        content_lower = (article.get("title", "") + " " + article.get("content", "")).lower()
        
        # Technology categories
        tech_categories = {
            "ai": ["ai", "artificial intelligence", "machine learning", "deep learning", "neural network"],
            "blockchain": ["blockchain", "cryptocurrency", "bitcoin", "ethereum", "crypto", "nft"],
            "mobile": ["iphone", "android", "mobile", "smartphone", "app store", "google play"],
            "cloud": ["cloud computing", "aws", "azure", "google cloud", "saas", "paas"],
            "cybersecurity": ["cybersecurity", "security", "hack", "breach", "malware", "vulnerability"],
            "startup": ["startup", "funding", "investment", "venture capital", "ipo", "acquisition"],
            "big_tech": ["apple", "google", "microsoft", "amazon", "meta", "facebook", "tesla"],
            "gaming": ["gaming", "game", "xbox", "playstation", "nintendo", "esports"],
            "social_media": ["social media", "twitter", "instagram", "tiktok", "linkedin", "youtube"],
            "hardware": ["processor", "chip", "gpu", "cpu", "semiconductor", "hardware"]
        }
        
        for category, keywords in tech_categories.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)
        
        return categories

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for LLaMA"""
        if not text:
            return ""
        
        # Remove HTML entities
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation but keep structure
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']+', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()

    def filter_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles based on quality criteria for LLaMA training"""
        filtered = []
        
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "")
            
            # Skip if too short or too long
            if len(content) < 150 or len(content) > 4000:
                continue
            
            # Skip if title is too short
            if len(title) < 15:
                continue
            
            # Clean the content
            article["content"] = self.clean_text(content)
            article["title"] = self.clean_text(title)
            
            # Skip if cleaning made it too short
            if len(article["content"]) < 150:
                continue
                
            # Skip duplicate or very similar titles
            if any(abs(len(title) - len(existing["title"])) < 10 and 
                   title[:20] == existing["title"][:20] for existing in filtered):
                continue
            
            filtered.append(article)
        
        return filtered

    def create_llama_conversations(self, articles: List[Dict]) -> List[Dict]:
        """Create conversational pairs optimized for LLaMA training"""
        conversations = []
        
        # Question templates for different conversation styles
        question_templates = {
            "general": [
                "What's the latest in tech news?",
                "Can you tell me about recent technology developments?",
                "What interesting tech news have you heard recently?",
                "Share some recent tech updates with me",
                "What's happening in the technology world?"
            ],
            "specific": [
                "Tell me about {topic}",
                "What do you know about {topic}?",
                "Can you explain {topic} to me?",
                "What are the latest developments in {topic}?",
                "I'm interested in {topic}, what can you tell me?"
            ],
            "category": [
                "What's new in {category}?",
                "Any recent {category} news?",
                "Tell me about {category} developments",
                "What's happening with {category} lately?",
                "Share some {category} updates"
            ],
            "analytical": [
                "What do you think about this: {title}",
                "Can you analyze this tech news: {title}",
                "What are the implications of {title}?",
                "How significant is this: {title}?",
                "What's your take on {title}?"
            ]
        }
        
        for article in articles:
            title = article["title"]
            content = article["content"]
            categories = article.get("categories", [])
            source = article["source"]
            
            # Create summary for responses
            summary = content[:800] + "..." if len(content) > 800 else content
            
            # General tech news conversations
            for template in random.sample(question_templates["general"], 2):
                conversations.append({
                    "system": random.choice(self.system_prompts),
                    "user": template,
                    "assistant": f"Here's an interesting development: {title}. {summary}",
                    "source": source,
                    "categories": categories,
                    "type": "general_news"
                })
            
            # Category-specific conversations
            for category in categories[:2]:
                template = random.choice(question_templates["category"])
                conversations.append({
                    "system": random.choice(self.system_prompts),
                    "user": template.format(category=category.replace("_", " ")),
                    "assistant": f"In {category.replace('_', ' ')}: {title}. {summary}",
                    "source": source,
                    "categories": categories,
                    "type": f"{category}_news"
                })
            
            # Specific topic conversations
            key_terms = self.extract_key_terms(title)
            if key_terms:
                topic = random.choice(key_terms)
                template = random.choice(question_templates["specific"])
                conversations.append({
                    "system": random.choice(self.system_prompts),
                    "user": template.format(topic=topic),
                    "assistant": f"Regarding {topic}: {title}. {summary}",
                    "source": source,
                    "categories": categories,
                    "type": "specific_topic"
                })
            
            # Analytical conversations
            template = random.choice(question_templates["analytical"])
            conversations.append({
                "system": random.choice(self.system_prompts),
                "user": template.format(title=title),
                "assistant": f"This is quite significant. {summary} This development could have important implications for the tech industry.",
                "source": source,
                "categories": categories,
                "type": "analysis"
            })
        
        return conversations

    def extract_key_terms(self, title: str) -> List[str]:
        """Extract key terms from title for topic-specific conversations"""
        stop_words = {"the", "a", "an", "and", "or", "but", "for", "with", "to", "from", "in", "on", "at"}
        words = [word.strip(".,!?;:") for word in title.lower().split()]
        key_terms = [word for word in words if len(word) > 3 and word not in stop_words]
        return key_terms[:3]

    def format_for_llama_training(self, conversations: List[Dict]) -> List[Dict]:
        """Format conversations for LLaMA fine-tuning"""
        formatted_data = []
        
        for conv in conversations:
            # ChatML format for LLaMA-2-Chat
            formatted_conv = {
                "messages": [
                    {"role": "system", "content": conv["system"]},
                    {"role": "user", "content": conv["user"]},
                    {"role": "assistant", "content": conv["assistant"]}
                ],
                "source": conv["source"],
                "categories": conv.get("categories", []),
                "type": conv["type"]
            }
            formatted_data.append(formatted_conv)
            
            # Also create Alpaca format for compatibility
            alpaca_format = {
                "instruction": conv["user"],
                "input": "",
                "output": conv["assistant"],
                "system": conv["system"],
                "source": conv["source"],
                "categories": conv.get("categories", [])
            }
            formatted_data.append(alpaca_format)
        
        return formatted_data

    def create_jsonl_format(self, conversations: List[Dict]) -> List[str]:
        """Create JSONL format for LLaMA training"""
        jsonl_lines = []
        
        for conv in conversations:
            jsonl_lines.append(json.dumps(conv, ensure_ascii=False))
        
        return jsonl_lines

    def fetch_all_sources(self, max_articles_per_source: int = 50) -> List[Dict]:
        """Fetch articles from all configured sources"""
        all_articles = []
        
        for source, feed_url in self.rss_feeds.items():
            articles = self.fetch_rss_articles(source, feed_url, max_articles_per_source)
            all_articles.extend(articles)
            
            # Be respectful between sources
            time.sleep(1)
        
        return all_articles

    def create_train_val_split(self, data: List[Dict], timestamp: str, val_ratio: float = 0.1):
        """Create training and validation splits"""
        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_ratio))
        
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Save splits to blob storage
        self.save_data(train_data, f"train_data_{timestamp}.json", blob_folder="training")
        self.save_data(val_data, f"val_data_{timestamp}.json", blob_folder="training")
        
        # Save as JSONL too
        train_jsonl = [json.dumps(item, ensure_ascii=False) for item in train_data]
        val_jsonl = [json.dumps(item, ensure_ascii=False) for item in val_data]
        
        self.save_data(train_jsonl, f"train_data_{timestamp}.jsonl", blob_folder="training", as_jsonl=True)
        self.save_data(val_jsonl, f"val_data_{timestamp}.jsonl", blob_folder="training", as_jsonl=True)
        
        logger.info(f"Created train/val split: {len(train_data)} train, {len(val_data)} val")

    def run_collection(self, max_articles_per_source: int = 30):
        """Run the complete data collection pipeline"""
        logger.info("Starting tech news data collection for Azure ML...")
        
        # Fetch raw articles
        raw_articles = self.fetch_all_sources(max_articles_per_source)
        
        if not raw_articles:
            logger.error("No articles fetched!")
            return
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_data(raw_articles, f"raw_articles_{timestamp}.json", blob_folder="raw")
        
        # Filter and clean articles
        filtered_articles = self.filter_articles(raw_articles)
        logger.info(f"Filtered to {len(filtered_articles)} quality articles")
        
        # Create LLaMA-optimized conversations
        conversations = self.create_llama_conversations(filtered_articles)
        logger.info(f"Created {len(conversations)} conversation examples")
        
        # Format for LLaMA training
        formatted_data = self.format_for_llama_training(conversations)
        logger.info(f"Formatted {len(formatted_data)} training examples")
        
        # Create JSONL format
        jsonl_data = self.create_jsonl_format(formatted_data)
        
        # Save all processed data to blob storage
        self.save_data(filtered_articles, f"filtered_articles_{timestamp}.json", blob_folder="processed")
        self.save_data(conversations, f"conversations_{timestamp}.json", blob_folder="processed")
        self.save_data(formatted_data, f"llama_training_data_{timestamp}.json", blob_folder="training")
        self.save_data(jsonl_data, f"llama_training_data_{timestamp}.jsonl", blob_folder="training", as_jsonl=True)
        
        # Create training split
        self.create_train_val_split(formatted_data, timestamp)
        
        logger.info("Azure ML data collection completed successfully!")
        
        return {
            "raw_articles": len(raw_articles),
            "filtered_articles": len(filtered_articles), 
            "conversations": len(conversations),
            "training_examples": len(formatted_data),
            "timestamp": timestamp,
            "blob_container": self.container_name
        }

def main():
    """Main execution function"""
    # Initialize with Azure connection (will try environment variable if not provided)
    fetcher = AzureBlobTechNewsFetcher()
    
    # You can modify these parameters
    max_articles_per_source = 25
    
    try:
        results = fetcher.run_collection(max_articles_per_source)
        print(f"Azure ML data collection complete! Results: {results}")
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {str(e)}")

if __name__ == "__main__":
    main()