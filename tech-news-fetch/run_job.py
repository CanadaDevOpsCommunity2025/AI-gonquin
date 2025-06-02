from azure_tech_news_fetcher import AzureBlobTechNewsFetcher
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-articles', type=int, default=25)
    parser.add_argument('--container-name', type=str, default='tech-news-data')
    args = parser.parse_args()
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    fetcher = AzureBlobTechNewsFetcher(connection_string=connection_string,container_name=args.container_name,)
    results = fetcher.run_collection(max_articles_per_source=args.max_articles)
    
    print(f"Job completed: {results}")

if __name__ == "__main__":
    main()