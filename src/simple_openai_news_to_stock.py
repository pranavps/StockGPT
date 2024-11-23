# # Latest code 0 use this
# # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_API.ipynb
# # https://cookbook.openai.com/
# import openai

# import requests
# from textblob import TextBlob
# import yfinance as yf

# # # Example text for sentiment analysis
# text = "RBI MPC October After US Federal Reserve reduced rates last month eyes Reserve Bank India RBI A rate cut could signal end rate era fixed deposits The RBI Monetary Policy Committee announcement October provide clarity rate change What best strategy fixed deposits"

# # Sentiment Analysis
# analysis = TextBlob(text)
# sentiment = analysis.sentiment

# # Sector Identification
# sectors = {
#     'banks': ['SBI.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',],
#     'auto': ['MARUTI.NS' 'ASHOKLEY.NS', 'TVSMOTOR.NS'],
#     'infrastructure': ['LT.NS', 'BEL.NS']
# }

# # Stock Recommendations based on sentiment
# if sentiment.polarity > 0:
#     recommended_stocks = sectors['banks']  # Example
# else:
#     recommended_stocks = sectors['auto']  # Example

# # Fetching Stock Data
# def get_stock_info(tickers):
#     stock_info = {}
#     for ticker in tickers:
#         stock = yf.Ticker(ticker)
#         stock_info[ticker] = stock.info
#     return stock_info

# # Example API Call to get stock data
# stocks_info = get_stock_info(recommended_stocks)

# print(stocks_info)


# # openai.api_version = "v1"
# # response = openai.Completion.create(
# #     engine="text-davinci-002", 
# #     prompt="Analyze this text for sentiment: " + text,
# #     max_tokens=50
# # )
# # model = openai.Model("text-davinci-002")
# # prompt = "Analyze this text for sentiment: " + text
# # response = model.completion(prompt=prompt, max_tokens=50)

# # sentiment_result = response.choices[0].text
# # =================================
# # Function to get stock recommendations based on sentiment
# def get_stock_recommendations(sentiment_score):
#     # response = client.completions.create(model="gpt-4-turbo",
#     #     # engine="text-davinci-002", 
#     #     prompt="Analyze this text for sentiment: " + text,
#     #     max_tokens=50
#     # )
#     # response = client.chat.completions.create(
#     #     model="gpt-4o-mini",
#     #     prompt="Analyze this text for sentiment: " + text,
#     #     max_tokens=50
#     # )
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",  # e.g. gpt-35-instant
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Analyze this text for sentiment: " + text,
#             },
#         ],
#     )
#     recommendations = response.choices[0]['message']['content'].strip()
#     return recommendations

# # Example usage
# sentiment_score = 0.7  # Assume some sentiment score here
# recommendations = get_stock_recommendations(sentiment_score)
# print("Stock Recommendations:", recommendations)

# ========= Working sanity test with OpenAI prompt engineering ==========
# from openai import OpenAI

# # Example text for sentiment analysis
# text = "RBI MPC October After US Federal Reserve reduced rates last month eyes Reserve Bank India RBI A rate cut could signal end rate era fixed deposits The RBI Monetary Policy Committee announcement October provide clarity rate change What best strategy fixed deposits"


# stream = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Analyze this text for sentiment: " + text}],
#     stream=True,
# )
# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")
# ===================
import ollama
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU

# Example function to generate a response from the model
def generate_response(prompt):
    response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
    print (response['message']['content'])

    #  -------------- Working stream mode chat ----------------
    # stream = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}], stream=True)
    #     #  (model='llama3.2:3b', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}, stream=True])
    # for chunk in stream:
    #     print(chunk['message']['content'], end='', flush=True)
    # ---------------------
    do_sentiment = input("---\nDo you want to run sentiment analysis for this news? (Y/N)").upper()
    if do_sentiment == "Y":
        prompt = f"Analyze this text for sentiment: " + response['message']['content']
        sentiment = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
        print("--------------------------")
        print(f"Sentiment Analysis Result from news: {sentiment['message']['content']}")
        print("--------------------------")
        return sentiment
    else:
        print("Thank you for using the sentiment analysis tool. Goodbye!")

# text = "RBI MPC October After US Federal Reserve reduced rates last month eyes Reserve Bank India RBI A rate cut could signal end rate era fixed deposits The RBI Monetary Policy Committee announcement October provide clarity rate change What best strategy fixed deposits"
text = input("Enter your NEWS link: ")
print("Analysing sentiment of NEWS: " + text)
# Test the function
# print(generate_response("Tell me about machine learning."))
# print(generate_response("Analyze this text for sentiment: " + text))
print(generate_response("Analyze NEWS from this link" + text))
# Stocktitan news source - get basic information like What is the current stock price of SOBR Safe (SOBR)?
# The current stock price of SOBR Safe (SOBR) is $8 as of October 24, 2024.

# What is the market cap of SOBR Safe (SOBR)?
# The market cap of SOBR Safe (SOBR) is approximately 18.7M.

# What is SOBR Safe, Inc. known for?
# SOBR Safe specializes in touch-based identity verification and alcohol detection systems.

# Where is SOBR Safe, Inc. headquartered?
# The company is headquartered in Boulder, Colorado.

# How does SOBR Safe generate revenue?
# SOBR Safe generates revenue through the sale of cloud-based software solutions, hardware devices, and data reporting services.

# What are the applications of SOBR Safe's technology?
# The technology has applications in commercial fleet management, school buses, manufacturing facilities, and warehousing.

# What is the focus of SOBR Safe's technology?
# SOBR Safe focuses on providing statistical and measurable user data with its scalable technology.

