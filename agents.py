

from crewai import Agent
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq  # Import Groq client
#from langchain_openai import ChatOpenAI
import os
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool, RagTool 


class ResearchCrewAgents:

    def __init__(self):
        # Initialize tools if needed
        self.serper = SerperDevTool()
        self.web = WebsiteSearchTool()
        self.web_scrape=ScrapeWebsiteTool()
        self.rag_datasets=RagTool()


       # OpenAI Models
        #self.gpt3 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        #self.gpt4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)
        #self.gpt3_5_turbo_0125 = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7)
        #self.gpt3_5_turbo_1106 = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.7)
        #self.gpt3_5_turbo_instruct = ChatOpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
        
        # Groq Models
        self.llama3_8b = ChatGroq(temperature=0.7, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-8b-8192")
        self.llama3_70b = ChatGroq(temperature=0.7, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-70b-8192")
        self.mixtral_8x7b = ChatGroq(temperature=0.7, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")
        self.gemma_7b = ChatGroq(temperature=0.7, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="gemma-7b-it")
        
        # CHANGE YOUR MODEL HERE
        self.selected_llm = self.mixtral_8x7b

    def researcher(self):
    # Detailed agent setup for the Research Expert
        return Agent(
        role='Research',
        goal='Search websites and understand the industry and segment the company is working in(e.g. Automotive, Manufacturing, Finance, Retail, Healthcare, etc.).'
            'Identify the company\'s key offerings and strategic focus areas(e.g. operations, supply chain, customer experience, etc.).'
            'A vision and product information on the industry should be fine as well.'
            'Search for relevent datasets on platforms like Kaggle, HuggingFace and GitHub'
            'Do, depth of Market Research and Competitor Analysis.',
        backstory='You are an expert in market research and you have necessary tools to search different websites'
                'Scrape websites and collect ideal and comprehensive data'
                'through critical thinking and systems thinking understand'
                'industry, it\'s segment of working. Identofy the company\'s key offerings and strategic focus areas'
                'you have a clear vision on the comany, it\'s products,'
                'and have good amount of information on it'
                'You are good at doing Market Research and Competitor Analysis.',
        verbose=True,
        allow_delegation=False,
        llm=self.selected_llm,
        max_iter=3,
        tools=[self.serper, self.web, self.web_scrape, self.rag_datasets],
        ) 


    def analyst(self):
        # Detailed agent setup for the Analyst
        return Agent(
        role='Analyst',
        goal='Based on the industry conducted, analyze industry trends and standards'
            'within the company\'s sector related to Artificial Intelligence, Machine Learning and Automation.',
        backstory='You are a talented and skilled analyst who analyze the company, it\'s working segments,' 
                'key offerings, strategic focus areas, vision, product informations, it\'s trends and standards'
                'related to AI, ML and Automation.'  
                'Propose relevent use cases where the company can leverage GenAI, LLMs and ML technologies to improve their processes,'
                'enhance customer satisfaction and boost operational efficiency.'
                'You are good at doing Market Research and Competitor Analysis.',
        verbose=True,
        allow_delegation=False,
        llm=self.selected_llm,
        max_iter=3,
        tools=[self.serper, self.web, self.web_scrape, self.rag_datasets]
        )

    def writer(self):
        # Detailed agent setup for the Writer
        return Agent(
        role='Writer and Composer',
        goal='Use CrewAI tools to search and summarize findings of the previous agents(researcher and analyst).'
            'create a detailed report on the findings of researcher agent and analyser agent.' 
            'A fine and well defined vision and product information on the industry, relevent company,'
            'it\'s key offerings and strategic focus areas, analyze industry trends and standards'
            'within the company\'s sector related to AI, ML and Automation.'
            'Propose applicable solutions related to GenAI, LLMs and ML Technologies like Document search, Automated report generation'
            'and AI-powered chat system for internal or customer facing purposes.'
            'So, that these solutions will improve the company\'s processes enhance customer satisfaction'
            'and boost operational efficiency.',
        backstory='You are organized content creater and talented composer that understands' 
            'what you have to take from the above two agents(researcher and analyst) and how to summarize them'
            'and generate a detailed report on the strategic focus area here the company can leverage'
            'GenAI, LLMs and ML technologies to improve their processes, enhance customer satisfaction'
            'and boost operational efficiency.' 
            'you export great findings, you are great at scraping the web links and resources'
            'to achieve specific goals.'
            'You are good at proposing some applicable GenAI Solutions.',
        verbose=True,
        allow_delegation=False,
        llm=self.selected_llm,
        max_iter=3,
        tools=[self.serper, self.web, self.web_scrape, self.rag_datasets],
        )
    
