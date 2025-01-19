from crewai import Task

from crewai import Task

class ResearchCrewTasks:

    def research_task(self, agent, inputs):
        return Task(
            agent=agent,
            description=f" Based {inputs} figure out what it is the specific industry, the name of 
            the company related to that industry, it's key offerings and startegic focus ares,  
            check https://www.thetoolbus.ai/ai-tools, 
            and https://appsumo.com/collections/new/ 
            for relevant tools that could be usefull.
            Do, depth of Market Research and Competitor Analysis.",
            expected_output=f"A clear explanation of the Market Research on the Industry, 
            related company, key offerings, strategic focus ares, a clear vision and product information
            on the industry."
            )


    def analysis_task(self, agent, context):
        return Task(
            agent=agent,
            context=context,
            description=f"Evaluate the following report: {context}. 
            Based on the findings on the industry conducted, analyze industry trends and standards
            within the company's sector related to Artificial Intelligence, Machine Learning and Automation.
            Do, depth of Market Research and Competitor Analysis.",
            expected_output=f"analysis of industry trends and standards
            within the company's sector related to Artificial Intelligence, Machine Learning and Automation.
            Propose relevent use cases where the company can leverage GenAI, LLMs,
            ML technologies to improve their processes, enhance customer satisfaction 
            and boost operational efficiency."
            )


    def writing_task(self, agent, context, inputs):
        return Task(
            agent=agent,
            context=context,
            description=f"Answer the users inquiry about 
            their requested industry: {inputs} Given the following 
            analysis on the company and strategic area {context}, using web search, web scraping,
            give 5 proposed GenAI solutions that the company can apply on it's starategic focus area to improve
            their processes, enhance customer satisfaction and boost operational efficiency. 
            With a short overview of each one and how it can be leveraged by the company, 
            5 most relevent resourse asset links and their URL, 
            give top 5 links and URLs of the relevent dataset on platforms like Kaggle, HuggingFace and Github if applicable.
            ",
            expected_output=f" 5 proposed GenAI solutions that the company can apply to improve 
            their processes, enhance customer satisfaction and boost operational efficiency.
            Save the resource links fetched in a Markdown file.  
            Ensure to add references through which certain use cases were suggested,  
            Resource Asset links and their URLs should be clickable.",
            )





