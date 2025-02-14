import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()
#class chain
class Chain:
    def __init__(self):
        self.chat = ChatGroq(temperature=0, groq_api_key="Your Key", model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            
            ### INSTRUCTION:
            Extract job postings and return them in **valid JSON format** with the following keys:
            - `role`: The job title.
            - `experience`: Years of experience required as an integer.
            - `skills`: A list of required skills.
            - `description`: A **detailed** job description (at least 3 sentences).

            **Ensure the JSON is properly formatted and provide a complete description.**
            Only return valid JSON without extra text.
            """
        )
        chain_extract = prompt_extract | self.chat
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
        {job_description}
        
            ### INSTRUCTION:
            You are Shri, a business development executive at Infosys. Infosys is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Infosys 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Infosys's portfolio: {link_list}
            Remember you are Shri, BDE at Infosys. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
        
        """
        )
        chain_email = prompt_email | self.chat
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))