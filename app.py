import streamlit as st
import os
import textwrap 
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool # <--- THE NEW HIRE
from langchain_community.tools.tavily_search import TavilySearchResults
from fpdf import FPDF
from datetime import datetime

# --- 1. CUSTOM TOOL WRAPPER ---
class CustomSearchTool(BaseTool):
    name: str = "search_tool"
    description: str = "Searches the internet for information about a company."

    def _run(self, query: str) -> str:
        search = TavilySearchResults(k=10)
        return search.run(query)

# --- 2. PDF GENERATOR ---
class IntelligenceReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "CONFIDENTIAL: PARTNER DUE DILIGENCE", border=False, ln=True, align="C")
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d')} | Page {self.page_no()}", align="C")

def sanitize_text(text):
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def create_pdf(report_text, company_name):
    safe_report = sanitize_text(report_text)
    safe_company = sanitize_text(company_name)
    
    pdf = IntelligenceReport()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Subject: {safe_company}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=11)
    
    for line in safe_report.split('\n'):
        if line.strip() == "":
            pdf.ln(4) 
            continue
            
        if line.startswith("##"):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            safe_header = textwrap.fill(line.replace("#", "").strip(), width=85, break_long_words=True)
            pdf.multi_cell(0, 10, safe_header)
            pdf.set_font("Helvetica", size=11)
        else:
            safe_line = textwrap.fill(line, width=85, break_long_words=True)
            try:
                pdf.multi_cell(0, 8, safe_line)
            except Exception:
                pass 
            
    return pdf.output()

# --- 3. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Partner Intel AI", page_icon="🕵️‍♂️", layout="wide")

with st.sidebar:
    st.header("🔑 API Keys")
    google_key = st.text_input("Gemini API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

st.title("🕵️‍♂️ Business Partner Due Diligence")
company_name = st.text_input("Target Company Name", placeholder="e.g., Apple Inc.")

if st.button("Start AI Investigation"):
    if not google_key or not tavily_key:
        st.error("Please provide both API keys in the sidebar.")
    else:
        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        
        gemini_model_string = "gemini/gemini-2.5-flash"
        
        # --- EQUIP THE TOOLS ---
        search_tool = CustomSearchTool()
        scrape_tool = ScrapeWebsiteTool() # <--- INITIALIZING THE SCRAPER

        with st.status("🕵️ Agents are collaborating...", expanded=True) as status:
            
            # --- AGENT 1: OSINT ---
            investigator = Agent(
                role='OSINT Lead',
                goal='Uncover public history and founders for {company_name}',
                backstory='Specialist in digital footprinting.',
                tools=[search_tool, scrape_tool], # <--- DUAL WIELDING
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )
            
            # --- AGENT 2: LEGAL ---
            legal_auditor = Agent(
                role='Legal Compliance Specialist',
                goal='Find lawsuits or regulatory issues for {company_name}',
                backstory='Expert in court filings and regulatory compliance.',
                tools=[search_tool, scrape_tool], # <--- DUAL WIELDING
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )

            # --- AGENT 3: FINANCIAL ---
            financial_analyst = Agent(
                role='Corporate Financial Analyst',
                goal='Find revenue figures, funding rounds, and financial health indicators for {company_name}',
                backstory='Wall Street veteran who specializes in reading the financial health of public and private companies.',
                tools=[search_tool, scrape_tool], # <--- DUAL WIELDING
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )
            
            # --- AGENT 4: RISK MANAGER ---
            risk_manager = Agent(
                role='Senior Risk Analyst',
                goal='Create a final 1-10 risk score and executive summary using OSINT, Legal, and Financial data.',
                backstory='Veteran consultant who compiles intelligence into clear business red flags.',
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )

            # TASKS
            t1 = Task(
                description="Gather founder names, HQ location, and general news for {company_name}. If you find a good link, scrape it.",
                expected_output="A summary of the company background.",
                agent=investigator
            )
            
            t2 = Task(
                description="Search for litigation, patents, or regulatory fines involving {company_name}. Scrape legal articles for details.",
                expected_output="A report detailing legal red flags.",
                agent=legal_auditor
            )

            t3 = Task(
                description="Search for recent revenue, VC funding rounds, or financial instability regarding {company_name}. Scrape financial press releases.",
                expected_output="A brief report on the company's financial footprint.",
                agent=financial_analyst
            )
            
            t4 = Task(
                description="Combine the OSINT background, legal history, and financial data into a comprehensive report with a 1-10 Risk Score.",
                expected_output="A finalized executive summary including a risk score.",
                agent=risk_manager
            )

            # CREW 
            crew = Crew(
                agents=[investigator, legal_auditor, financial_analyst, risk_manager],
                tasks=[t1, t2, t3, t4],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff(inputs={'company_name': company_name})
            status.update(label="Investigation Complete!", state="complete")

        st.subheader("Final Intelligence Report")
        st.markdown(result)
        
        pdf_bytes = create_pdf(str(result), company_name)
        st.download_button(
            label="📥 Download PDF Report", 
            data=bytes(pdf_bytes), 
            file_name=f"Report_{sanitize_text(company_name).replace(' ', '_')}.pdf", 
            mime="application/pdf"
        )