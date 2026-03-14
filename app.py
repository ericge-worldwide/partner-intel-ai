import streamlit as st
import os
import textwrap 
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI # <-- NEW: For the Chatbot
from fpdf import FPDF
from datetime import datetime

# --- 1. CUSTOM TOOL WRAPPER ---
class CustomSearchTool(BaseTool):
    name: str = "search_tool"
    description: str = "Searches the internet for information about a company or person."

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

def create_pdf(report_text, target_name):
    safe_report = sanitize_text(report_text)
    safe_target = sanitize_text(target_name)
    
    pdf = IntelligenceReport()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Subject: {safe_target}", ln=True)
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

# Initialize Session State Variables
if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "report_target" not in st.session_state:
    st.session_state.report_target = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.header("🔑 API Keys")
    google_key = st.text_input("Gemini API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

st.title("🕵️‍♂️ Business Partner Due Diligence")

target_name = st.text_input("Target Name (Company or Person) *", placeholder="e.g., Apple Inc. or John Smith")

with st.expander("⚙️ Optional Search Criteria (Recommended for People)"):
    location = st.text_input("Location (State/Country)", placeholder="e.g., New York, NY")
    industry = st.text_input("Industry or Specialty", placeholder="e.g., Real Estate Attorney")
    extra_context = st.text_area("Additional Context", placeholder="e.g., Former partner at XYZ Law Firm")

if st.button("Start AI Investigation"):
    if not google_key or not tavily_key:
        st.error("Please provide both API keys in the sidebar.")
    elif not target_name:
        st.warning("Please enter a Target Name to begin.")
    else:
        # Clear previous chat history if starting a new search
        st.session_state.chat_messages = []
        
        search_context = target_name
        if location:
            search_context += f", located in {location}"
        if industry:
            search_context += f", specializing in {industry}"
        if extra_context:
            search_context += f". Additional details: {extra_context}"

        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        
        gemini_model_string = "gemini/gemini-2.5-flash"
        search_tool = CustomSearchTool()
        scrape_tool = ScrapeWebsiteTool() 

        with st.status("🕵️ Agents are collaborating...", expanded=True) as status:
            
            investigator = Agent(
                role='OSINT Lead',
                goal='Uncover public history and background for {company_name}',
                backstory='Specialist in digital footprinting.',
                tools=[search_tool, scrape_tool], 
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )
            
            legal_auditor = Agent(
                role='Legal Compliance Specialist',
                goal='Find lawsuits or regulatory issues for {company_name}',
                backstory='Expert in court filings and regulatory compliance.',
                tools=[search_tool, scrape_tool], 
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )

            financial_analyst = Agent(
                role='Corporate Financial Analyst',
                goal='Find revenue figures, funding rounds, and financial health indicators for {company_name}',
                backstory='Wall Street veteran who specializes in reading the financial health of public and private entities.',
                tools=[search_tool, scrape_tool], 
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )
            
            risk_manager = Agent(
                role='Senior Risk Analyst',
                goal='Create a final 1-10 risk score and executive summary using OSINT, Legal, and Financial data.',
                backstory='Veteran consultant who compiles intelligence into clear business red flags.',
                llm=gemini_model_string, 
                verbose=True, 
                allow_delegation=False
            )

            t1 = Task(
                description="Gather background, location, and general news for {company_name}. If you find a good link, scrape it.",
                expected_output="A summary of the subject's background.",
                agent=investigator
            )
            
            t2 = Task(
                description="Search for litigation, patents, or regulatory fines involving {company_name}. Scrape legal articles for details.",
                expected_output="A report detailing legal red flags.",
                agent=legal_auditor
            )

            t3 = Task(
                description="Search for recent revenue, funding, or financial instability regarding {company_name}. Scrape relevant press releases.",
                expected_output="A brief report on the subject's financial footprint.",
                agent=financial_analyst
            )
            
            t4 = Task(
                description="Combine the OSINT background, legal history, and financial data into a comprehensive report with a 1-10 Risk Score.",
                expected_output="A finalized executive summary including a risk score.",
                agent=risk_manager
            )

            crew = Crew(
                agents=[investigator, legal_auditor, financial_analyst, risk_manager],
                tasks=[t1, t2, t3, t4],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff(inputs={'company_name': search_context})
            
            # --- THE FIX: Save the results to memory ---
            st.session_state.report_result = str(result)
            st.session_state.report_target = target_name
            status.update(label="Investigation Complete!", state="complete")

# --- DISPLAY THE REPORT & CHATBOT ---
if st.session_state.report_result:
    st.markdown("---")
    st.subheader("Final Intelligence Report")
    st.markdown(st.session_state.report_result)
    
    pdf_bytes = create_pdf(st.session_state.report_result, st.session_state.report_target)
    st.download_button(
        label="📥 Download PDF Report", 
        data=bytes(pdf_bytes), 
        file_name=f"Report_{sanitize_text(st.session_state.report_target).replace(' ', '_')}.pdf", 
        mime="application/pdf"
    )

    # --- THE CHATBOT UI ---
    st.markdown("---")
    st.subheader("💬 Ask the Report")
    st.caption("Ask specific questions about the data uncovered above.")

    # Display past chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # The Chat Input Box
    if user_question := st.chat_input("E.g., What was the outcome of the lawsuit in 2019?"):
        if not google_key:
            st.error("Please enter your Gemini API Key in the sidebar to use the chat.")
        else:
            # 1. Display user message
            st.chat_message("user").markdown(user_question)
            st.session_state.chat_messages.append({"role": "user", "content": user_question})

            # 2. Ask Gemini the question using the report as context
            chat_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_key)
            prompt = f"Context (Intelligence Report):\n{st.session_state.report_result}\n\nUser Question: {user_question}\n\nPlease answer the question using ONLY the provided context."
            
            with st.spinner("Analyzing report..."):
                response = chat_llm.invoke(prompt)
                
            # 3. Display AI answer
            st.chat_message("assistant").markdown(response.content)
            st.session_state.chat_messages.append({"role": "assistant", "content": response.content})
