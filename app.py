import streamlit as st
import os
import textwrap
import threading
import time
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from fpdf import FPDF
from datetime import datetime
from unidecode import unidecode

# ============================================================
# 1. CUSTOM TOOL WRAPPER
# ============================================================
class CustomSearchTool(BaseTool):
    name: str = "search_tool"
    description: str = "Searches the internet for information about a company or person."

    def _run(self, query: str) -> str:
        search = TavilySearchResults(k=10)
        return search.run(query)


class CleanScrapeWebsiteTool(BaseTool):
    name: str = "scrape_website"
    description: str = (
        "Scrapes a website URL and returns only the meaningful content. "
        "Strips out navigation, footers, forms, ads, and boilerplate. "
        "Use this when you find a relevant URL and need to read its content."
    )

    def _run(self, website_url: str) -> str:
        import re
        scraper = ScrapeWebsiteTool()
        try:
            raw = scraper.run(website_url=website_url)
        except Exception as e:
            return f"[SCRAPE FAILED] Could not read {website_url}: {str(e)}"

        if not raw or len(raw.strip()) < 50:
            return "[SCRAPE FAILED] Page returned no usable content."

        # --- Clean boilerplate patterns ---
        noise_patterns = [
            # Navigation / menu fragments
            r'(?i)(home|about us|contact us|privacy policy|disclaimer|terms of service'
            r'|cookie policy|sitemap|sign up|log in|subscribe|newsletter'
            r'|follow us|share this|back to top|all rights reserved'
            r'|copyright \d{4}|©)[^\n]{0,80}\n?',
            # Phone/fax with surrounding boilerplate
            r'(?i)(call|phone|fax|toll.free|consultation)\s*:?\s*[\d\-\(\)\+\. ]{7,20}',
            # Repeated menu-like short lines (e.g., "Business Law\nCorporate Law\nLitigation\n")
            r'(?m)(^.{2,40}\n){5,}',
            # Form fields and labels
            r'(?i)(name\s*\*|email\s*\*|phone\s*\*|message\s*|please prove you are human'
            r'|request a consultation|areas of expertise:|i have read and agree)',
            # SEO / marketing filler
            r'(?i)(we look forward to|our team of experienced|are at your service'
            r'|please provide us with|a lawyer will be in touch'
            r'|website design|lawyer seo|hey ai learn about us)',
        ]

        cleaned = raw
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)

        # Collapse excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()

        # Truncate to a reasonable length (avoid dumping 50-page sites)
        max_chars = 8000
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "\n\n[TRUNCATED — content exceeded limit]"

        if len(cleaned) < 50:
            return "[SCRAPE FAILED] Page contained only boilerplate with no substantive content."

        return cleaned


# ============================================================
# 2. STRUCTURED RISK RUBRIC
# ============================================================
RISK_RUBRIC = """
You MUST evaluate the subject using the structured rubric below. 
For EACH category, assign a score from 1 (no concern) to 10 (critical risk) 
and provide a 1-2 sentence justification referencing specific findings.

| Category               | Weight | What to Evaluate                                              |
|------------------------|--------|---------------------------------------------------------------|
| Litigation History     | 25%    | Active lawsuits, past settlements, pattern of disputes        |
| Financial Stability    | 20%    | Revenue trends, debt levels, funding status, bankruptcy signs |
| Regulatory Compliance  | 20%    | Fines, sanctions, license revocations, consent decrees        |
| Reputation             | 15%    | News sentiment, consumer complaints, public controversies     |
| Corporate Governance   | 10%    | Officer history, entity status, registered agent, turnover    |
| Transparency           | 10%    | Information availability, willingness to disclose, red flags  |

FORMAT YOUR OUTPUT EXACTLY LIKE THIS (do not skip any section):

## RISK SCORECARD

### Litigation History — Score: X/10 (Weight: 25%)
[Justification citing specific findings from the legal report]

### Financial Stability — Score: X/10 (Weight: 20%)
[Justification citing specific findings from the financial report]

### Regulatory Compliance — Score: X/10 (Weight: 20%)
[Justification citing specific findings]

### Reputation — Score: X/10 (Weight: 15%)
[Justification citing specific findings from the OSINT report]

### Corporate Governance — Score: X/10 (Weight: 10%)
[Justification citing specific findings]

### Transparency — Score: X/10 (Weight: 10%)
[Justification citing specific findings]

### WEIGHTED RISK SCORE: X.X / 10
Calculate as: (Litigation * 0.25) + (Financial * 0.20) + (Regulatory * 0.20) + (Reputation * 0.15) + (Governance * 0.10) + (Transparency * 0.10)

### RISK TIER
- 1.0 - 3.0: LOW RISK — Proceed with standard due diligence
- 3.1 - 5.0: MODERATE RISK — Proceed with enhanced monitoring
- 5.1 - 7.0: ELEVATED RISK — Requires senior review before engagement
- 7.1 - 10.0: HIGH RISK — Recommend against engagement without mitigation

## EXECUTIVE SUMMARY
[A concise 3-5 paragraph narrative synthesizing ALL findings from the OSINT, legal, 
and financial investigations. Lead with the most critical findings. End with a 
clear recommendation.]

## KEY FINDINGS
[Bullet list of the 3-5 most important individual facts uncovered, each prefixed 
with a 🔴 (critical), 🟡 (notable), or 🟢 (positive) indicator.]
"""

# ============================================================
# 3. PDF GENERATOR
# ============================================================
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
    if text is None:
        return ""
    return unidecode(str(text))


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


# ============================================================
# 4. PARALLEL AGENT RUNNER
# ============================================================
def run_agent_task(agent, task, inputs, results_dict, key, progress_dict, start_delay=0):
    """Runs a single CrewAI agent/task in a thread with retry logic."""
    if start_delay > 0:
        time.sleep(start_delay)
    progress_dict[key] = "running"
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            mini_crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = mini_crew.kickoff(inputs=inputs)
            results_dict[key] = str(result)
            progress_dict[key] = "complete"
            return
        except Exception as e:
            err_str = str(e)
            is_retryable = any(code in err_str for code in ["503", "429", "UNAVAILABLE", "rate", "quota", "overloaded"])
            if is_retryable and attempt < max_retries:
                wait_time = (attempt + 1) * 15  # 15s, 30s
                progress_dict[key] = f"retry ({attempt + 1}/{max_retries})"
                time.sleep(wait_time)
            else:
                results_dict[key] = f"[AGENT ERROR] {key} failed: {err_str}"
                progress_dict[key] = "error"
                return


# ============================================================
# 5. STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="Partner Intel AI", page_icon="🕵️‍♂️", layout="wide")

if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "report_target" not in st.session_state:
    st.session_state.report_target = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "comms_drafts" not in st.session_state:
    st.session_state.comms_drafts = None

with st.sidebar:
    st.header("🔑 API Keys")
    google_key = st.text_input("Gemini API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

st.title("🕵️‍♂️ Business Partner Due Diligence")

target_name = st.text_input(
    "Target Name (Company or Person) *",
    placeholder="e.g., Apple Inc. or John Smith",
)

with st.expander("⚙️ Optional Search Criteria (Recommended for People)"):
    location = st.text_input("Location (State/Country)", placeholder="e.g., New York, NY")
    industry = st.text_input("Industry or Specialty", placeholder="e.g., Real Estate Attorney")
    extra_context = st.text_area(
        "Additional Context",
        placeholder="e.g., Former partner at XYZ Law Firm",
    )

st.markdown("### 🗄️ Deep Dive Databases (Optional, takes longer)")
col1, col2, col3 = st.columns(3)
with col1:
    use_sec = st.checkbox("🏛️ SEC Financials (10-K/10-Q)")
with col2:
    use_courts = st.checkbox("⚖️ Federal Court Dockets")
with col3:
    use_registry = st.checkbox("🏢 Corporate Registry")

if use_sec or use_courts or use_registry:
    st.info(
        "⏱️ **Extended Wait Time:** Deep dive databases require the AI to read "
        "massive legal and financial documents and navigate government rate limits. "
        "This investigation may take up to **5 minutes**. Please do not refresh "
        "the page while the agents are working."
    )

if st.button("Start AI Investigation"):
    if not google_key or not tavily_key:
        st.error("Please provide both API keys in the sidebar.")
    elif not target_name:
        st.warning("Please enter a Target Name to begin.")
    else:
        st.session_state.chat_messages = []

        search_context = target_name
        if location:
            search_context += f", located in {location}"
        if industry:
            search_context += f", specializing in {industry}"
        if extra_context:
            search_context += f". Additional details: {extra_context}"

        # --- Set ALL env var names that litellm / CrewAI may look for ---
        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key

        gemini_model_string = "gemini/gemini-2.5-flash"
        search_tool = CustomSearchTool()
        scrape_tool = CleanScrapeWebsiteTool()

        current_date_str = datetime.now().strftime("%B %d, %Y")
        inputs = {"company_name": search_context, "current_date": current_date_str}

        # -------------------------------------------------------
        # PHASE 1-A: Per-agent progress + parallel execution
        # -------------------------------------------------------
        progress_placeholder = st.empty()

        def render_progress(progress_dict):
            icons = {"pending": "⏳", "running": "🔄", "complete": "✅", "error": "❌"}
            labels = {
                "osint": "OSINT Background",
                "legal": "Legal & Compliance",
                "financial": "Financial Analysis",
                "risk": "Risk Assessment",
                "comms": "Communications Drafts",
            }
            md = "### 🕵️ Investigation Progress\n\n"
            for key, label in labels.items():
                status = progress_dict.get(key, "pending")
                icon = icons.get(status, "🔁")  # 🔁 for retry states
                md += f"{icon}  **{label}** — {status.upper()}\n\n"
            return md

        progress = {
            "osint": "pending",
            "legal": "pending",
            "financial": "pending",
            "risk": "pending",
            "comms": "pending",
        }
        progress_placeholder.markdown(render_progress(progress))

        # --- Build agents ---
        investigator = Agent(
            role="OSINT Lead",
            goal="Uncover public history and background for {company_name}",
            backstory="Specialist in digital footprinting and open-source intelligence.",
            tools=[search_tool, scrape_tool],
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        legal_auditor = Agent(
            role="Legal Compliance Specialist",
            goal="Find lawsuits or regulatory issues for {company_name}",
            backstory="Expert in court filings, regulatory compliance, and legal risk.",
            tools=[search_tool, scrape_tool],
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        financial_analyst = Agent(
            role="Corporate Financial Analyst",
            goal="Find revenue figures, funding rounds, and financial health indicators for {company_name}",
            backstory="Wall Street veteran who specializes in reading the financial health of entities.",
            tools=[search_tool, scrape_tool],
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        # --- Build task descriptions (with optional deep-dive instructions) ---
        t1_desc = (
            "Gather background, location, and general news for {company_name}. "
            "If you find a relevant link, scrape it for details. "
            "IMPORTANT: When reporting scraped content, SUMMARIZE the key facts in your own words. "
            "NEVER copy-paste raw website text, navigation menus, contact forms, disclaimers, or boilerplate. "
            "Only include substantive findings: names of key people, business activities, locations, "
            "notable events, and any red flags. Translate foreign findings to English."
        )
        t2_desc = (
            "Search for litigation, patents, or regulatory fines involving {company_name}. "
            "Scrape legal articles for details. "
            "IMPORTANT: When reporting scraped content, SUMMARIZE the key legal facts in your own words. "
            "NEVER copy-paste raw website text, navigation menus, or boilerplate. "
            "Only include case names, dates, outcomes, regulatory actions, and legal red flags. "
            "Translate foreign legal documents to English."
        )
        t3_desc = (
            "Search for recent revenue, funding, or financial instability regarding {company_name}. "
            "Scrape relevant press releases for details. "
            "IMPORTANT: When reporting scraped content, SUMMARIZE the key financial facts in your own words. "
            "NEVER copy-paste raw website text, navigation menus, or boilerplate. "
            "Only include revenue figures, funding rounds, debt indicators, and financial red flags. "
            "Translate foreign financial data to English."
        )

        if use_registry:
            t1_desc += (
                " CRITICAL: You MUST also query 'site:opencorporates.com {company_name}' "
                "to find and verify their official corporate registry details."
            )
        if use_courts:
            t2_desc += (
                " CRITICAL: You MUST also query 'site:courtlistener.com {company_name}' "
                "to find specific federal court dockets and opinions."
            )
        if use_sec:
            t3_desc += (
                " CRITICAL: You MUST also query 'site:sec.gov {company_name} 10-K' "
                "to find and summarize their official SEC filings."
            )

        t1 = Task(
            description=t1_desc,
            expected_output="A concise, well-organized summary of the subject's background in your own words. No raw website text or boilerplate. Written STRICTLY in English.",
            agent=investigator,
        )
        t2 = Task(
            description=t2_desc,
            expected_output="A concise report of legal red flags with case names, dates, and outcomes. No raw website text or boilerplate. Written STRICTLY in English.",
            agent=legal_auditor,
        )
        t3 = Task(
            description=t3_desc,
            expected_output="A concise report on the subject's financial footprint with specific figures. No raw website text or boilerplate. Written STRICTLY in English.",
            agent=financial_analyst,
        )

        # --- Run T1, T2, T3 in parallel threads (staggered to avoid rate limits) ---
        agent_results = {}
        threads = [
            threading.Thread(
                target=run_agent_task,
                args=(investigator, t1, inputs, agent_results, "osint", progress, 0),
            ),
            threading.Thread(
                target=run_agent_task,
                args=(legal_auditor, t2, inputs, agent_results, "legal", progress, 5),
            ),
            threading.Thread(
                target=run_agent_task,
                args=(financial_analyst, t3, inputs, agent_results, "financial", progress, 10),
            ),
        ]

        for t in threads:
            t.start()

        # Poll for progress updates while threads are alive
        while any(t.is_alive() for t in threads):
            progress_placeholder.markdown(render_progress(progress))
            time.sleep(2)

        for t in threads:
            t.join()

        progress_placeholder.markdown(render_progress(progress))

        # Re-assert API keys after threads complete (thread safety)
        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key

        # -------------------------------------------------------
        # PHASE 1-B: Risk manager with structured rubric
        # -------------------------------------------------------
        progress["risk"] = "running"
        progress_placeholder.markdown(render_progress(progress))

        combined_intel = (
            f"## OSINT BACKGROUND REPORT\n{agent_results.get('osint', 'No data.')}\n\n"
            f"## LEGAL & COMPLIANCE REPORT\n{agent_results.get('legal', 'No data.')}\n\n"
            f"## FINANCIAL ANALYSIS REPORT\n{agent_results.get('financial', 'No data.')}"
        )

        risk_manager = Agent(
            role="Senior Risk Analyst",
            goal=(
                "Synthesize all intelligence into a structured risk assessment with "
                "category-level scores and a weighted composite score."
            ),
            backstory=(
                "Veteran risk consultant who compiles multi-source intelligence into "
                "clear, actionable business risk assessments using standardized rubrics."
            ),
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        t4 = Task(
            description=(
                f"Today's date is {{current_date}}. Using the intelligence below, "
                f"produce a comprehensive risk assessment.\n\n"
                f"{combined_intel}\n\n"
                f"--- SCORING INSTRUCTIONS ---\n{RISK_RUBRIC}"
            ),
            expected_output=(
                "A finalized risk assessment with the RISK SCORECARD, EXECUTIVE SUMMARY, "
                "and KEY FINDINGS sections, written entirely in professional English."
            ),
            agent=risk_manager,
        )

        try:
            risk_crew = Crew(
                agents=[risk_manager],
                tasks=[t4],
                process=Process.sequential,
                verbose=True,
            )
            risk_result = risk_crew.kickoff(inputs=inputs)

            # Assemble the full report: individual findings + risk assessment
            full_report = (
                f"# Due Diligence Report: {target_name}\n"
                f"**Date:** {current_date_str}\n\n"
                f"---\n\n"
                f"{str(risk_result)}\n\n"
                f"---\n\n"
                f"# DETAILED FINDINGS\n\n"
                f"## OSINT Background\n{agent_results.get('osint', 'No data.')}\n\n"
                f"## Legal & Compliance\n{agent_results.get('legal', 'No data.')}\n\n"
                f"## Financial Analysis\n{agent_results.get('financial', 'No data.')}"
            )

            st.session_state.report_result = full_report
            st.session_state.report_target = target_name
            progress["risk"] = "complete"
            progress_placeholder.markdown(render_progress(progress))

        except Exception as e:
            progress["risk"] = "error"
            progress_placeholder.markdown(render_progress(progress))
            st.error(f"🚨 **Risk assessment failed:** {str(e)}")
            # Still save partial results so the user gets something
            st.session_state.report_result = (
                f"# Partial Report: {target_name}\n"
                f"**Date:** {current_date_str}\n\n"
                f"⚠️ Risk scoring failed. Raw findings below.\n\n"
                f"## OSINT Background\n{agent_results.get('osint', 'No data.')}\n\n"
                f"## Legal & Compliance\n{agent_results.get('legal', 'No data.')}\n\n"
                f"## Financial Analysis\n{agent_results.get('financial', 'No data.')}"
            )
            st.session_state.report_target = target_name

        # -------------------------------------------------------
        # PHASE 1-C: Communications drafts
        # -------------------------------------------------------
        if st.session_state.report_result:
            progress["comms"] = "running"
            progress_placeholder.markdown(render_progress(progress))

            try:
                chat_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=google_key,
                )
                comms_prompt = (
                    "Based on the following due diligence report, draft two items:\n"
                    "1. A professional 3-bullet-point email for an executive summarizing "
                    "key risks and findings. Include a Subject Line.\n"
                    "2. A short Slack message with emojis to alert the team that the "
                    "report is ready, highlighting the risk score.\n\n"
                    "Note: Some sections may contain error messages where an agent failed. "
                    "Acknowledge any gaps in coverage in your drafts.\n\n"
                    f"Report:\n{st.session_state.report_result}"
                )
                comms_response = chat_llm.invoke(comms_prompt)
                st.session_state.comms_drafts = comms_response.content
                progress["comms"] = "complete"
            except Exception as e:
                st.session_state.comms_drafts = None
                progress["comms"] = "error"
                st.warning(f"⚠️ Communications drafts failed (non-critical): {str(e)}")

            progress_placeholder.markdown(render_progress(progress))
        else:
            progress["comms"] = "error"
            progress_placeholder.markdown(render_progress(progress))


# ============================================================
# 6. DISPLAY REPORT, COMMS, AND CHATBOT
# ============================================================
if st.session_state.report_result:
    st.markdown("---")
    st.subheader("Final Intelligence Report")
    st.markdown(st.session_state.report_result)

    pdf_bytes = create_pdf(st.session_state.report_result, st.session_state.report_target)
    st.download_button(
        label="📥 Download PDF Report",
        data=bytes(pdf_bytes),
        file_name=f"Report_{sanitize_text(st.session_state.report_target).replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

    if st.session_state.comms_drafts:
        st.markdown("---")
        with st.expander("📬 Quick Share: Auto-Generated Email & Slack Drafts", expanded=True):
            st.markdown(st.session_state.comms_drafts)

    # --- CHATBOT ---
    st.markdown("---")
    st.subheader("💬 Ask the Report")
    st.caption("Ask specific questions about the data uncovered above.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("E.g., What was the outcome of the lawsuit in 2019?"):
        if not google_key:
            st.error("Please enter your Gemini API Key in the sidebar to use the chat.")
        else:
            st.chat_message("user").markdown(user_question)
            st.session_state.chat_messages.append({"role": "user", "content": user_question})

            # Keep only the last 10 messages to avoid token overflow
            recent_history = st.session_state.chat_messages[-10:]
            chat_history_text = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in recent_history[:-1]]
            )

            prompt = (
                f"Context (Intelligence Report):\n{st.session_state.report_result}\n\n"
                f"Past Chat History:\n{chat_history_text}\n\n"
                f"User's New Question: {user_question}\n\n"
                f"Please answer the question using ONLY the provided context."
            )

            try:
                with st.spinner("Analyzing report..."):
                    chat_llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=google_key,
                    )
                    response = chat_llm.invoke(prompt)

                st.chat_message("assistant").markdown(response.content)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": response.content}
                )

            except Exception as e:
                st.error(
                    f"⚠️ **Chat Error:** The AI encountered a glitch (likely a momentary "
                    f"rate limit). Your report is safe! Try asking again in a few seconds."
                    f"\n\n*Technical Details: {str(e)}*"
                )
