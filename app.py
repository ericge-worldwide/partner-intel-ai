import streamlit as st
import os
import re
import textwrap
import threading
import time
import requests
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from fpdf import FPDF
from datetime import datetime, timedelta
from unidecode import unidecode

# ============================================================
# 1. TOOL DEFINITIONS
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
        scraper = ScrapeWebsiteTool()
        try:
            raw = scraper.run(website_url=website_url)
        except Exception as e:
            return f"[SCRAPE FAILED] Could not read {website_url}: {str(e)}"

        if not raw or len(raw.strip()) < 50:
            return "[SCRAPE FAILED] Page returned no usable content."

        noise_patterns = [
            r'(?i)(home|about us|contact us|privacy policy|disclaimer|terms of service'
            r'|cookie policy|sitemap|sign up|log in|subscribe|newsletter'
            r'|follow us|share this|back to top|all rights reserved'
            r'|copyright \d{4}|©)[^\n]{0,80}\n?',
            r'(?i)(call|phone|fax|toll.free|consultation)\s*:?\s*[\d\-\(\)\+\. ]{7,20}',
            r'(?m)(^.{2,40}\n){5,}',
            r'(?i)(name\s*\*|email\s*\*|phone\s*\*|message\s*|please prove you are human'
            r'|request a consultation|areas of expertise:|i have read and agree)',
            r'(?i)(we look forward to|our team of experienced|are at your service'
            r'|please provide us with|a lawyer will be in touch'
            r'|website design|lawyer seo|hey ai learn about us)',
        ]

        cleaned = raw
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)

        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()

        max_chars = 8000
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "\n\n[TRUNCATED — content exceeded limit]"

        if len(cleaned) < 50:
            return "[SCRAPE FAILED] Page contained only boilerplate with no substantive content."

        return cleaned


class SECEdgarTool(BaseTool):
    name: str = "sec_edgar_search"
    description: str = (
        "Searches the SEC EDGAR database for official filings (10-K, 10-Q, 8-K, etc.) "
        "for a given company. Returns filing titles, dates, and direct links."
    )

    def _run(self, company_name: str) -> str:
        try:
            url = "https://efts.sec.gov/LATEST/search-index"
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
            params = {
                "q": company_name,
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "forms": "10-K,10-Q,8-K,DEF 14A",
            }
            headers = {"User-Agent": "DueDiligenceBot/1.0 (research@example.com)"}
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code != 200:
                return f"[SEC EDGAR] Search returned status {resp.status_code}. Trying fallback."

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                return f"[SEC EDGAR] No filings found for '{company_name}' in the past 3 years."

            results = [f"SEC EDGAR Results for '{company_name}' (last 3 years):\n"]
            for hit in hits[:10]:
                src = hit.get("_source", {})
                form = src.get("form_type", "N/A")
                filed = src.get("file_date", "N/A")
                entity = src.get("entity_name", "N/A")
                desc = src.get("display_names", [""])[0] if src.get("display_names") else ""
                file_num = src.get("file_num", [""])[0] if src.get("file_num") else ""
                results.append(
                    f"- Form {form} | Filed: {filed} | Entity: {entity} | "
                    f"File#: {file_num} | {desc}"
                )
            return "\n".join(results)

        except requests.exceptions.Timeout:
            return "[SEC EDGAR] Request timed out. SEC servers may be slow."
        except Exception as e:
            return f"[SEC EDGAR] Error: {str(e)}"


class CourtListenerTool(BaseTool):
    name: str = "court_listener_search"
    description: str = (
        "Searches CourtListener's database of federal court opinions and dockets. "
        "Returns case names, courts, dates, and summaries."
    )

    def _run(self, query: str) -> str:
        try:
            url = "https://www.courtlistener.com/api/rest/v3/search/"
            params = {"q": query, "type": "o", "format": "json"}
            headers = {"User-Agent": "DueDiligenceBot/1.0 (research@example.com)"}
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:
                return "[COURTLISTENER] Rate limited. Try again in a moment."
            if resp.status_code != 200:
                return f"[COURTLISTENER] Search returned status {resp.status_code}."

            data = resp.json()
            results_list = data.get("results", [])

            if not results_list:
                return f"[COURTLISTENER] No federal court opinions found for '{query}'."

            results = [f"CourtListener Results for '{query}':\n"]
            for case in results_list[:8]:
                name = case.get("caseName", "N/A")
                court = case.get("court", "N/A")
                date = case.get("dateFiled", "N/A")
                status = case.get("status", "")
                snippet = case.get("snippet", "")
                snippet_clean = re.sub(r'<[^>]+>', '', snippet)[:200]
                abs_url = case.get("absolute_url", "")
                link = f"https://www.courtlistener.com{abs_url}" if abs_url else ""
                results.append(
                    f"- {name}\n  Court: {court} | Filed: {date} | Status: {status}\n"
                    f"  Summary: {snippet_clean}\n  Link: {link}\n"
                )
            return "\n".join(results)

        except requests.exceptions.Timeout:
            return "[COURTLISTENER] Request timed out."
        except Exception as e:
            return f"[COURTLISTENER] Error: {str(e)}"


class OpenCorporatesTool(BaseTool):
    name: str = "opencorporates_search"
    description: str = (
        "Searches the OpenCorporates global corporate registry for company details. "
        "Returns official registration info: jurisdiction, status, incorporation date."
    )

    def _run(self, company_name: str) -> str:
        try:
            url = "https://api.opencorporates.com/v0.4/companies/search"
            params = {"q": company_name, "format": "json"}
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 429:
                return "[OPENCORPORATES] Rate limited. Free tier may be exhausted."
            if resp.status_code != 200:
                return f"[OPENCORPORATES] Search returned status {resp.status_code}."

            data = resp.json()
            companies = data.get("results", {}).get("companies", [])

            if not companies:
                return f"[OPENCORPORATES] No registry records found for '{company_name}'."

            results = [f"OpenCorporates Registry Results for '{company_name}':\n"]
            for entry in companies[:5]:
                co = entry.get("company", {})
                name = co.get("name", "N/A")
                jurisdiction = co.get("jurisdiction_code", "N/A")
                status = co.get("current_status", "N/A")
                inc_date = co.get("incorporation_date", "N/A")
                co_type = co.get("company_type", "N/A")
                addr = co.get("registered_address_in_full", "N/A")
                oc_url = co.get("opencorporates_url", "")
                results.append(
                    f"- {name}\n  Jurisdiction: {jurisdiction} | Status: {status}\n"
                    f"  Incorporated: {inc_date} | Type: {co_type}\n"
                    f"  Address: {addr}\n  Link: {oc_url}\n"
                )
            return "\n".join(results)

        except requests.exceptions.Timeout:
            return "[OPENCORPORATES] Request timed out."
        except Exception as e:
            return f"[OPENCORPORATES] Error: {str(e)}"


class OFACSanctionsTool(BaseTool):
    name: str = "ofac_sanctions_search"
    description: str = (
        "Searches the US Treasury OFAC Sanctions list (SDN) to check "
        "if a person or company appears on any US sanctions lists."
    )

    def _run(self, name: str) -> str:
        try:
            url = "https://search.ofac.treas.gov/OpenData/JSON/SDN.json"
            headers = {"User-Agent": "DueDiligenceBot/1.0 (research@example.com)"}
            resp = requests.get(url, headers=headers, timeout=20)

            if resp.status_code != 200:
                return (
                    f"[OFAC] Could not access SDN list directly (status {resp.status_code}). "
                    f"Use the search_tool to query: 'OFAC sanctions {name}' as a fallback."
                )

            sdn_data = resp.json()
            name_lower = name.lower()
            matches = []

            for entry in sdn_data.get("sdnEntry", []):
                entry_name = (entry.get("lastName", "") + " " + entry.get("firstName", "")).strip()
                if not entry_name:
                    entry_name = entry.get("lastName", "")
                if name_lower in entry_name.lower():
                    sdn_type = entry.get("sdnType", "N/A")
                    programs = ", ".join(
                        [p.get("program", "") for p in entry.get("programList", {}).get("program", [])]
                    ) if entry.get("programList") else "N/A"
                    remarks = entry.get("remarks", "N/A")
                    matches.append(
                        f"- {entry_name} | Type: {sdn_type} | Programs: {programs} | Remarks: {remarks}"
                    )

            if matches:
                return f"⚠️ OFAC SDN MATCHES FOUND for '{name}':\n" + "\n".join(matches[:10])
            else:
                return f"[OFAC] No matches found for '{name}' on the SDN list. CLEAR."

        except requests.exceptions.Timeout:
            return f"[OFAC] SDN list download timed out. Use search_tool for 'OFAC sanctions {name}' instead."
        except Exception as e:
            return f"[OFAC] Error: {str(e)}. Use search_tool for 'OFAC sanctions {name}' instead."


class PeopleRecordsTool(BaseTool):
    name: str = "people_records_search"
    description: str = (
        "Performs targeted searches for an individual across public records databases. "
        "Input format: 'Name | Location1, Location2' where locations are all known jurisdictions."
    )

    def _run(self, query: str) -> str:
        search = TavilySearchResults(k=5)

        if "|" in query:
            parts = query.split("|", 1)
            name = parts[0].strip()
            locations = [l.strip() for l in parts[1].split(",") if l.strip()]
        else:
            name = query.strip()
            locations = []

        all_results = []

        # Professional licenses per jurisdiction
        license_results = []
        if locations:
            for loc in locations:
                try:
                    r = search.run(f'"{name}" {loc} professional license OR licensed OR certification OR bar admission')
                    if r and str(r).strip() != "[]":
                        license_results.append(f"  [{loc}]: {r}")
                except Exception as e:
                    license_results.append(f"  [{loc}]: Search failed: {str(e)}")
        else:
            try:
                r = search.run(f'"{name}" professional license OR licensed OR certification')
                if r and str(r).strip() != "[]":
                    license_results.append(f"  {r}")
            except Exception as e:
                license_results.append(f"  Search failed: {str(e)}")

        all_results.append(
            f"\n### PROFESSIONAL LICENSES & CERTIFICATIONS\n"
            + ("\n".join(license_results) if license_results else "No results found in any jurisdiction.")
        )

        # FEC donations (federal)
        try:
            r = search.run(f'"{name}" FEC political donation OR campaign contribution')
            if r and str(r).strip() != "[]":
                all_results.append(f"\n### POLITICAL DONATIONS (FEC)\n{r}")
            else:
                all_results.append(f"\n### POLITICAL DONATIONS (FEC)\nNo results found.")
        except Exception as e:
            all_results.append(f"\n### POLITICAL DONATIONS (FEC)\nSearch failed: {str(e)}")

        # Disciplinary actions per jurisdiction
        disc_results = []
        if locations:
            for loc in locations:
                try:
                    r = search.run(f'"{name}" {loc} disciplinary action OR suspended OR revoked OR sanctioned')
                    if r and str(r).strip() != "[]":
                        disc_results.append(f"  [{loc}]: {r}")
                except Exception as e:
                    disc_results.append(f"  [{loc}]: Search failed: {str(e)}")
        else:
            try:
                r = search.run(f'"{name}" disciplinary action OR suspended OR revoked OR sanctioned')
                if r and str(r).strip() != "[]":
                    disc_results.append(f"  {r}")
            except Exception as e:
                disc_results.append(f"  Search failed: {str(e)}")

        all_results.append(
            f"\n### DISCIPLINARY ACTIONS\n"
            + ("\n".join(disc_results) if disc_results else "No results found in any jurisdiction.")
        )

        # Bankruptcy, liens, judgments
        lien_results = []
        search_locs = locations if locations else [""]
        for loc in search_locs:
            try:
                loc_query = f" {loc}" if loc else ""
                r = search.run(f'"{name}"{loc_query} bankruptcy OR lien OR judgment filed')
                if r and str(r).strip() != "[]":
                    label = f"  [{loc}]: " if loc else "  "
                    lien_results.append(f"{label}{r}")
            except Exception as e:
                label = f"  [{loc}]: " if loc else "  "
                lien_results.append(f"{label}Search failed: {str(e)}")

        all_results.append(
            f"\n### BANKRUPTCY, LIENS & JUDGMENTS\n"
            + ("\n".join(lien_results) if lien_results else "No results found.")
        )

        return "\n".join(all_results)


class IdentityResolutionTool(BaseTool):
    name: str = "identity_resolver"
    description: str = (
        "Establishes the correct identity of a person by cross-referencing their name "
        "with known employers, affiliations, and locations. Run this FIRST. "
        "Input format: 'Name | affiliation1, affiliation2 | location1, location2'"
    )

    def _run(self, query: str) -> str:
        search = TavilySearchResults(k=5)
        parts = [p.strip() for p in query.split("|")]
        name = parts[0] if len(parts) > 0 else query
        affiliations = [a.strip() for a in parts[1].split(",") if a.strip()] if len(parts) > 1 else []
        locations = [l.strip() for l in parts[2].split(",") if l.strip()] if len(parts) > 2 else []

        results = [f"IDENTITY RESOLUTION for '{name}':\n"]

        for aff in affiliations:
            try:
                r = search.run(f'"{name}" "{aff}"')
                if r and str(r).strip() != "[]":
                    results.append(f"\n[{name} + {aff}]:\n{r}")
            except Exception as e:
                results.append(f"\n[{name} + {aff}]: Search failed: {str(e)}")

        if affiliations:
            linkedin_query = f'"{name}" {affiliations[0]} LinkedIn'
        else:
            linkedin_query = f'"{name}" LinkedIn profile'
        try:
            r = search.run(linkedin_query)
            if r and str(r).strip() != "[]":
                results.append(f"\n[LinkedIn Search]:\n{r}")
        except Exception as e:
            results.append(f"\n[LinkedIn Search]: Failed: {str(e)}")

        if affiliations and locations:
            combined = f'"{name}" {" OR ".join(f""""{a}""" for a in affiliations[:3])} {locations[0]}'
            try:
                r = search.run(combined)
                if r and str(r).strip() != "[]":
                    results.append(f"\n[Combined Cross-Reference]:\n{r}")
            except Exception as e:
                results.append(f"\n[Combined]: Failed: {str(e)}")

        if len(results) <= 1:
            results.append(
                "No identity-confirming results found. Proceed with caution — "
                "findings may relate to a different person with the same name."
            )

        return "\n".join(results)


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
[A concise 3-5 paragraph narrative synthesizing ALL findings. Lead with the most
critical findings. End with a clear recommendation.]

## KEY FINDINGS
[Bullet list of the 3-5 most important facts uncovered, each prefixed
with a 🔴 (critical), 🟡 (notable), or 🟢 (positive) indicator.]
"""


# ============================================================
# 3. ENHANCED PDF GENERATOR
# ============================================================
class IntelligenceReport(FPDF):
    def header(self):
        # Dark header bar
        self.set_fill_color(30, 40, 60)
        self.rect(0, 0, 210, 18, 'F')
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        self.cell(0, 10, "CONFIDENTIAL: PARTNER DUE DILIGENCE REPORT", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)

    def section_header(self, text):
        """Render a styled section header with a colored underline."""
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 40, 60)
        safe = sanitize_text(text.replace("#", "").strip())
        self.cell(0, 8, safe, ln=True)
        # Underline
        self.set_draw_color(50, 100, 180)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def subsection_header(self, text):
        """Render a subsection header."""
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 70, 100)
        safe = sanitize_text(text.replace("#", "").strip())
        self.cell(0, 7, safe, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def risk_badge(self, score_text, tier_text):
        """Render a colored risk score badge."""
        try:
            score = float(re.search(r'(\d+\.?\d*)', score_text).group(1))
        except (AttributeError, ValueError):
            score = 5.0

        if score <= 3.0:
            r, g, b = 40, 160, 70   # green
        elif score <= 5.0:
            r, g, b = 220, 180, 30  # yellow
        elif score <= 7.0:
            r, g, b = 230, 130, 30  # orange
        else:
            r, g, b = 200, 50, 50   # red

        self.ln(4)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        badge_text = sanitize_text(f"  RISK SCORE: {score:.1f} / 10  ")
        self.cell(self.get_string_width(badge_text) + 10, 12, badge_text, fill=True, ln=True)
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(r, g, b)
        self.cell(0, 7, sanitize_text(tier_text), ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def bullet_item(self, text):
        """Render a bullet point with proper wrapping."""
        self.set_font("Helvetica", size=10)
        safe = sanitize_text(text.strip().lstrip("-•* "))
        # Indicator colors
        if safe.startswith("🔴") or safe.startswith("[RED]"):
            self.set_text_color(200, 50, 50)
            prefix = ">> "
            safe = safe.replace("🔴", "").replace("[RED]", "").strip()
        elif safe.startswith("🟡") or safe.startswith("[YEL]"):
            self.set_text_color(180, 140, 20)
            prefix = ">> "
            safe = safe.replace("🟡", "").replace("[YEL]", "").strip()
        elif safe.startswith("🟢") or safe.startswith("[GRN]"):
            self.set_text_color(40, 140, 60)
            prefix = ">> "
            safe = safe.replace("🟢", "").replace("[GRN]", "").strip()
        else:
            self.set_text_color(0, 0, 0)
            prefix = "- "

        wrapped = textwrap.fill(f"{prefix}{safe}", width=82, subsequent_indent="    ", break_long_words=True)
        self.multi_cell(0, 6, wrapped)
        self.ln(1)
        self.set_text_color(0, 0, 0)


def sanitize_text(text):
    if text is None:
        return ""
    return unidecode(str(text))


def create_pdf(report_text, target_name):
    safe_report = sanitize_text(report_text)
    safe_target = sanitize_text(target_name)

    pdf = IntelligenceReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 40, 60)
    pdf.cell(0, 10, f"Subject: {safe_target}", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Separator
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Extract risk score and tier for badge (if present)
    score_match = re.search(r'WEIGHTED RISK SCORE[:\s]*(\d+\.?\d*)\s*/\s*10', safe_report, re.IGNORECASE)
    tier_match = re.search(r'RISK TIER[:\s\n]*(.*?)(?:\n\n|\n##)', safe_report, re.IGNORECASE | re.DOTALL)
    if not tier_match:
        tier_match = re.search(r'(LOW RISK|MODERATE RISK|ELEVATED RISK|HIGH RISK)[^\n]*', safe_report, re.IGNORECASE)

    if score_match:
        score_text = score_match.group(1)
        tier_text = tier_match.group(1).strip() if tier_match else ""
        # Clean tier text to one line
        for tier_label in ["HIGH RISK", "ELEVATED RISK", "MODERATE RISK", "LOW RISK"]:
            if tier_label in tier_text.upper():
                idx = tier_text.upper().index(tier_label)
                end = tier_text.find("\n", idx)
                tier_text = tier_text[idx:end].strip() if end != -1 else tier_text[idx:].strip()
                break
        pdf.risk_badge(score_text, tier_text)

    # Parse and render body
    pdf.set_font("Helvetica", size=10)
    in_bullet_section = False

    for line in safe_report.split('\n'):
        stripped = line.strip()
        if stripped == "" or stripped == "---":
            pdf.ln(3)
            in_bullet_section = False
            continue

        # H1 headers
        if line.startswith("# ") and not line.startswith("##"):
            pdf.section_header(line)
            in_bullet_section = False
            continue

        # H2 headers
        if line.startswith("## "):
            pdf.section_header(line)
            in_bullet_section = False
            continue

        # H3 headers
        if line.startswith("### "):
            # Skip the risk score/tier lines we already rendered as badge
            if any(kw in stripped.upper() for kw in ["WEIGHTED RISK SCORE", "RISK TIER"]):
                continue
            pdf.subsection_header(line)
            in_bullet_section = False
            continue

        # Bullet points
        if stripped.startswith(("-", "•", "*")) and len(stripped) > 2:
            pdf.bullet_item(stripped)
            in_bullet_section = True
            continue

        # Lines starting with emoji indicators
        if any(stripped.startswith(ind) for ind in ["🔴", "🟡", "🟢", "[RED]", "[YEL]", "[GRN]"]):
            pdf.bullet_item(stripped)
            in_bullet_section = True
            continue

        # Bold lines (**text**)
        bold_match = re.match(r'^\*\*(.*?)\*\*(.*)$', stripped)
        if bold_match:
            pdf.set_font("Helvetica", "B", 10)
            try:
                pdf.multi_cell(0, 6, sanitize_text(f"{bold_match.group(1)}{bold_match.group(2)}"))
            except Exception:
                pass
            pdf.set_font("Helvetica", size=10)
            continue

        # Normal text
        in_bullet_section = False
        pdf.set_font("Helvetica", size=10)
        safe_line = textwrap.fill(stripped, width=90, break_long_words=True)
        try:
            pdf.multi_cell(0, 6, safe_line)
        except Exception:
            pass

    return pdf.output()


# ============================================================
# 4. CHAT RETRIEVAL HELPER
# ============================================================
def chunk_report(report_text):
    """Split a report into labeled sections for targeted retrieval."""
    sections = {}
    current_label = "Introduction"
    current_lines = []

    for line in report_text.split('\n'):
        if line.startswith("## ") or line.startswith("# "):
            if current_lines:
                sections[current_label] = "\n".join(current_lines)
            current_label = line.replace("#", "").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_label] = "\n".join(current_lines)

    return sections


def retrieve_relevant_sections(question, sections, max_sections=4):
    """Simple keyword-based retrieval to find the most relevant report sections."""
    question_lower = question.lower()
    scored = []

    # Keyword mappings for common question types
    keyword_boost = {
        "risk": ["risk scorecard", "risk score", "risk tier", "key findings", "executive summary"],
        "legal": ["legal", "compliance", "litigation", "court", "lawsuit"],
        "financial": ["financial", "revenue", "funding", "sec", "fiscal"],
        "background": ["osint", "background", "overview", "introduction"],
        "license": ["people records", "license", "certification", "bar"],
        "donation": ["people records", "fec", "donation", "political"],
        "sanction": ["osint", "ofac", "sanction", "sdn"],
        "score": ["risk scorecard", "risk score", "weighted", "risk tier"],
        "summary": ["executive summary", "key findings"],
        "recommend": ["executive summary", "key findings", "risk tier"],
    }

    for section_label, section_text in sections.items():
        score = 0
        label_lower = section_label.lower()
        text_lower = section_text.lower()

        # Direct keyword match in question vs section label
        question_words = set(re.findall(r'\w+', question_lower))
        label_words = set(re.findall(r'\w+', label_lower))
        overlap = question_words & label_words
        score += len(overlap) * 3

        # Check question words against section content
        for word in question_words:
            if len(word) > 3 and word in text_lower:
                score += 1

        # Boost from keyword mappings
        for trigger, boost_labels in keyword_boost.items():
            if trigger in question_lower:
                for bl in boost_labels:
                    if bl in label_lower:
                        score += 5

        # Executive summary and key findings are always somewhat relevant
        if any(kw in label_lower for kw in ["executive summary", "key findings"]):
            score += 1

        scored.append((score, section_label, section_text))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:max_sections]

    # Always include executive summary if it exists and isn't already selected
    summary_labels = [s[1] for s in selected]
    for s_score, s_label, s_text in scored:
        if "executive summary" in s_label.lower() and s_label not in summary_labels:
            selected.append((s_score, s_label, s_text))
            break

    context = "\n\n---\n\n".join(
        [f"## {label}\n{text}" for _, label, text in selected]
    )
    return context


# ============================================================
# 5. PARALLEL AGENT RUNNER WITH RETRY
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
            is_retryable = any(
                code in err_str for code in ["503", "429", "UNAVAILABLE", "rate", "quota", "overloaded"]
            )
            if is_retryable and attempt < max_retries:
                wait_time = (attempt + 1) * 15
                progress_dict[key] = f"retry ({attempt + 1}/{max_retries})"
                time.sleep(wait_time)
            else:
                results_dict[key] = f"[AGENT ERROR] {key} failed: {err_str}"
                progress_dict[key] = "error"
                return


# ============================================================
# 6. STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="Partner Intel AI", page_icon="🕵️‍♂️", layout="wide")

# --- Session state initialization ---
if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "report_target" not in st.session_state:
    st.session_state.report_target = None
if "report_sections" not in st.session_state:
    st.session_state.report_sections = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "comms_drafts" not in st.session_state:
    st.session_state.comms_drafts = None
if "investigation_history" not in st.session_state:
    st.session_state.investigation_history = []

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 API Keys")
    google_key = st.text_input("Gemini API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")


# --- Main content ---
st.title("🕵️‍♂️ Business Partner Due Diligence")

# --- Investigation History & Comparison (top of page) ---
if st.session_state.investigation_history:
    with st.expander(f"📁 Investigation History ({len(st.session_state.investigation_history)} reports)", expanded=False):
        for i, past in enumerate(reversed(st.session_state.investigation_history)):
            risk_preview = ""
            score_m = re.search(r'WEIGHTED RISK SCORE[:\s]*(\d+\.?\d*)', past.get("report", ""), re.IGNORECASE)
            if score_m:
                s = float(score_m.group(1))
                if s <= 3.0:
                    risk_preview = "🟢"
                elif s <= 5.0:
                    risk_preview = "🟡"
                elif s <= 7.0:
                    risk_preview = "🟠"
                else:
                    risk_preview = "🔴"
                risk_preview = f" {risk_preview} Risk: {s:.1f}/10"

            col_info, col_load, col_dl = st.columns([4, 1, 1])
            with col_info:
                st.markdown(f"**{past['target']}**{risk_preview} — {past['date']}")
            with col_load:
                if st.button("Load", key=f"load_{i}"):
                    st.session_state.report_result = past["report"]
                    st.session_state.report_target = past["target"]
                    st.session_state.report_sections = chunk_report(past["report"])
                    st.session_state.comms_drafts = past.get("comms")
                    st.session_state.chat_messages = []
                    st.rerun()
            with col_dl:
                pdf_bytes_hist = create_pdf(past["report"], past["target"])
                st.download_button(
                    "PDF",
                    data=bytes(pdf_bytes_hist),
                    file_name=f"Report_{sanitize_text(past['target']).replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key=f"dl_{i}",
                )

        st.markdown("---")

        # Comparison tool
        if len(st.session_state.investigation_history) >= 2:
            st.markdown("#### ⚖️ Compare Two Subjects")
            compare_options = [f"{h['target']} ({h['date']})" for h in st.session_state.investigation_history]
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                compare_a = st.selectbox("Subject A", compare_options, index=0)
            with comp_col2:
                compare_b = st.selectbox("Subject B", compare_options, index=min(1, len(compare_options) - 1))

            if st.button("Run Comparison"):
                def get_hist_entry(label):
                    for h in st.session_state.investigation_history:
                        if f"{h['target']} ({h['date']})" == label:
                            return h
                    return None

                hist_a = get_hist_entry(compare_a)
                hist_b = get_hist_entry(compare_b)

                if hist_a and hist_b:
                    def extract_scores(report):
                        categories = [
                            "Litigation History", "Financial Stability", "Regulatory Compliance",
                            "Reputation", "Corporate Governance", "Transparency",
                        ]
                        scores = {}
                        for cat in categories:
                            m = re.search(rf'{cat}.*?Score[:\s]*(\d+\.?\d*)\s*/\s*10', report, re.IGNORECASE)
                            scores[cat] = float(m.group(1)) if m else 0.0
                        wm = re.search(r'WEIGHTED RISK SCORE[:\s]*(\d+\.?\d*)', report, re.IGNORECASE)
                        scores["OVERALL"] = float(wm.group(1)) if wm else 0.0
                        return scores

                    scores_a = extract_scores(hist_a["report"])
                    scores_b = extract_scores(hist_b["report"])

                    ov1, ov2 = st.columns(2)
                    with ov1:
                        st.metric(hist_a["target"], f"{scores_a['OVERALL']:.1f} / 10")
                    with ov2:
                        st.metric(hist_b["target"], f"{scores_b['OVERALL']:.1f} / 10")

                    for cat in ["Litigation History", "Financial Stability", "Regulatory Compliance",
                                "Reputation", "Corporate Governance", "Transparency"]:
                        c1, c2, c3 = st.columns([1, 2, 1])
                        sa, sb = scores_a[cat], scores_b[cat]
                        with c1:
                            clr = "🟢" if sa <= 3 else ("🟡" if sa <= 5 else ("🟠" if sa <= 7 else "🔴"))
                            st.markdown(f"{clr} **{sa:.1f}**")
                        with c2:
                            st.markdown(f"**{cat}**")
                        with c3:
                            clr = "🟢" if sb <= 3 else ("🟡" if sb <= 5 else ("🟠" if sb <= 7 else "🔴"))
                            st.markdown(f"{clr} **{sb:.1f}**")

        st.markdown("---")
        if st.button("🗑️ Clear All History"):
            st.session_state.investigation_history = []
            st.rerun()

# --- New Investigation form ---

# --- Target input ---
target_type = st.radio(
    "What are you investigating?",
    ["🏢 Company / Organization", "👤 Individual Person"],
    horizontal=True,
)
is_person = "Person" in target_type

target_name = st.text_input(
    "Target Name *",
    placeholder="e.g., John Smith" if is_person else "e.g., Apple Inc.",
)

# Initialize all person-specific variables with defaults
residence_location = ""
work_location = ""
license_location = ""
current_employer = ""
former_employers = ""
known_affiliations = []
all_locations = []
location = ""
industry = ""
extra_context = ""

with st.expander("⚙️ Optional Search Criteria" + (" (Recommended)" if is_person else ""), expanded=is_person):
    if is_person:
        st.caption(
            "💡 **Tip:** People often live, work, and hold licenses in different states. "
            "Fill in as many as you know — the AI will search all jurisdictions."
        )
        loc_col1, loc_col2, loc_col3 = st.columns(3)
        with loc_col1:
            residence_location = st.text_input(
                "Residence Location",
                placeholder="e.g., Westfield, NJ",
                help="Where the person lives or has lived",
            )
        with loc_col2:
            work_location = st.text_input(
                "Work Location",
                placeholder="e.g., New York, NY",
                help="Where the person works or has offices",
            )
        with loc_col3:
            license_location = st.text_input(
                "License / Bar Jurisdiction",
                placeholder="e.g., New Jersey, Pennsylvania",
                help="State(s) where they hold professional licenses. Separate multiple with commas.",
            )
        all_locations = [loc.strip() for loc in [residence_location, work_location, license_location] if loc.strip()]
        expanded_locations = []
        for loc in all_locations:
            expanded_locations.extend([l.strip() for l in loc.split(",") if l.strip()])
        all_locations = list(dict.fromkeys(expanded_locations))
        location = ", ".join(all_locations) if all_locations else ""

        industry = st.text_input(
            "Profession or Industry",
            placeholder="e.g., Real Estate Attorney",
        )

        st.markdown("**Known Affiliations** (critical for finding the right person)")
        aff_col1, aff_col2 = st.columns(2)
        with aff_col1:
            current_employer = st.text_input(
                "Current Employer / Company",
                placeholder="e.g., Manifest",
                help="The person's current employer or company",
            )
        with aff_col2:
            former_employers = st.text_input(
                "Former Employers (comma-separated)",
                placeholder="e.g., EY Law, Microsoft",
                help="Previous companies or firms. Separate multiple with commas.",
            )
        known_affiliations = []
        if current_employer.strip():
            known_affiliations.append(current_employer.strip())
        if former_employers.strip():
            known_affiliations.extend([e.strip() for e in former_employers.split(",") if e.strip()])
    else:
        location = st.text_input("Location (State/Country)", placeholder="e.g., New York, NY")
        all_locations = [location.strip()] if location.strip() else []
        industry = st.text_input(
            "Industry or Specialty",
            placeholder="e.g., Fintech",
        )

    extra_context = st.text_area(
        "Additional Context",
        placeholder=(
            "e.g., Former partner at XYZ Law Firm, also known as J. Smith"
            if is_person
            else "e.g., Subsidiary of XYZ Holdings"
        ),
    )

st.markdown("### 🗄️ Deep Dive Databases")
st.caption("These use dedicated API integrations for authoritative results.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    use_sec = st.checkbox("🏛️ SEC Filings", help="10-K, 10-Q, 8-K from EDGAR")
with col2:
    use_courts = st.checkbox("⚖️ Federal Courts", help="CourtListener opinions & dockets")
with col3:
    use_registry = st.checkbox("🏢 Corporate Registry", help="OpenCorporates global registry")
with col4:
    use_ofac = st.checkbox("🚫 OFAC Sanctions", help="US Treasury SDN sanctions list")

if is_person:
    use_people_records = st.checkbox(
        "🔍 People Records (Licenses, FEC Donations, Disciplinary Actions, Liens)",
        value=True,
        help="Targeted searches across professional license boards, FEC, and public record databases",
    )
else:
    use_people_records = False

deep_dive_active = use_sec or use_courts or use_registry or use_ofac or use_people_records
if deep_dive_active:
    st.info(
        "⏱️ **Extended Wait Time:** Deep dive databases involve direct API calls to government "
        "and legal databases. Investigation may take up to **5 minutes**."
    )

if st.button("Start AI Investigation"):
    if not google_key or not tavily_key:
        st.error("Please provide both API keys in the sidebar.")
    elif not target_name:
        st.warning("Please enter a Target Name to begin.")
    else:
        st.session_state.chat_messages = []

        search_context = target_name
        if is_person and any([residence_location, work_location, license_location]):
            loc_parts = []
            if residence_location:
                loc_parts.append(f"resides in {residence_location}")
            if work_location:
                loc_parts.append(f"works in {work_location}")
            if license_location:
                loc_parts.append(f"licensed/registered in {license_location}")
            search_context += f" ({'; '.join(loc_parts)})"
        elif location:
            search_context += f", located in {location}"
        if is_person and known_affiliations:
            if current_employer:
                search_context += f", currently at {current_employer}"
            if former_employers.strip():
                search_context += f", formerly at {former_employers.strip()}"
        if industry:
            search_context += f", specializing in {industry}"
        if extra_context:
            search_context += f". Additional details: {extra_context}"

        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key

        gemini_model_string = "gemini/gemini-2.5-flash"
        search_tool = CustomSearchTool()
        scrape_tool = CleanScrapeWebsiteTool()
        current_date_str = datetime.now().strftime("%B %d, %Y")
        inputs = {"company_name": search_context, "current_date": current_date_str}

        # Build tool sets per agent
        osint_tools = [search_tool, scrape_tool]
        legal_tools = [search_tool, scrape_tool]
        financial_tools = [search_tool, scrape_tool]

        if is_person and known_affiliations:
            osint_tools.append(IdentityResolutionTool())
        if use_registry:
            osint_tools.append(OpenCorporatesTool())
        if use_ofac:
            osint_tools.append(OFACSanctionsTool())
        if use_courts:
            legal_tools.append(CourtListenerTool())
        if use_sec:
            financial_tools.append(SECEdgarTool())

        # Progress tracking
        progress_placeholder = st.empty()

        agent_labels = {
            "osint": "OSINT Background",
            "legal": "Legal & Compliance",
            "financial": "Financial Analysis",
        }
        if is_person and use_people_records:
            agent_labels["people"] = "People Records"
        agent_labels["risk"] = "Risk Assessment"
        agent_labels["comms"] = "Communications Drafts"

        def render_progress(progress_dict):
            icons = {"pending": "⏳", "running": "🔄", "complete": "✅", "error": "❌"}
            md = "### 🕵️ Investigation Progress\n\n"
            for key, label in agent_labels.items():
                status = progress_dict.get(key, "pending")
                icon = icons.get(status, "🔁")
                md += f"{icon}  **{label}** — {status.upper()}\n\n"
            return md

        progress = {k: "pending" for k in agent_labels}
        progress_placeholder.markdown(render_progress(progress))

        # Build agents
        investigator = Agent(
            role="OSINT Lead",
            goal="Uncover public history and background for {company_name}",
            backstory=(
                "Specialist in digital footprinting and open-source intelligence. "
                "You always summarize findings in your own words — never paste raw website content."
            ),
            tools=osint_tools,
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        legal_auditor = Agent(
            role="Legal Compliance Specialist",
            goal="Find lawsuits or regulatory issues for {company_name}",
            backstory=(
                "Expert in court filings, regulatory compliance, and legal risk analysis. "
                "You always summarize findings in your own words — never paste raw website content."
            ),
            tools=legal_tools,
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        financial_analyst = Agent(
            role="Corporate Financial Analyst",
            goal="Find revenue figures, funding rounds, and financial health indicators for {company_name}",
            backstory=(
                "Wall Street veteran who reads the financial health of entities. "
                "You always summarize findings in your own words — never paste raw website content."
            ),
            tools=financial_tools,
            llm=gemini_model_string,
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
        )

        # Build task descriptions
        t1_desc = (
            "Gather background, location, and general news for {company_name}. "
            "If you find a relevant link, scrape it for details. "
            "IMPORTANT: SUMMARIZE all findings in your own words. "
            "NEVER copy-paste raw website text, navigation, or boilerplate. "
            "Only include substantive findings. Translate foreign findings to English."
        )
        if is_person and known_affiliations:
            aff_str = ", ".join(known_affiliations)
            loc_str = ", ".join(all_locations) if all_locations else "unknown"
            resolver_input = f"{target_name} | {aff_str} | {loc_str}"
            t1_desc = (
                f"STEP 1 — IDENTITY RESOLUTION (do this FIRST):\n"
                f"You have the 'identity_resolver' tool. Use it immediately with this exact input:\n"
                f"'{resolver_input}'\n"
                f"This will cross-reference the name with known employers to find the RIGHT person.\n\n"
                f"Known affiliations for this person:\n"
            )
            if current_employer:
                t1_desc += f"- CURRENT: {current_employer}\n"
            if former_employers.strip():
                t1_desc += f"- FORMER: {former_employers.strip()}\n"
            t1_desc += (
                f"\nSTEP 2 — BACKGROUND INVESTIGATION:\n"
                f"Using the identity established above, gather background, location, and "
                f"general news for {{company_name}}.\n"
                f"CRITICAL: When searching, ALWAYS combine the person's name with a known "
                f"employer to disambiguate. For example, search for:\n"
            )
            for aff in known_affiliations[:3]:
                t1_desc += f'- "{target_name}" "{aff}"\n'
            t1_desc += (
                f"\nNEVER search for just the bare name alone — this returns wrong people.\n"
                f"If you find a relevant link, scrape it for details.\n"
                f"SUMMARIZE all findings in your own words. No raw website content."
            )
        if is_person and all_locations:
            t1_desc += (
                f"\n\nMULTI-JURISDICTION NOTICE: This individual may be associated with "
                f"multiple locations: {', '.join(all_locations)}. Search for the subject's "
                f"presence, business activity, and public records in ALL of these locations. "
                f"Note which location each finding relates to."
            )
        if use_registry:
            t1_desc += (
                "\n\nYou have the 'opencorporates_search' tool available. "
                "You MUST use it to look up official corporate registry details."
            )
        if use_ofac:
            t1_desc += (
                "\n\nYou have the 'ofac_sanctions_search' tool available. "
                "You MUST use it to check if the subject appears on any US sanctions lists."
            )

        t2_desc = (
            "Search for litigation, patents, or regulatory fines involving {company_name}. "
            "Scrape legal articles for details. "
            "IMPORTANT: SUMMARIZE all findings in your own words. "
            "Only include case names, dates, outcomes, and legal red flags. "
            "Translate foreign legal documents to English."
        )
        if is_person and known_affiliations:
            t2_desc += (
                f"\n\nIDENTITY DISAMBIGUATION: This is a person search. ALWAYS combine the name "
                f"with a known employer in your searches. Known affiliations:\n"
            )
            if current_employer:
                t2_desc += f"- CURRENT: {current_employer}\n"
            if former_employers.strip():
                t2_desc += f"- FORMER: {former_employers.strip()}\n"
            t2_desc += (
                f"For example: \"{target_name}\" \"{known_affiliations[0]}\" lawsuit\n"
                f"NEVER search for just the bare name plus legal terms."
            )
        if is_person and all_locations:
            t2_desc += (
                f"\n\nMULTI-JURISDICTION NOTICE: This individual may have legal records in "
                f"multiple locations: {', '.join(all_locations)}. Search ALL jurisdictions."
            )
        if use_courts:
            t2_desc += (
                "\n\nYou have the 'court_listener_search' tool available. "
                "You MUST use it to search for federal court opinions and dockets."
            )

        t3_desc = (
            "Search for recent revenue, funding, or financial instability regarding {company_name}. "
            "Scrape relevant press releases. "
            "IMPORTANT: SUMMARIZE all findings in your own words. "
            "Only include revenue figures, funding rounds, and financial red flags. "
            "Translate foreign financial data to English."
        )
        if is_person and known_affiliations:
            t3_desc += (
                f"\n\nIDENTITY DISAMBIGUATION: This is a person search. ALWAYS combine the name "
                f"with known employers:\n"
            )
            if current_employer:
                t3_desc += f"- CURRENT: {current_employer}\n"
            if former_employers.strip():
                t3_desc += f"- FORMER: {former_employers.strip()}\n"
            t3_desc += f"NEVER search for just the bare name alone."
        if use_sec:
            t3_desc += (
                "\n\nYou have the 'sec_edgar_search' tool available. "
                "You MUST use it to find official SEC filings (10-K, 10-Q, 8-K)."
            )

        t1 = Task(
            description=t1_desc,
            expected_output="A concise, well-organized summary in your own words. No raw website text. Written STRICTLY in English.",
            agent=investigator,
        )
        t2 = Task(
            description=t2_desc,
            expected_output="A concise report of legal red flags with case names, dates, and outcomes. No raw website text. Written STRICTLY in English.",
            agent=legal_auditor,
        )
        t3 = Task(
            description=t3_desc,
            expected_output="A concise report on financial footprint with specific figures. No raw website text. Written STRICTLY in English.",
            agent=financial_analyst,
        )

        # People Records agent (conditional)
        people_agent = None
        people_task = None
        if is_person and use_people_records:
            people_loc_str = ", ".join(all_locations) if all_locations else ""
            people_tool_input = f"{target_name} | {people_loc_str}" if people_loc_str else target_name

            people_agent = Agent(
                role="People Intelligence Analyst",
                goal="Find professional licenses, political donations, disciplinary actions, and public records for {company_name}",
                backstory=(
                    "Specialist in individual background checks using professional license "
                    "databases, FEC records, and disciplinary boards. You always summarize "
                    "findings in your own words."
                ),
                tools=[search_tool, PeopleRecordsTool()],
                llm=gemini_model_string,
                verbose=True,
                allow_delegation=False,
                max_execution_time=300,
            )

            people_search_instructions = (
                f"Conduct a thorough individual background search for {{company_name}}.\n\n"
                f"IDENTITY DISAMBIGUATION: This person has known affiliations that MUST be "
                f"used to ensure you find the right individual:\n"
            )
            if current_employer:
                people_search_instructions += f"- CURRENT EMPLOYER: {current_employer}\n"
            if former_employers.strip():
                people_search_instructions += f"- FORMER EMPLOYERS: {former_employers.strip()}\n"
            people_search_instructions += (
                f"\nCRITICAL MULTI-JURISDICTION NOTICE:\n"
                f"This individual may be associated with MULTIPLE locations:\n"
            )
            if residence_location:
                people_search_instructions += f"- RESIDENCE: {residence_location}\n"
            if work_location:
                people_search_instructions += f"- WORK: {work_location}\n"
            if license_location:
                people_search_instructions += f"- LICENSE JURISDICTIONS: {license_location}\n"
            people_search_instructions += (
                f"\nYou MUST search ALL of these jurisdictions, not just one.\n\n"
                f"Use the 'people_records_search' tool with this exact input:\n"
                f"'{people_tool_input}'\n\n"
                f"This will search professional licenses, FEC donations, disciplinary "
                f"actions, and liens/judgments across ALL provided jurisdictions.\n\n"
                f"Also use the search_tool to look for LinkedIn profile details, "
                f"professional associations, board memberships, and news coverage. "
                f"When using search_tool, ALWAYS include a known employer in the query "
                f"to disambiguate, e.g.: \"{target_name}\" \"{known_affiliations[0] if known_affiliations else ''}\" LinkedIn\n\n"
                f"SUMMARIZE all findings in your own words. Clearly indicate WHICH "
                f"jurisdiction each finding comes from."
            )

            people_task = Task(
                description=people_search_instructions,
                expected_output=(
                    "A concise report on the individual's professional background, licenses "
                    "(specifying which jurisdiction), donations, and any flags. Written STRICTLY in English."
                ),
                agent=people_agent,
            )

        # Run research agents in parallel (staggered)
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
        if people_agent and people_task:
            threads.append(
                threading.Thread(
                    target=run_agent_task,
                    args=(people_agent, people_task, inputs, agent_results, "people", progress, 15),
                )
            )

        for t in threads:
            t.start()

        while any(t.is_alive() for t in threads):
            progress_placeholder.markdown(render_progress(progress))
            time.sleep(2)

        for t in threads:
            t.join()

        progress_placeholder.markdown(render_progress(progress))

        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key

        # Risk manager
        progress["risk"] = "running"
        progress_placeholder.markdown(render_progress(progress))

        combined_intel = (
            f"## OSINT BACKGROUND REPORT\n{agent_results.get('osint', 'No data.')}\n\n"
            f"## LEGAL & COMPLIANCE REPORT\n{agent_results.get('legal', 'No data.')}\n\n"
            f"## FINANCIAL ANALYSIS REPORT\n{agent_results.get('financial', 'No data.')}"
        )
        if is_person and use_people_records:
            combined_intel += (
                f"\n\n## PEOPLE RECORDS & BACKGROUND\n{agent_results.get('people', 'No data.')}"
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
            if is_person and use_people_records:
                full_report += (
                    f"\n\n## People Records & Background\n{agent_results.get('people', 'No data.')}"
                )

            st.session_state.report_result = full_report
            st.session_state.report_target = target_name
            st.session_state.report_sections = chunk_report(full_report)
            progress["risk"] = "complete"
            progress_placeholder.markdown(render_progress(progress))

        except Exception as e:
            progress["risk"] = "error"
            progress_placeholder.markdown(render_progress(progress))
            st.error(f"🚨 **Risk assessment failed:** {str(e)}")
            partial = (
                f"# Partial Report: {target_name}\n"
                f"**Date:** {current_date_str}\n\n"
                f"⚠️ Risk scoring failed. Raw findings below.\n\n"
                f"## OSINT Background\n{agent_results.get('osint', 'No data.')}\n\n"
                f"## Legal & Compliance\n{agent_results.get('legal', 'No data.')}\n\n"
                f"## Financial Analysis\n{agent_results.get('financial', 'No data.')}"
            )
            if is_person and use_people_records:
                partial += f"\n\n## People Records\n{agent_results.get('people', 'No data.')}"
            st.session_state.report_result = partial
            st.session_state.report_target = target_name
            st.session_state.report_sections = chunk_report(partial)

        # Communications drafts
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

        # Save to investigation history and rerun so the history expander renders
        st.session_state.investigation_history.append({
            "target": target_name,
            "date": current_date_str,
            "report": st.session_state.report_result,
            "comms": st.session_state.comms_drafts,
        })
        st.rerun()


# ============================================================
# 7. DISPLAY REPORT, COMMS, AND CHATBOT
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

    # --- CHATBOT WITH SMART RETRIEVAL ---
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

            # Smart retrieval: send only relevant sections instead of full report
            sections = st.session_state.report_sections
            if sections:
                relevant_context = retrieve_relevant_sections(user_question, sections)
            else:
                relevant_context = st.session_state.report_result

            # Sliding window: last 10 messages
            recent_history = st.session_state.chat_messages[-10:]
            chat_history_text = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in recent_history[:-1]]
            )

            prompt = (
                f"Context (Relevant sections from Intelligence Report):\n{relevant_context}\n\n"
                f"Past Chat History:\n{chat_history_text}\n\n"
                f"User's New Question: {user_question}\n\n"
                f"Please answer the question using ONLY the provided context. "
                f"If the answer is not in the provided sections, say so clearly."
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
